#!/usr/bin/env python3
"""
export_hex.py — Export quantized weights, biases, test images, and requant
parameters as .hex files for SystemVerilog $readmemh.

Replicates the C++ quantization pipeline (sw/quantized/cnn.cpp):
  1. Load float32 .bin weights
  2. Calibrate: float forward pass on 100 images → find activation ranges
  3. Quantize: weights → int8, biases → int32
  4. Compute requantization params: multiplier (int32) + shift (int)
  5. Export everything as .hex files
  6. Run quantized inference on N test images → export golden predictions

Usage:
    python3 export_hex.py [--data_dir DIR] [--weights_dir DIR] [--out_dir DIR] [--num_test N]

Output files (in out_dir):
    conv1_weights.hex, conv1_bias.hex, conv2_weights.hex, conv2_bias.hex
    fc1_weights.hex, fc1_bias.hex, fc2_weights.hex, fc2_bias.hex
    requant_params.svh          — SV include file with localparam
    test_images.hex             — N×784 int8 pixel values
    test_expected.hex           — N expected class labels
"""

import numpy as np
import struct
import os
import sys
import argparse

# ═══════════════════════════════════════════════════════════════════════════
# Layer dimensions (must match cnn_pkg.sv and cnn.cpp exactly)
# ═══════════════════════════════════════════════════════════════════════════
C1_IN, H1_IN, W1_IN = 1, 28, 28
C1_OUT, K1 = 4, 3
H1_OUT, W1_OUT = H1_IN - K1 + 1, W1_IN - K1 + 1  # 26, 26
POOL1 = 2
H1P, W1P = H1_OUT // POOL1, W1_OUT // POOL1        # 13, 13

C2_IN, H2_IN, W2_IN = C1_OUT, H1P, W1P
C2_OUT, K2 = 8, 3
H2_OUT, W2_OUT = H2_IN - K2 + 1, W2_IN - K2 + 1    # 11, 11
POOL2 = 2
H2P, W2P = H2_OUT // POOL2, W2_OUT // POOL2          # 5, 5

FC1_IN, FC1_OUT = C2_OUT * H2P * W2P, 64              # 200, 64
FC2_IN, FC2_OUT = FC1_OUT, 10                          # 64, 10

# ═══════════════════════════════════════════════════════════════════════════
# MNIST IDX file readers
# ═══════════════════════════════════════════════════════════════════════════
def read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, f"Bad magic {magic:#x}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 0x00000801, f"Bad magic {magic:#x}"
        return np.frombuffer(f.read(), dtype=np.uint8)

# ═══════════════════════════════════════════════════════════════════════════
# Float forward pass (for calibration) — matches C++ cnn.cpp exactly
# ═══════════════════════════════════════════════════════════════════════════
def conv2d_float(inp, kernel, bias, C_in, H_in, W_in, C_out, K):
    H_out = H_in - K + 1
    W_out = W_in - K + 1
    inp_r = inp.reshape(C_in, H_in, W_in)
    kern_r = kernel.reshape(C_out, C_in, K, K)
    out = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    for n in range(C_out):
        for c in range(C_in):
            for i in range(K):
                for j in range(K):
                    out[n] += inp_r[c, i:H_out+i, j:W_out+j] * kern_r[n, c, i, j]
        out[n] += bias[n]
    return out

def relu_float(data):
    return np.maximum(data, 0.0)

def max_pool2d_float(inp, C, H_in, W_in, pool):
    H_out = H_in // pool
    W_out = W_in // pool
    inp_r = inp.reshape(C, H_in, W_in)
    out = np.zeros((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        for x in range(H_out):
            for y in range(W_out):
                out[c, x, y] = inp_r[c, x*pool:(x+1)*pool, y*pool:(y+1)*pool].max()
    return out

def fc_float(inp, weights, bias, N, M):
    w = weights.reshape(M, N)
    return w @ inp.flatten() + bias

# ═══════════════════════════════════════════════════════════════════════════
# Quantized forward pass — matches C++ conv2d_q / fc_forward_q exactly
# ═══════════════════════════════════════════════════════════════════════════
def compute_requant_params(M_float):
    """Replicate C++ compute_requant_params() from fixed_point.h"""
    if M_float == 0.0:
        return 0, 0
    m = float(M_float)
    n = 0
    while m < 0.5:
        m *= 2.0; n += 1
    while m >= 1.0:
        m /= 2.0; n -= 1
    multiplier = int(m * (1 << 31) + 0.5)
    shift = 31 + n
    return multiplier, shift

def requantize(acc, multiplier, shift, relu=False):
    """Replicate C++ requantize() / requantize_relu()"""
    v = int(acc) * int(multiplier)
    if shift > 0:
        v = (v + (1 << (shift - 1))) >> shift
    if relu:
        v = max(0, min(127, v))
    else:
        v = max(-128, min(127, v))
    return int(v)

def conv2d_q(inp_q, kern_q, bias_q, C_in, H_in, W_in, C_out, K, rq_mult, rq_shift, relu=True):
    H_out = H_in - K + 1
    W_out = W_in - K + 1
    inp_r = inp_q.reshape(C_in, H_in, W_in).astype(np.int32)
    kern_r = kern_q.reshape(C_out, C_in, K, K).astype(np.int32)
    out = np.zeros(C_out * H_out * W_out, dtype=np.int8)
    for n in range(C_out):
        for x in range(H_out):
            for y in range(W_out):
                acc = int(bias_q[n])
                for c in range(C_in):
                    for i in range(K):
                        for j in range(K):
                            acc += int(inp_r[c, x+i, y+j]) * int(kern_r[n, c, i, j])
                out[n * H_out * W_out + x * W_out + y] = requantize(acc, rq_mult, rq_shift, relu)
    return out

def max_pool2d_q(inp_q, C, H_in, W_in, pool):
    H_out = H_in // pool
    W_out = W_in // pool
    inp_r = inp_q.reshape(C, H_in, W_in)
    out = np.zeros(C * H_out * W_out, dtype=np.int8)
    for c in range(C):
        for x in range(H_out):
            for y in range(W_out):
                mv = -128
                for i in range(pool):
                    for j in range(pool):
                        v = int(inp_r[c, x*pool+i, y*pool+j])
                        if v > mv:
                            mv = v
                out[c * H_out * W_out + x * W_out + y] = mv
    return out

def fc_q(inp_q, wt_q, bias_q, N, M, rq_mult, rq_shift, relu=True):
    wt = wt_q.reshape(M, N).astype(np.int32)
    inp = inp_q.flatten().astype(np.int32)
    out = np.zeros(M, dtype=np.int8)
    for i in range(M):
        acc = int(bias_q[i])
        for j in range(N):
            acc += int(wt[i, j]) * int(inp[j])
        out[i] = requantize(acc, rq_mult, rq_shift, relu)
    return out

def fc_q_raw(inp_q, wt_q, bias_q, N, M):
    wt = wt_q.reshape(M, N).astype(np.int32)
    inp = inp_q.flatten().astype(np.int32)
    out = np.zeros(M, dtype=np.int32)
    for i in range(M):
        acc = int(bias_q[i])
        for j in range(N):
            acc += int(wt[i, j]) * int(inp[j])
        out[i] = acc
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Hex export helpers
# ═══════════════════════════════════════════════════════════════════════════
def to_hex8(val):
    """Signed int8 → 2-digit hex (unsigned representation)"""
    return f"{int(val) & 0xFF:02x}"

def to_hex32(val):
    """Signed int32 → 8-digit hex (unsigned representation)"""
    return f"{int(val) & 0xFFFFFFFF:08x}"

def write_hex8(path, arr):
    with open(path, 'w') as f:
        for v in arr:
            f.write(to_hex8(v) + '\n')
    print(f"  {path}: {len(arr)} entries (int8)")

def write_hex32(path, arr):
    with open(path, 'w') as f:
        for v in arr:
            f.write(to_hex32(v) + '\n')
    print(f"  {path}: {len(arr)} entries (int32)")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Export quantized CNN weights and test vectors as .hex")
    parser.add_argument("--data_dir",    default="../../../../sw/common/data")
    parser.add_argument("--weights_dir", default="../../../../sw/common/weights")
    parser.add_argument("--out_dir",     default="../hex")
    parser.add_argument("--num_test",    type=int, default=20, help="Number of test images")
    parser.add_argument("--num_calib",   type=int, default=100, help="Calibration images")
    args = parser.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # ── Load MNIST test data ──
    print("[1/6] Loading MNIST test data...")
    images_raw = read_idx_images(os.path.join(args.data_dir, "t10k-images-idx3-ubyte"))
    labels_raw = read_idx_labels(os.path.join(args.data_dir, "t10k-labels-idx1-ubyte"))
    print(f"  Loaded {len(images_raw)} images")

    # ── Load float weights ──
    print("[2/6] Loading float32 weights...")
    w = args.weights_dir
    conv1_w = np.fromfile(os.path.join(w, "conv1_weights.bin"), dtype=np.float32)
    conv1_b = np.fromfile(os.path.join(w, "conv1_bias.bin"),    dtype=np.float32)
    conv2_w = np.fromfile(os.path.join(w, "conv2_weights.bin"), dtype=np.float32)
    conv2_b = np.fromfile(os.path.join(w, "conv2_bias.bin"),    dtype=np.float32)
    fc1_w   = np.fromfile(os.path.join(w, "fc1_weights.bin"),   dtype=np.float32)
    fc1_b   = np.fromfile(os.path.join(w, "fc1_bias.bin"),      dtype=np.float32)
    fc2_w   = np.fromfile(os.path.join(w, "fc2_weights.bin"),   dtype=np.float32)
    fc2_b   = np.fromfile(os.path.join(w, "fc2_bias.bin"),      dtype=np.float32)
    print(f"  conv1_w={conv1_w.shape}, conv2_w={conv2_w.shape}, fc1_w={fc1_w.shape}, fc2_w={fc2_w.shape}")

    # ── Calibrate: float forward on N images to find activation ranges ──
    print(f"[3/6] Calibrating on {args.num_calib} images...")
    max_conv1 = max_conv2 = max_fc1 = 0.0

    for n in range(min(args.num_calib, len(images_raw))):
        img = images_raw[n].astype(np.float32).flatten() / 255.0

        c1 = conv2d_float(img, conv1_w, conv1_b, C1_IN, H1_IN, W1_IN, C1_OUT, K1)
        c1 = relu_float(c1)
        m = np.max(np.abs(c1))
        if m > max_conv1: max_conv1 = m

        p1 = max_pool2d_float(c1, C1_OUT, H1_OUT, W1_OUT, POOL1)

        c2 = conv2d_float(p1, conv2_w, conv2_b, C2_IN, H2_IN, W2_IN, C2_OUT, K2)
        c2 = relu_float(c2)
        m = np.max(np.abs(c2))
        if m > max_conv2: max_conv2 = m

        p2 = max_pool2d_float(c2, C2_OUT, H2_OUT, W2_OUT, POOL2)

        f1 = fc_float(p2, fc1_w, fc1_b, FC1_IN, FC1_OUT)
        f1 = relu_float(f1)
        m = np.max(np.abs(f1))
        if m > max_fc1: max_fc1 = m

    # ── Compute scales and requant params ──
    scale_input    = 1.0 / 127.0
    scale_conv1_w  = np.max(np.abs(conv1_w)) / 127.0
    scale_conv1_out = max_conv1 / 127.0
    scale_conv2_w  = np.max(np.abs(conv2_w)) / 127.0
    scale_conv2_out = max_conv2 / 127.0
    scale_fc1_w    = np.max(np.abs(fc1_w)) / 127.0
    scale_fc1_out  = max_fc1 / 127.0
    scale_fc2_w    = np.max(np.abs(fc2_w)) / 127.0

    M_conv1 = (scale_conv1_w * scale_input) / scale_conv1_out
    M_conv2 = (scale_conv2_w * scale_conv1_out) / scale_conv2_out
    M_fc1   = (scale_fc1_w * scale_conv2_out) / scale_fc1_out

    rq_conv1_mult, rq_conv1_shift = compute_requant_params(M_conv1)
    rq_conv2_mult, rq_conv2_shift = compute_requant_params(M_conv2)
    rq_fc1_mult,   rq_fc1_shift   = compute_requant_params(M_fc1)

    print(f"  CONV1: M={M_conv1:.6f}, mult={rq_conv1_mult}, shift={rq_conv1_shift}")
    print(f"  CONV2: M={M_conv2:.6f}, mult={rq_conv2_mult}, shift={rq_conv2_shift}")
    print(f"  FC1:   M={M_fc1:.6f}, mult={rq_fc1_mult}, shift={rq_fc1_shift}")

    # ── Quantize weights and biases ──
    print("[4/6] Quantizing weights...")

    def quantize_w(src, scale):
        q = np.round(src / scale).astype(np.int32)
        return np.clip(q, -128, 127).astype(np.int8)

    def quantize_b(src, scale_w, scale_in):
        acc_scale = scale_w * scale_in
        return np.round(src / acc_scale).astype(np.int32)

    conv1_w_q = quantize_w(conv1_w, scale_conv1_w)
    conv2_w_q = quantize_w(conv2_w, scale_conv2_w)
    fc1_w_q   = quantize_w(fc1_w, scale_fc1_w)
    fc2_w_q   = quantize_w(fc2_w, scale_fc2_w)

    conv1_b_q = quantize_b(conv1_b, scale_conv1_w, scale_input)
    conv2_b_q = quantize_b(conv2_b, scale_conv2_w, scale_conv1_out)
    fc1_b_q   = quantize_b(fc1_b, scale_fc1_w, scale_conv2_out)
    fc2_b_q   = quantize_b(fc2_b, scale_fc2_w, scale_fc1_out)

    # ── Export weight/bias hex files ──
    print("[5/6] Writing hex files...")
    write_hex8(os.path.join(out, "conv1_weights.hex"), conv1_w_q)
    write_hex32(os.path.join(out, "conv1_bias.hex"), conv1_b_q)
    write_hex8(os.path.join(out, "conv2_weights.hex"), conv2_w_q)
    write_hex32(os.path.join(out, "conv2_bias.hex"), conv2_b_q)
    write_hex8(os.path.join(out, "fc1_weights.hex"), fc1_w_q)
    write_hex32(os.path.join(out, "fc1_bias.hex"), fc1_b_q)
    write_hex8(os.path.join(out, "fc2_weights.hex"), fc2_w_q)
    write_hex32(os.path.join(out, "fc2_bias.hex"), fc2_b_q)

    # ── Export requant params as SV include ──
    svh_path = os.path.join(out, "requant_params.svh")
    with open(svh_path, 'w') as f:
        f.write("// Auto-generated by export_hex.py — do not edit\n")
        f.write(f"localparam logic signed [31:0] CONV1_RQ_MULT  = 32'sh{rq_conv1_mult & 0xFFFFFFFF:08x};\n")
        f.write(f"localparam logic        [5:0]  CONV1_RQ_SHIFT = 6'd{rq_conv1_shift};\n")
        f.write(f"localparam logic signed [31:0] CONV2_RQ_MULT  = 32'sh{rq_conv2_mult & 0xFFFFFFFF:08x};\n")
        f.write(f"localparam logic        [5:0]  CONV2_RQ_SHIFT = 6'd{rq_conv2_shift};\n")
        f.write(f"localparam logic signed [31:0] FC1_RQ_MULT    = 32'sh{rq_fc1_mult & 0xFFFFFFFF:08x};\n")
        f.write(f"localparam logic        [5:0]  FC1_RQ_SHIFT   = 6'd{rq_fc1_shift};\n")
    print(f"  {svh_path}: requant params")

    # ── Run quantized inference and export test vectors ──
    print(f"[6/6] Running quantized inference on {args.num_test} test images...")
    N_test = min(args.num_test, len(images_raw))

    all_pixels = []
    expected_classes = []
    correct = 0

    for t in range(N_test):
        # Quantize input: uint8 [0,255] → float [0,1] → int8 [0,127]
        img_float = images_raw[t].astype(np.float32).flatten() / 255.0
        img_q = np.round(img_float / scale_input).astype(np.int32)
        img_q = np.clip(img_q, 0, 127).astype(np.int8)

        # Quantized forward pass
        c1 = conv2d_q(img_q, conv1_w_q, conv1_b_q, C1_IN, H1_IN, W1_IN, C1_OUT, K1,
                       rq_conv1_mult, rq_conv1_shift, relu=True)
        p1 = max_pool2d_q(c1, C1_OUT, H1_OUT, W1_OUT, POOL1)
        c2 = conv2d_q(p1, conv2_w_q, conv2_b_q, C2_IN, H2_IN, W2_IN, C2_OUT, K2,
                       rq_conv2_mult, rq_conv2_shift, relu=True)
        p2 = max_pool2d_q(c2, C2_OUT, H2_OUT, W2_OUT, POOL2)
        f1 = fc_q(p2, fc1_w_q, fc1_b_q, FC1_IN, FC1_OUT,
                   rq_fc1_mult, rq_fc1_shift, relu=True)
        f2 = fc_q_raw(f1, fc2_w_q, fc2_b_q, FC2_IN, FC2_OUT)

        pred = int(np.argmax(f2))
        label = int(labels_raw[t])
        if pred == label:
            correct += 1

        all_pixels.extend(img_q.tolist())
        expected_classes.append(pred)

        print(f"  Image {t:3d}: label={label}, predicted={pred} {'OK' if pred == label else 'MISMATCH'}")

    # Write test image hex (flat: N_test * 784 entries)
    write_hex8(os.path.join(out, "test_images.hex"), all_pixels)

    # Write expected classes
    with open(os.path.join(out, "test_expected.hex"), 'w') as f:
        for c in expected_classes:
            f.write(f"{c:02x}\n")
    print(f"  {os.path.join(out, 'test_expected.hex')}: {N_test} entries")

    # Write num_test for TB
    with open(os.path.join(out, "num_test.svh"), 'w') as f:
        f.write(f"// Auto-generated by export_hex.py\n")
        f.write(f"localparam int NUM_TEST_IMAGES = {N_test};\n")

    print(f"\nQuantized accuracy: {correct}/{N_test} = {100.0*correct/N_test:.1f}%")
    print("Done! Hex files written to", out)

if __name__ == "__main__":
    main()
