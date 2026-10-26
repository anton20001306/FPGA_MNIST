#!/usr/bin/env python3
"""
verify_output.py — Compare PyTorch model outputs against the C++ model outputs.

Runs the same N images through:
  1. The PyTorch TinyCNN (using trained weights)
  2. The C++ mnist_cnn binary (using the same exported weights)

Then checks that both produce:
  - The same predicted class for every image
  - Logit values within a small floating-point tolerance

Any mismatch reveals a bug in the C++ layer implementations.

Usage:
    python3 verify_output.py [--n N] [--data_dir DIR] [--weights_dir DIR] [--binary PATH]

Requirements:
    - Weights already exported by train_tiny_cnn.py
    - C++ binary already built (make in sw/float/ or sw/quantized/)
"""

import torch
import torch.nn as nn
import numpy as np
import struct
import subprocess
import os
import argparse
import sys

# Import the same model definition from the training script
sys.path.insert(0, os.path.dirname(__file__))
from train_tiny_cnn import TinyCNN, MNISTDataset

# ---------------------------------------------------------------------------
# Run a single image through PyTorch and return (predicted_class, logits)
# ---------------------------------------------------------------------------
def pytorch_inference(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        x      = image_tensor.unsqueeze(0).to(device)  # [1, 1, 28, 28]
        logits = model(x).squeeze(0).cpu().numpy()      # [10]
    return int(np.argmax(logits)), logits

# ---------------------------------------------------------------------------
# Generate a tiny IDX file for a single image, run the C++ binary,
# and parse its stdout to extract logits and prediction.
# ---------------------------------------------------------------------------
def cpp_inference_batch(binary_path, data_dir, weights_dir, n):
    """
    Runs the C++ binary on the first `n` test images and returns a list of
    (prediction, logits_array) tuples parsed from verbose stdout.
    """
    result = subprocess.run(
        [binary_path,
         os.path.join(data_dir, "t10k-images-idx3-ubyte"),
         os.path.join(data_dir, "t10k-labels-idx1-ubyte"),
         weights_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[verify] C++ binary error:")
        print(result.stderr)
        sys.exit(1)

    # Parse "First 10 predictions" block from C++ stdout
    # Format:
    #   Image 0  label=5  pred=3  WRONG
    #     logits: 0.06 0.40 ...
    lines   = result.stdout.splitlines()
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Image "):
            parts = line.split()
            # pred=X
            pred_str = [p for p in parts if p.startswith("pred=")][0]
            pred = int(pred_str.split("=")[1])
            # next line has logits
            logit_line = lines[i + 1].strip()
            assert logit_line.startswith("logits:"), f"Unexpected line: {logit_line}"
            logits = np.array([float(x) for x in logit_line.split()[1:]], dtype=np.float32)
            entries.append((pred, logits))
            i += 2
        else:
            i += 1
    return entries

# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def verify(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TinyCNN().to(device)
    ckpt  = os.path.join(args.weights_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"[verify] No checkpoint found at {ckpt}")
        print("         Run train_tiny_cnn.py first.")
        sys.exit(1)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"Loaded model from {ckpt}")

    # Load test dataset
    test_ds = MNISTDataset(
        os.path.join(args.data_dir, "t10k-images-idx3-ubyte"),
        os.path.join(args.data_dir, "t10k-labels-idx1-ubyte"))

    n = min(args.n, 10)  # C++ verbose mode prints first 10 only
    print(f"Comparing first {n} images between PyTorch and C++...\n")

    # Get C++ results
    cpp_results = cpp_inference_batch(args.binary, args.data_dir, args.weights_dir, n)
    if len(cpp_results) < n:
        print(f"[verify] WARNING: only got {len(cpp_results)} results from C++ (expected {n})")
        n = len(cpp_results)

    # Compare image by image
    all_match      = True
    logit_max_err  = 0.0

    header = f"{'Img':>4}  {'Label':>5}  {'PyTorch':>7}  {'C++':>4}  {'Match':>5}  {'MaxLogitErr':>11}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        image, label     = test_ds[i]
        py_pred, py_logits = pytorch_inference(model, image, device)
        cpp_pred, cpp_logits = cpp_results[i]

        match     = (py_pred == cpp_pred)
        max_err   = float(np.max(np.abs(py_logits - cpp_logits)))
        logit_max_err = max(logit_max_err, max_err)

        flag = "OK" if match else "MISMATCH"
        print(f"{i:>4}  {int(label):>5}  {py_pred:>7}  {cpp_pred:>4}  {flag:>5}  {max_err:>11.6f}")

        if not match:
            all_match = False
            print(f"       PyTorch logits : {py_logits}")
            print(f"       C++     logits : {cpp_logits}")

    print()
    print(f"Prediction match : {'ALL MATCH ✓' if all_match else 'MISMATCHES FOUND ✗'}")
    print(f"Max logit error  : {logit_max_err:.6f}  "
          f"({'OK — within float32 tolerance' if logit_max_err < 1e-4 else 'WARNING — too large'})")

    if all_match and logit_max_err < 1e-4:
        print("\nVerification PASSED — C++ model matches PyTorch exactly.")
    else:
        print("\nVerification FAILED — investigate layer implementations.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=10,
                        help="Number of images to compare (max 10, limited by C++ verbose output)")
    parser.add_argument("--data_dir",    default="../data",
                        help="Directory with MNIST IDX files")
    parser.add_argument("--weights_dir", default="../weights",
                        help="Directory with exported .bin weight files")
    parser.add_argument("--binary",      default="../../float/mnist_cnn",
                        help="Path to the C++ inference binary (float/ or quantized/)")
    args = parser.parse_args()
    verify(args)
