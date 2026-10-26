# Phase 5 — C++ to SystemVerilog Mapping

This document describes how the quantized C++ inference model (`sw/quantized/`)
maps to the SystemVerilog implementation in `hw/v1/rtl/`.

---

## Architecture Overview

**C++ model**: sequential function calls, each layer reads/writes arrays in memory.
**SV design**: BRAM-based layer-by-layer compute, mimicking the C++ exactly.

```
C++ forward()                  SV cnn_top
─────────────                  ──────────
quantize input → int8          Host writes int8 pixels to input BRAM
conv2d_q(CONV1)                conv_engine #(C1) reads input BRAM → writes conv1 BRAM
max_pool2d_q(POOL1)            pool_engine #(P1) reads conv1 BRAM → writes pool1 BRAM
conv2d_q(CONV2)                conv_engine #(C2) reads pool1 BRAM → writes conv2 BRAM
max_pool2d_q(POOL2)            pool_engine #(P2) reads conv2 BRAM → writes pool2 BRAM
fc_forward_q(FC1)              fc_engine #(FC1)  reads pool2 BRAM → writes fc1 BRAM
fc_forward_q_raw(FC2)          fc_engine #(FC2, RAW) reads fc1 BRAM → writes fc2 BRAM
argmax_q()                     argmax reads fc2 BRAM → pred_class
```

Each layer runs to completion before the next starts, controlled by the
top-level FSM (`S_IDLE → S_CONV1 → S_POOL1 → ... → S_DONE`).

---

## Module-by-Module Mapping

### 1. Input Quantization

| C++ (`cnn.cpp:350-356`) | SV |
|---|---|
| `buf_input_q[i] = round(input[i] / scale_input)` | Done by host before loading. Host writes int8 pixels directly via `img_wr_en/addr/data`. |

The host (testbench or SoC) pre-quantizes the float image to int8 and loads it
into the input BRAM. This keeps the FPGA design purely integer — no float hardware needed.

### 2. Convolution: `conv2d_q()` → `conv_engine.sv`

| C++ (`conv_layer.cpp`) | SV (`conv_engine.sv`) |
|---|---|
| 6 nested `for` loops (n, x, y, c, i, j) | FSM with counters: `n_cnt`, `x_cnt`, `y_cnt`, `c_cnt`, `i_cnt`, `j_cnt` |
| `acc = bias[n]` | FSM reads bias ROM, loads into `acc` register |
| `acc += input[idx] * kernel[idx]` | FSM reads activation BRAM + weight ROM, computes `acc + a*b` |
| `requantize_relu(acc, rq)` | `requantize.sv` submodule: `(acc * mult + round) >> shift`, clamp |
| `output[idx] = result` | FSM writes int8 result to output BRAM |

**Arithmetic**: int8 × int8 → int16 product, accumulated into int32.
No overflow: max accumulation for CONV2 = 4×9 × 127² = 580,644 (fits int32).

**Cycles per output pixel**: C_in × K × K = 9 (CONV1) or 36 (CONV2), plus overhead.

### 3. Max Pooling: `max_pool2d_q()` → `pool_engine.sv`

| C++ (`pool_layer.cpp`) | SV (`pool_engine.sv`) |
|---|---|
| `max_val = -128` | `max_val <= -8'sd128` (minimum int8) |
| `if (v > max_val) max_val = v` | Signed int8 comparison in FSM |
| 5 nested loops (c, x, y, i, j) | FSM counters: `c_cnt`, `x_cnt`, `y_cnt`, `i_cnt`, `j_cnt` |

**Key property**: Max pooling preserves the quantization scale (no arithmetic,
just comparison). No requantization needed.

**Cycles per output pixel**: POOL × POOL = 4, plus BRAM read latency.

### 4. Fully Connected: `fc_forward_q()` → `fc_engine.sv`

| C++ (`fc_layer.cpp`) | SV (`fc_engine.sv`) |
|---|---|
| `acc = bias[i]` | FSM reads bias ROM |
| `acc += weight[i*N+j] * input[j]` | Sequential MAC: read weight ROM + activation BRAM each cycle |
| `requantize_relu(acc, rq)` | `requantize.sv` submodule (FC1 only) |
| `output[i] = acc` (FC2 raw) | Direct int32 write to output BRAM (FC2, `RAW_OUT=1`) |

**FC1 cycles**: 64 × 200 = 12,800 MACs
**FC2 cycles**: 10 × 64 = 640 MACs

### 5. Argmax: `argmax_q()` → `argmax.sv`

| C++ (`activations.h`) | SV (`argmax.sv`) |
|---|---|
| `for (i=1; i<size; i++) if (logits[i] > logits[best]) best = i` | Sequential read of 10 int32 values from FC2 BRAM, tracking `best_val` and `best_idx` |

**Cycles**: 10 (one read per class).

### 6. Requantization: `requantize()` → `requantize.sv`

| C++ (`fixed_point.h`) | SV (`requantize.sv`) |
|---|---|
| `v = (int64)acc * multiplier` | 32×32→64 bit multiply (DSP48 inference) |
| `v = (v + (1 << (shift-1))) >> shift` | Add rounding constant, arithmetic right shift |
| `clamp to [-128,127] or [0,127]` | Saturation logic, `apply_relu` selects range |

**Pipeline**: 2 stages (multiply → shift+clamp). This matches the C++ `RequantParams`
struct: `multiplier` (int32) and `shift` (int).

---

## Memory Mapping

### Activation BRAMs

| Buffer | C++ array | SV BRAM | Width | Depth |
|--------|-----------|---------|-------|-------|
| Input image | `buf_input_q[784]` | `u_input_bram` | 8b | 784 |
| Conv1 output | `buf_conv1[2704]` | `u_conv1_bram` | 8b | 2704 |
| Pool1 output | `buf_pool1[676]` | `u_pool1_bram` | 8b | 676 |
| Conv2 output | `buf_conv2[968]` | `u_conv2_bram` | 8b | 968 |
| Pool2 output | `buf_pool2[200]` | `u_pool2_bram` | 8b | 200 |
| FC1 output | `buf_fc1[64]` | `u_fc1_bram` | 8b | 64 |
| FC2 output | `buf_fc2[10]` | `u_fc2_bram` | 32b | 10 |

Total activation storage: ~5,436 bytes + 40 bytes (int32) = ~5.5 KB

### Weight ROMs

| Layer | C++ array | SV ROM | Width | Depth | .hex file |
|-------|-----------|--------|-------|-------|-----------|
| CONV1 weights | `conv1_w_q[36]` | `u_conv1_wt` | 8b | 36 | `conv1_weights.hex` |
| CONV1 biases | `conv1_b_q[4]` | `u_conv1_bias` | 32b | 4 | `conv1_bias.hex` |
| CONV2 weights | `conv2_w_q[288]` | `u_conv2_wt` | 8b | 288 | `conv2_weights.hex` |
| CONV2 biases | `conv2_b_q[8]` | `u_conv2_bias` | 32b | 8 | `conv2_bias.hex` |
| FC1 weights | `fc1_w_q[12800]` | `u_fc1_wt` | 8b | 12800 | `fc1_weights.hex` |
| FC1 biases | `fc1_b_q[64]` | `u_fc1_bias` | 32b | 64 | `fc1_bias.hex` |
| FC2 weights | `fc2_w_q[640]` | `u_fc2_wt` | 8b | 640 | `fc2_weights.hex` |
| FC2 biases | `fc2_b_q[10]` | `u_fc2_bias` | 32b | 10 | `fc2_bias.hex` |

Total weight storage: 13,764 bytes + 344 bytes (int32 biases) ≈ 14 KB

---

## Reference Design Reuse

The reference modules in `hw/Learn/parameterized_thedatabus/` use (N,Q)
sign-magnitude fixed-point arithmetic, which is fundamentally different
from our int8 symmetric quantization. Here is what was reused vs new:

| Reference Module | Reused As | Notes |
|-----------------|-----------|-------|
| `variable_shift_reg.sv` | Pattern for `bram_sp.sv` | BRAM replaces shift-register row buffers |
| `comparator2.sv` | Inspired signed comparison in `pool_engine.sv` | Simplified: int8 signed `>` is native in SV |
| `mac_manual.sv` | Pattern for MAC in `conv_engine.sv` | Replaced qmult/qadd with int8 multiply+accumulate |
| `convolver.sv` | Structural pattern for `conv_engine.sv` | Changed from streaming to BRAM-based compute |
| `pooler.sv` + `control_logic2.sv` | Pattern for `pool_engine.sv` | Simplified: BRAM reads instead of streaming control |
| `accelerator.sv` | Pattern for `cnn_top.sv` | Layer sequencing FSM instead of pipeline |
| `relu.sv` | Fused into `requantize.sv` | Clamp to [0,127] instead of separate sign-bit check |
| `qmult.sv` / `qadd.sv` | **Not used** | (N,Q) arithmetic replaced by int8 quantization |
| `tanh_lut.sv` | **Not used** | Model uses ReLU only |

**New modules** (no reference equivalent):
- `requantize.sv` — int32→int8 requantization (multiply+shift+clamp)
- `fc_engine.sv` — fully connected layer (reference only had conv+pool)
- `argmax.sv` — 10-class argmax comparator
- `rom_sp.sv` — weight/bias ROM storage
- `bram_sp.sv` — generic activation buffer

---

## Timing Estimate

At 100 MHz clock, one inference takes approximately:

| Layer | Cycles (approx) | Time |
|-------|---------|------|
| CONV1 | 4 × 26 × 26 × 9 + overhead ≈ 25,000 | 0.25 ms |
| POOL1 | 4 × 13 × 13 × 4 ≈ 2,700 | 0.03 ms |
| CONV2 | 8 × 11 × 11 × 36 + overhead ≈ 36,000 | 0.36 ms |
| POOL2 | 8 × 5 × 5 × 4 ≈ 800 | 0.01 ms |
| FC1 | 64 × 200 ≈ 13,000 | 0.13 ms |
| FC2 | 10 × 64 ≈ 700 | 0.01 ms |
| ARGMAX | 10 | <0.01 ms |
| **Total** | **~78,000** | **~0.78 ms** |

This is a conservative v1 with one MAC per cycle. See `scaling.md` for
how to improve throughput via parallelism.
