# Phase 7 — Scaling Guide

How to scale the v1 CNN accelerator for larger models and higher throughput.

---

## Current v1 Baseline

- **Model**: Tiny CNN (4→8 filters, 200→64→10 FC)
- **Arithmetic**: 1 MAC/cycle (sequential)
- **Latency**: ~78,000 cycles per inference (~0.78 ms at 100 MHz)
- **Resources**: Minimal (1 DSP for MAC, ~20 KB BRAM)

---

## Scaling Dimension 1: Parallelism (more MACs)

### 1a. Multiple MACs per output pixel

Instead of computing one `input[c,x+i,y+j] * weight[n,c,i,j]` per cycle,
instantiate P MAC units and feed P multiply-accumulate pairs per cycle.

**How to modify `conv_engine.sv`:**
- Add parameter `NUM_MACS` (e.g., 4 or 9)
- Read P weights and P activations per cycle (widen BRAM ports or use multi-port)
- Accumulate partial sums from all P MACs into a single int32 register
- Speedup: ~P× for convolution layers

**Package change:**
```systemverilog
parameter int CONV_PARALLELISM = 4;  // MACs per cycle for conv
```

**Resource cost:** P DSP slices per conv engine.

### 1b. Multiple output channels in parallel

Instantiate C_OUT conv engines, each computing a different output channel
simultaneously. All share the same input BRAM but read different weight ROMs.

**How to modify:**
- `conv_engine.sv`: keep as-is (1 output channel)
- `cnn_top.sv`: instantiate array of `conv_engine` modules
- Weight ROMs: split per output channel, or use wider ROM with demux

**Speedup**: C_OUT× (4× for CONV1, 8× for CONV2)
**Resource cost**: C_OUT × (1 DSP + control logic)

### 1c. Systolic array for FC layers

FC layers are matrix-vector multiplies — ideal for systolic arrays.
Reuse the systolic array designs in `hw/Learn/systolic_arrays/` as a
starting point and adapt for int8 arithmetic.

**How to modify `fc_engine.sv`:**
- Replace sequential MAC loop with a row of P PEs
- Feed P weights per cycle, accumulate across PEs
- FC1 speedup: 200/P cycles per output neuron instead of 200

---

## Scaling Dimension 2: Larger Models

### 2a. More filters / channels

All layer dimensions are parameterized in `cnn_pkg.sv`. To use a larger model:

1. Update `cnn_pkg.sv`:
```systemverilog
parameter int C1_OUT = 16;  // was 4
parameter int C2_OUT = 32;  // was 8
parameter int FC1_OUT = 128; // was 64
```

2. Regenerate weight `.hex` files from the retrained PyTorch model
3. Update BRAM depths (auto-derived from parameters)
4. Re-synthesize

**Impact**: More BRAM for weights and activations, more cycles for compute.
FC1 weight storage grows fast (FC1_IN × FC1_OUT).

### 2b. Deeper networks (more layers)

1. Add new entries in `cnn_pkg.sv` for each layer (C3_IN, C3_OUT, K3, etc.)
2. Instantiate additional `conv_engine` / `pool_engine` in `cnn_top.sv`
3. Add BRAM and ROM instances for each new layer
4. Extend the top-level FSM with new states

### 2c. Larger input images

Change `IMG_H` and `IMG_W` in the package. All derived dimensions update
automatically. The only concern is BRAM capacity for larger activation maps.

---

## Scaling Dimension 3: Memory Optimization

### 3a. Weight sharing / BRAM reuse

v1 uses separate ROMs per layer. Since layers execute sequentially, a single
large weight BRAM can be time-shared across layers using address offsets:

```
Weight BRAM layout:
[0..35]      CONV1 weights
[36..323]    CONV2 weights
[324..13123] FC1 weights
[13124..13763] FC2 weights
```

**Package change:**
```systemverilog
parameter int CONV1_W_OFFSET = 0;
parameter int CONV2_W_OFFSET = CONV1_W_SIZE;
parameter int FC1_W_OFFSET   = CONV2_W_OFFSET + CONV2_W_SIZE;
parameter int FC2_W_OFFSET   = FC1_W_OFFSET + FC1_W_SIZE;
```

### 3b. Activation buffer ping-pong

Since each layer reads from one buffer and writes to another, only 2 activation
BRAMs are needed (ping-pong). Reduces BRAM usage from 7 buffers to 2.

**Trade-off**: More complex address muxing, but significant BRAM savings for
larger models.

### 3c. Weight compression

For larger models, store weights in a compressed format and decompress on-the-fly.
Common approaches:
- Run-length encoding for sparse weights (after pruning)
- Huffman coding
- Weight clustering (e.g., 16 centroids → 4-bit indices)

---

## Scaling Dimension 4: Clock Frequency

### 4a. Pipeline arithmetic

v1 requantize has 2 pipeline stages. For higher Fmax:
- Pipeline the MAC itself (register after multiply, register after add)
- Add pipeline stages in the BRAM read path
- Target: 200+ MHz on modern FPGAs

### 4b. Retiming

Use synthesis retiming directives to let the tool automatically redistribute
registers for optimal timing.

---

## Example: v2 Target Configuration

A reasonable next step after v1 is verified:

| Parameter | v1 | v2 |
|-----------|----|----|
| Conv MACs/cycle | 1 | 9 (full kernel) |
| FC MACs/cycle | 1 | 8 |
| Weight storage | Separate ROMs | Shared BRAM |
| Activation storage | 7 BRAMs | 2 BRAMs (ping-pong) |
| Est. latency | 78K cycles | ~12K cycles |
| Est. throughput | ~1.3K images/sec | ~8.3K images/sec |

---

## Summary of Package Parameters to Change

| What to scale | Package parameters | Files affected |
|---------------|-------------------|----------------|
| Model size | `C1_OUT`, `C2_OUT`, `FC1_OUT`, `K1`, `K2` | All (auto-derived) |
| Input size | `IMG_H`, `IMG_W` | All (auto-derived) |
| Parallelism | Add `NUM_MACS`, `FC_PARALLELISM` | `conv_engine.sv`, `fc_engine.sv` |
| Memory sharing | Add offset parameters | `cnn_top.sv` |
| Number of layers | Add `C3_*`, `K3`, etc. | `cnn_pkg.sv`, `cnn_top.sv` |
