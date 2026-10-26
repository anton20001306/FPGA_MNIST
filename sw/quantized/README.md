# Quantized MNIST CNN (int8)

Fixed-point (int8) version of the MNIST CNN inference engine.
Same architecture as `../float/`, but all inference arithmetic is integer-only.

## Quantization Scheme

- **Type**: Post-training quantization (PTQ), symmetric, per-tensor
- **Weights**: `int8_t` [-128, 127], scale = max|w| / 127
- **Activations**: `int8_t` [0, 127] (post-ReLU, fused into requantization)
- **Accumulators**: `int32_t` (int8 x int8 MACs)
- **Biases**: `int32_t`, pre-quantized to accumulator scale
- **Requantization**: `M ~= multiplier * 2^(-shift)` (one multiply + one shift, FPGA-friendly)
- **Final layer**: raw `int32_t` logits, argmax directly (no requantization)

## How It Maps to FPGA

| C++ operation | FPGA resource |
|---|---|
| `int8 * int8` MAC | DSP slice (8x8 multiply-accumulate) |
| `int32` accumulator | DSP slice accumulator or fabric registers |
| `requantize(acc, M, shift)` | One DSP multiply + barrel shifter |
| `max()` in pooling | Comparator tree |
| `argmax()` | Comparator chain |

## Build & Run

```bash
make        # build
make run    # calibrate + quantized inference on MNIST test set
make clean  # cleanup
```

## Expected Output

Accuracy should be close to the float model (98.75%), typically within 0.5-1% drop.
