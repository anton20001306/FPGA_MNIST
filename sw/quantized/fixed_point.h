#pragma once
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// fixed_point.h
// Quantization utilities for int8 inference.
//
// Quantization scheme (symmetric, per-tensor):
//   real_value = quantized_value × scale
//   scale      = max|real_values| / 127.0
//
// Requantization (accumulator → int8):
//   After MAC in int32, we need to rescale to the next layer's int8 range.
//   M = (scale_weight × scale_input) / scale_output
//   We approximate M as: multiplier × 2^(-shift)
//   where multiplier is a 32-bit fixed-point integer.
//
//   This is FPGA-friendly: one integer multiply + one right-shift.
// ---------------------------------------------------------------------------

// Parameters for integer-only requantization.
// On FPGA these become a DSP multiply and a barrel shifter.
struct RequantParams {
    int32_t multiplier;  // fixed-point representation of M in Q0.31
    int     shift;       // total right-shift amount
};

// Compute RequantParams from a float multiplier M.
// M is typically in (0, 1] for neural network layers.
inline RequantParams compute_requant_params(float M) {
    RequantParams p;
    double m = (double)M;

    if (m == 0.0) { p.multiplier = 0; p.shift = 0; return p; }

    // Normalize m to [0.5, 1.0)
    int n = 0;
    while (m < 0.5)  { m *= 2.0; n++; }
    while (m >= 1.0)  { m /= 2.0; n--; }

    // Express as Q0.31 fixed-point: multiplier = round(m × 2^31)
    p.multiplier = (int32_t)(m * (1LL << 31) + 0.5);
    p.shift = 31 + n;
    return p;
}

// Apply requantization: result = round(acc × M), clamped to [-128, 127].
// On FPGA: one 32×32→64 multiply, add rounding constant, right-shift, saturate.
inline int8_t requantize(int32_t acc, RequantParams p) {
    int64_t v = (int64_t)acc * p.multiplier;
    // Round-half-up: add 2^(shift-1) before shifting
    if (p.shift > 0)
        v = (v + (1LL << (p.shift - 1))) >> p.shift;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

// Requantize with ReLU fused: clamp to [0, 127] instead of [-128, 127].
// Fusing ReLU into requantization is standard in quantized inference and
// saves a separate pass over the data (one fewer memory read/write).
inline int8_t requantize_relu(int32_t acc, RequantParams p) {
    int64_t v = (int64_t)acc * p.multiplier;
    if (p.shift > 0)
        v = (v + (1LL << (p.shift - 1))) >> p.shift;
    if (v > 127) v = 127;
    if (v < 0)   v = 0;   // ReLU: clamp negatives
    return (int8_t)v;
}
