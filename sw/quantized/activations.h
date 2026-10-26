#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// activations.h — Quantized activation functions
//
// ReLU on int8: with symmetric quantization (zero_point = 0),
// negative int8 values correspond to negative real values, so
// clamping to 0 is equivalent to float ReLU.
//
// Note: In practice, ReLU is fused into the requantization step
// (see requantize_relu in fixed_point.h). This standalone function
// is provided for cases where ReLU is applied separately.
//
// Argmax on int32: used on raw FC2 logits (final layer, no requantization).
// On FPGA this is a simple comparator chain on the accumulator outputs.
// ---------------------------------------------------------------------------

// ReLU on int8 array (in-place)
inline void relu_q(int8_t* data, int size) {
    for (int i = 0; i < size; i++)
        if (data[i] < 0) data[i] = 0;
}

// Argmax on int32 logits — returns index of largest value (predicted class)
inline int argmax_q(const int32_t* logits, int size) {
    int best = 0;
    for (int i = 1; i < size; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}
