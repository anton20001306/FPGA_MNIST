#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// pool_layer.h — Quantized 2D Max Pooling
//
// Operates on int8 values. Since max-pooling only compares values, the
// quantization scale passes through unchanged (no multiply needed).
//
// On FPGA: a simple comparator tree, identical to the float version's logic.
// ---------------------------------------------------------------------------

void max_pool2d_q(const int8_t* input,
                  int8_t*       output,
                  int C, int H_in, int W_in,
                  int pool_size);
