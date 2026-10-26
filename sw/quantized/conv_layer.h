#pragma once
#include <cstdint>
#include "fixed_point.h"

// ---------------------------------------------------------------------------
// conv_layer.h — Quantized 2D Convolution
//
// Same tensor layout as float version: CHW flat arrays.
//   input  : int8_t  [C_in ][H_in ][W_in ]
//   kernel : int8_t  [C_out][C_in ][K    ][K]
//   bias   : int32_t [C_out]  — pre-quantized to accumulator scale
//   output : int8_t  [C_out][H_out][W_out]
//
// Arithmetic: int8 × int8 → int32 accumulator → requantize → int8
// ReLU can be fused into the requantization step (apply_relu flag).
//
// Maps to FPGA: each MAC is a DSP slice (8×8→16, accumulated into 32-bit).
// ---------------------------------------------------------------------------

void conv2d_q(const int8_t*  input,
              const int8_t*  kernel,
              const int32_t* bias,
              int8_t*        output,
              int C_in, int H_in, int W_in,
              int C_out, int K,
              RequantParams rq,
              bool apply_relu);
