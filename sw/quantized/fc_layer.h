#pragma once
#include <cstdint>
#include "fixed_point.h"

// ---------------------------------------------------------------------------
// fc_layer.h — Quantized Fully Connected Layer
//
// Two variants:
//   fc_forward_q     — int8 → int32 MAC → requantize → int8  (hidden layers)
//   fc_forward_q_raw — int8 → int32 MAC → int32 output       (final layer)
//
// The final FC layer keeps raw int32 logits because:
//   1. argmax is order-preserving, so requantization adds error for no benefit
//   2. On FPGA, the final comparator chain operates on the raw accumulator
// ---------------------------------------------------------------------------

// FC with requantization + optional ReLU (for FC1 hidden layer)
void fc_forward_q(const int8_t*  input,
                  const int8_t*  weights,
                  const int32_t* bias,
                  int8_t*        output,
                  int N, int M,
                  RequantParams rq,
                  bool apply_relu);

// FC without requantization — returns raw int32 logits (for FC2 output layer)
void fc_forward_q_raw(const int8_t*  input,
                      const int8_t*  weights,
                      const int32_t* bias,
                      int32_t*       output,
                      int N, int M);
