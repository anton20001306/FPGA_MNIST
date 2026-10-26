#include "conv_layer.h"

// ---------------------------------------------------------------------------
// conv2d_q — Quantized 6-loop 2D convolution.
//
// Identical loop structure to the float version, but:
//   - Inputs and weights are int8
//   - Accumulation in int32 (no overflow: 9 MACs of int8×int8 ≤ 9×127×127 = 145,161)
//   - Output requantized to int8, optionally with ReLU fused
//
// This loop order matches what the FPGA will implement with nested counters.
// ---------------------------------------------------------------------------
void conv2d_q(const int8_t*  input,
              const int8_t*  kernel,
              const int32_t* bias,
              int8_t*        output,
              int C_in, int H_in, int W_in,
              int C_out, int K,
              RequantParams rq,
              bool apply_relu)
{
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    for (int n = 0; n < C_out; n++) {
        for (int x = 0; x < H_out; x++) {
            for (int y = 0; y < W_out; y++) {

                int32_t acc = bias[n];

                for (int c = 0; c < C_in; c++) {
                    for (int i = 0; i < K; i++) {
                        for (int j = 0; j < K; j++) {
                            int in_idx  = c * H_in * W_in + (x + i) * W_in + (y + j);
                            int ker_idx = n * C_in * K * K + c * K * K + i * K + j;
                            acc += (int32_t)input[in_idx] * (int32_t)kernel[ker_idx];
                        }
                    }
                }

                int out_idx = n * H_out * W_out + x * W_out + y;
                output[out_idx] = apply_relu ? requantize_relu(acc, rq)
                                             : requantize(acc, rq);
            }
        }
    }
}
