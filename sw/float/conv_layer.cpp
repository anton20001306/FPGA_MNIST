#include "conv_layer.h"

// ---------------------------------------------------------------------------
// conv2d — naive 6-loop 2D convolution.
//
// Loop order: output_filter → output_row → output_col →
//             input_channel → kernel_row → kernel_col
//
// This order is deliberately explicit for learning: every multiply-accumulate
// corresponds exactly to one entry in the formula from convolution.md.
//
// Performance is not the goal here — correctness and readability are.
// The HW implementation will parallelise these loops.
// ---------------------------------------------------------------------------
void conv2d(const float* input,
            const float* kernel,
            const float* bias,
            float*       output,
            int C_in, int H_in, int W_in,
            int C_out, int K)
{
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    for (int n = 0; n < C_out; n++) {           // output filter index
        for (int x = 0; x < H_out; x++) {       // output row
            for (int y = 0; y < W_out; y++) {   // output col

                float acc = bias[n];

                for (int c = 0; c < C_in; c++) {       // input channel
                    for (int i = 0; i < K; i++) {       // kernel row
                        for (int j = 0; j < K; j++) {   // kernel col
                            // input  : [C_in][H_in][W_in]
                            int in_idx  = c * H_in * W_in + (x + i) * W_in + (y + j);
                            // kernel : [C_out][C_in][K][K]
                            int ker_idx = n * C_in * K * K + c * K * K + i * K + j;
                            acc += input[in_idx] * kernel[ker_idx];
                        }
                    }
                }

                // output : [C_out][H_out][W_out]
                output[n * H_out * W_out + x * W_out + y] = acc;
            }
        }
    }
}
