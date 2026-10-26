#include "pool_layer.h"

// ---------------------------------------------------------------------------
// max_pool2d_q — non-overlapping 2D max pooling on int8 data.
//
// Quantization scale is preserved through max-pooling (no arithmetic,
// just comparison). On FPGA this is a comparator tree.
// ---------------------------------------------------------------------------
void max_pool2d_q(const int8_t* input,
                  int8_t*       output,
                  int C, int H_in, int W_in,
                  int pool_size)
{
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;

    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H_out; x++) {
            for (int y = 0; y < W_out; y++) {

                int8_t max_val = -128;  // minimum int8

                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        int ix  = x * pool_size + i;
                        int iy  = y * pool_size + j;
                        int8_t v = input[c * H_in * W_in + ix * W_in + iy];
                        if (v > max_val) max_val = v;
                    }
                }

                output[c * H_out * W_out + x * W_out + y] = max_val;
            }
        }
    }
}
