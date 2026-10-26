#include "pool_layer.h"

// ---------------------------------------------------------------------------
// max_pool2d — non-overlapping 2D max pooling.
//
// For each (channel, output_row, output_col), scan the pool_size×pool_size
// window in the input and keep the maximum value.
//
// Maps directly to a comparator tree in hardware (see max_pooling.md).
// ---------------------------------------------------------------------------
void max_pool2d(const float* input,
                float*       output,
                int C, int H_in, int W_in,
                int pool_size)
{
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;

    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H_out; x++) {
            for (int y = 0; y < W_out; y++) {

                float max_val = -1e30f;  // start below any real activation

                for (int i = 0; i < pool_size; i++) {
                    for (int j = 0; j < pool_size; j++) {
                        int ix  = x * pool_size + i;
                        int iy  = y * pool_size + j;
                        float v = input[c * H_in * W_in + ix * W_in + iy];
                        if (v > max_val) max_val = v;
                    }
                }

                output[c * H_out * W_out + x * W_out + y] = max_val;
            }
        }
    }
}
