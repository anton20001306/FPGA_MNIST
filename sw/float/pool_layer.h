#pragma once

// ---------------------------------------------------------------------------
// pool_layer.h
// 2D Max Pooling — pool_size × pool_size window, stride = pool_size (no overlap).
//
// Tensor layout: CHW flat float array.
//   input  : [C][H_in ][W_in ]
//   output : [C][H_out][W_out]
//
//   H_out = H_in / pool_size   (integer division — odd rows/cols are dropped)
//   W_out = W_in / pool_size
//   Channels are unchanged.
// ---------------------------------------------------------------------------

// Max pooling forward pass.
// pool_size = 2 gives 2×2 pooling (halves spatial dimensions).
void max_pool2d(const float* input,
                float*       output,
                int C, int H_in, int W_in,
                int pool_size);
