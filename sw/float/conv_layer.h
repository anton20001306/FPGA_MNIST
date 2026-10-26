#pragma once

// ---------------------------------------------------------------------------
// conv_layer.h
// 2D Convolution — no padding, stride = 1.
//
// Tensor layout throughout: CHW (channel-first), flat float array.
//   input  : [C_in ][H_in ][W_in ]  → index: c*H_in*W_in  + h*W_in  + w
//   kernel : [C_out][C_in ][K    ][K]  → index: n*C_in*K*K  + c*K*K   + i*K + j
//   output : [C_out][H_out][W_out]  → index: n*H_out*W_out + h*W_out + w
//
//   H_out = H_in - K + 1
//   W_out = W_in - K + 1
// ---------------------------------------------------------------------------

// Convolution forward pass.
// Does NOT apply ReLU — call relu() from activations.h afterwards.
void conv2d(const float* input,
            const float* kernel,
            const float* bias,
            float*       output,
            int C_in, int H_in, int W_in,
            int C_out, int K);
