#pragma once

// ---------------------------------------------------------------------------
// fc_layer.h
// Fully Connected (Dense / Linear) layer.
//
// output[i] = bias[i] + Σ_j  weights[i*N + j] * input[j]
//
//   input   : [N]  (flat vector)
//   weights : [M][N] row-major flat array
//   bias    : [M]
//   output  : [M]
//
// Does NOT apply ReLU — call relu() from activations.h afterwards for hidden
// layers. The final FC layer (FC2) leaves raw logits for argmax.
// ---------------------------------------------------------------------------

// Fully connected forward pass.
// N = input size, M = output size.
void fc_forward(const float* input,
                const float* weights,
                const float* bias,
                float*       output,
                int N, int M);
