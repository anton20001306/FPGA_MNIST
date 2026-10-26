#include "fc_layer.h"

// ---------------------------------------------------------------------------
// fc_forward — matrix-vector multiply + bias.
//
// Each output neuron i computes:
//   output[i] = bias[i] + dot(weights[i*N .. i*N+N-1], input[0..N-1])
//
// This is the same MAC (multiply-accumulate) operation as the CONV layer;
// on FPGA both will share the same PE (processing element).
// ---------------------------------------------------------------------------
void fc_forward(const float* input,
                const float* weights,
                const float* bias,
                float*       output,
                int N, int M)
{
    for (int i = 0; i < M; i++) {
        float acc = bias[i];
        for (int j = 0; j < N; j++) {
            acc += weights[i * N + j] * input[j];
        }
        output[i] = acc;
    }
}
