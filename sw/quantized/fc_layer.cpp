#include "fc_layer.h"

// ---------------------------------------------------------------------------
// fc_forward_q — Quantized fully connected, with requantization.
//
// Each output neuron:
//   acc = bias_q[i] + Σ(weight_q[i*N+j] × input_q[j])   (all int32 arithmetic)
//   output[i] = requantize(acc)                           (→ int8)
// ---------------------------------------------------------------------------
void fc_forward_q(const int8_t*  input,
                  const int8_t*  weights,
                  const int32_t* bias,
                  int8_t*        output,
                  int N, int M,
                  RequantParams rq,
                  bool apply_relu)
{
    for (int i = 0; i < M; i++) {
        int32_t acc = bias[i];
        for (int j = 0; j < N; j++) {
            acc += (int32_t)weights[i * N + j] * (int32_t)input[j];
        }
        output[i] = apply_relu ? requantize_relu(acc, rq) : requantize(acc, rq);
    }
}

// ---------------------------------------------------------------------------
// fc_forward_q_raw — No requantization, returns int32 accumulator directly.
// Used for the final layer where argmax on raw logits is sufficient.
// ---------------------------------------------------------------------------
void fc_forward_q_raw(const int8_t*  input,
                      const int8_t*  weights,
                      const int32_t* bias,
                      int32_t*       output,
                      int N, int M)
{
    for (int i = 0; i < M; i++) {
        int32_t acc = bias[i];
        for (int j = 0; j < N; j++) {
            acc += (int32_t)weights[i * N + j] * (int32_t)input[j];
        }
        output[i] = acc;
    }
}
