#pragma once
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// activations.h
// Inline activation functions used across layers.
// All operate on flat float arrays (CHW layout).
// ---------------------------------------------------------------------------

// ReLU applied in-place on an array of `size` elements.
// f(x) = max(0, x)
inline void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// Softmax written into `probs` from raw `logits` of length `size`.
// Uses the max-subtraction trick for numerical stability.
// Only used for debugging / probability display — inference uses argmax().
inline void softmax(const float* logits, float* probs, int size) {
    float max_val = logits[0];
    for (int i = 1; i < size; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < size; i++)
        probs[i] /= sum;
}

// Returns the index of the largest value — the predicted class.
// On FPGA this becomes a simple comparator chain.
inline int argmax(const float* logits, int size) {
    int best = 0;
    for (int i = 1; i < size; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}
