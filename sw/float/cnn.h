#pragma once
#include <string>

// ---------------------------------------------------------------------------
// cnn.h
// TinyCNN — wires all layers together and owns all weight buffers.
//
// Architecture (from main_plan.md):
//   Input    28×28×1
//   CONV1    3×3, 4 filters  → 26×26×4  + ReLU
//   MaxPool1 2×2             → 13×13×4
//   CONV2    3×3, 8 filters  → 11×11×8  + ReLU
//   MaxPool2 2×2             →  5×5×8
//   Flatten                  → 200
//   FC1      200→64          + ReLU
//   FC2      64→10           (raw logits)
//   argmax                   → predicted digit 0–9
// ---------------------------------------------------------------------------

class TinyCNN {
public:
    TinyCNN();
    ~TinyCNN();

    // Load all weights from binary files in `weights_dir`.
    // Files must be named as documented in weights_io.h.
    // Returns false if any file is missing or malformed.
    bool load_weights(const std::string& weights_dir);

    // Initialize all weights with random values (He initialisation).
    // Useful for smoke-testing the shape pipeline before trained weights exist.
    void init_random_weights();

    // Run a full forward pass.
    // `input` must point to 28*28 = 784 floats in [0,1] (one image, [1][28][28]).
    // Returns the predicted digit (0–9).
    int forward(const float* input);

    // Run forward pass and also fill `logits_out` with the 10 raw FC2 outputs.
    // Useful for debugging confidence / verifying against Python reference.
    int forward_verbose(const float* input, float* logits_out);

private:
    // -----------------------------------------------------------------------
    // Weight tensors (heap-allocated flat float arrays)
    // -----------------------------------------------------------------------
    float* conv1_w;   // [4][1][3][3]  = 36
    float* conv1_b;   // [4]
    float* conv2_w;   // [8][4][3][3]  = 288
    float* conv2_b;   // [8]
    float* fc1_w;     // [64][200]     = 12800
    float* fc1_b;     // [64]
    float* fc2_w;     // [10][64]      = 640
    float* fc2_b;     // [10]

    // -----------------------------------------------------------------------
    // Intermediate activation buffers (reused across calls, heap-allocated)
    // -----------------------------------------------------------------------
    float* buf_conv1;  // [4][26][26]  = 2704
    float* buf_pool1;  // [4][13][13]  = 676
    float* buf_conv2;  // [8][11][11]  = 968
    float* buf_pool2;  // [8][5][5]    = 200  (also used as flat FC input)
    float* buf_fc1;    // [64]
    float* buf_fc2;    // [10]

    void allocate_buffers();
    void free_buffers();
};
