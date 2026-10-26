#pragma once
#include <string>
#include <cstdint>
#include "fixed_point.h"

struct MNISTData;  // forward declaration (from mnist_loader.h)

// ---------------------------------------------------------------------------
// cnn.h — QuantizedCNN
//
// Same architecture as the float TinyCNN, but all inference arithmetic is
// integer-only (int8 weights/activations, int32 accumulators).
//
// Usage:
//   1. load_weights()    — load float32 .bin files from disk
//   2. calibrate()       — run float forward on N images to find activation ranges
//   3. quantize_model()  — convert all weights/biases to integer, compute requant params
//   4. forward()         — run pure-integer inference (FPGA-equivalent)
//
// Architecture (identical to float version):
//   Input    28×28×1
//   CONV1    3×3, 4 filters  → 26×26×4  + ReLU  (fused into requantize)
//   MaxPool1 2×2             → 13×13×4
//   CONV2    3×3, 8 filters  → 11×11×8  + ReLU  (fused into requantize)
//   MaxPool2 2×2             →  5×5×8
//   Flatten                  → 200
//   FC1      200→64          + ReLU              (fused into requantize)
//   FC2      64→10           → int32 logits      (no requantize)
//   argmax                   → predicted digit 0–9
// ---------------------------------------------------------------------------

class QuantizedCNN {
public:
    QuantizedCNN();
    ~QuantizedCNN();

    // Load float weights from binary files. These will be quantized after calibration.
    bool load_weights(const std::string& weights_dir);

    // Calibrate: run float forward pass on first N images to determine
    // per-layer activation scales. Must be called after load_weights().
    void calibrate(const MNISTData& data, int num_samples = 100);

    // Convert float weights → int8, float biases → int32.
    // Computes RequantParams for each layer. Must be called after calibrate().
    void quantize_model();

    // Run quantized forward pass.
    // `input` is 784 floats in [0,1] (same as float model).
    // Internally quantizes to int8, then all arithmetic is integer-only.
    // Returns predicted digit (0–9).
    int forward(const float* input);

    // Print all quantization parameters (for debugging / FPGA export)
    void print_quant_params() const;

private:
    // --- Float weights (loaded from disk, freed after quantize_model()) ---
    float *conv1_w_f, *conv1_b_f;
    float *conv2_w_f, *conv2_b_f;
    float *fc1_w_f,   *fc1_b_f;
    float *fc2_w_f,   *fc2_b_f;

    // --- Quantized weights (int8) and biases (int32) ---
    int8_t  *conv1_w_q, *conv2_w_q, *fc1_w_q, *fc2_w_q;
    int32_t *conv1_b_q, *conv2_b_q, *fc1_b_q, *fc2_b_q;

    // --- Activation buffers (all integer) ---
    int8_t  *buf_input_q;  // quantized input [784]
    int8_t  *buf_conv1;    // [4][26][26]  = 2704
    int8_t  *buf_pool1;    // [4][13][13]  = 676
    int8_t  *buf_conv2;    // [8][11][11]  = 968
    int8_t  *buf_pool2;    // [8][5][5]    = 200
    int8_t  *buf_fc1;      // [64]
    int32_t *buf_fc2;      // [10]  — raw logits, no requantization

    // --- Per-layer quantization scales ---
    float scale_input;      // input pixel scale: 1.0/127.0
    float scale_conv1_w;    // conv1 weight scale
    float scale_conv1_out;  // conv1 output activation scale (after ReLU)
    float scale_conv2_w;
    float scale_conv2_out;
    float scale_fc1_w;
    float scale_fc1_out;
    float scale_fc2_w;
    // No output scale for FC2 (raw int32 logits)

    // --- Requantization parameters (integer-only, FPGA-ready) ---
    RequantParams rq_conv1;   // conv1: int32 acc → int8
    RequantParams rq_conv2;   // conv2: int32 acc → int8
    RequantParams rq_fc1;     // fc1:   int32 acc → int8
    // FC2: no requantization

    bool calibrated_;
    bool quantized_;

    void allocate_buffers();
    void free_buffers();
};
