#include "cnn.h"
#include "conv_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include "activations.h"
#include "weights_io.h"
#include "mnist_loader.h"
#include "fixed_point.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

// ---------------------------------------------------------------------------
// Layer dimension constants — identical to float version (from main_plan.md)
// ---------------------------------------------------------------------------
static const int C1_IN=1,  H1_IN=28, W1_IN=28;
static const int C1_OUT=4, K1=3;
static const int H1_OUT = H1_IN - K1 + 1;   // 26
static const int W1_OUT = W1_IN - K1 + 1;   // 26
static const int POOL1 = 2;
static const int H1P = H1_OUT / POOL1;      // 13
static const int W1P = W1_OUT / POOL1;      // 13

static const int C2_IN=C1_OUT, H2_IN=H1P, W2_IN=W1P;
static const int C2_OUT=8, K2=3;
static const int H2_OUT = H2_IN - K2 + 1;   // 11
static const int W2_OUT = W2_IN - K2 + 1;   // 11
static const int POOL2 = 2;
static const int H2P = H2_OUT / POOL2;      // 5
static const int W2P = W2_OUT / POOL2;      // 5

static const int FC1_IN  = C2_OUT * H2P * W2P;  // 200
static const int FC1_OUT = 64;
static const int FC2_IN  = FC1_OUT;              // 64
static const int FC2_OUT = 10;

// Weight tensor sizes
static const int CONV1_W_SIZE = C1_OUT * C1_IN * K1 * K1;   // 36
static const int CONV2_W_SIZE = C2_OUT * C2_IN * K2 * K2;   // 288
static const int FC1_W_SIZE   = FC1_OUT * FC1_IN;            // 12800
static const int FC2_W_SIZE   = FC2_OUT * FC2_IN;            // 640

// Activation buffer sizes
static const int BUF_CONV1 = C1_OUT * H1_OUT * W1_OUT;  // 2704
static const int BUF_POOL1 = C1_OUT * H1P    * W1P;     // 676
static const int BUF_CONV2 = C2_OUT * H2_OUT * W2_OUT;  // 968
static const int BUF_POOL2 = C2_OUT * H2P    * W2P;     // 200

// ===========================================================================
// Float helper functions for calibration (self-contained, not exported)
// These replicate the float layer logic so quantized/ is independent of float/
// ===========================================================================
static void float_conv2d(const float* input, const float* kernel,
                         const float* bias, float* output,
                         int C_in, int H_in, int W_in, int C_out, int K)
{
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    for (int n = 0; n < C_out; n++)
        for (int x = 0; x < H_out; x++)
            for (int y = 0; y < W_out; y++) {
                float acc = bias[n];
                for (int c = 0; c < C_in; c++)
                    for (int i = 0; i < K; i++)
                        for (int j = 0; j < K; j++)
                            acc += input[c*H_in*W_in + (x+i)*W_in + (y+j)]
                                 * kernel[n*C_in*K*K + c*K*K + i*K + j];
                output[n*H_out*W_out + x*W_out + y] = acc;
            }
}

static void float_relu(float* data, int size) {
    for (int i = 0; i < size; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

static void float_max_pool2d(const float* input, float* output,
                             int C, int H_in, int W_in, int pool_size)
{
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;
    for (int c = 0; c < C; c++)
        for (int x = 0; x < H_out; x++)
            for (int y = 0; y < W_out; y++) {
                float mv = -1e30f;
                for (int i = 0; i < pool_size; i++)
                    for (int j = 0; j < pool_size; j++) {
                        float v = input[c*H_in*W_in + (x*pool_size+i)*W_in + (y*pool_size+j)];
                        if (v > mv) mv = v;
                    }
                output[c*H_out*W_out + x*W_out + y] = mv;
            }
}

static void float_fc(const float* input, const float* weights,
                     const float* bias, float* output, int N, int M)
{
    for (int i = 0; i < M; i++) {
        float acc = bias[i];
        for (int j = 0; j < N; j++)
            acc += weights[i*N + j] * input[j];
        output[i] = acc;
    }
}

// Find maximum absolute value in an array
static float find_max_abs(const float* data, int size) {
    float m = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(data[i]);
        if (a > m) m = a;
    }
    return m;
}

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
QuantizedCNN::QuantizedCNN() {
    conv1_w_f = conv1_b_f = conv2_w_f = conv2_b_f = nullptr;
    fc1_w_f   = fc1_b_f   = fc2_w_f   = fc2_b_f   = nullptr;
    conv1_w_q = conv2_w_q = fc1_w_q   = fc2_w_q   = nullptr;
    conv1_b_q = conv2_b_q = fc1_b_q   = fc2_b_q   = nullptr;
    buf_input_q = buf_conv1 = buf_pool1 = buf_conv2 = buf_pool2 = buf_fc1 = nullptr;
    buf_fc2 = nullptr;
    calibrated_ = quantized_ = false;
    allocate_buffers();
}

QuantizedCNN::~QuantizedCNN() {
    free_buffers();
}

void QuantizedCNN::allocate_buffers() {
    // Float weights (for loading + calibration)
    conv1_w_f = new float[CONV1_W_SIZE];
    conv1_b_f = new float[C1_OUT];
    conv2_w_f = new float[CONV2_W_SIZE];
    conv2_b_f = new float[C2_OUT];
    fc1_w_f   = new float[FC1_W_SIZE];
    fc1_b_f   = new float[FC1_OUT];
    fc2_w_f   = new float[FC2_W_SIZE];
    fc2_b_f   = new float[FC2_OUT];

    // Quantized weights
    conv1_w_q = new int8_t[CONV1_W_SIZE];
    conv1_b_q = new int32_t[C1_OUT];
    conv2_w_q = new int8_t[CONV2_W_SIZE];
    conv2_b_q = new int32_t[C2_OUT];
    fc1_w_q   = new int8_t[FC1_W_SIZE];
    fc1_b_q   = new int32_t[FC1_OUT];
    fc2_w_q   = new int8_t[FC2_W_SIZE];
    fc2_b_q   = new int32_t[FC2_OUT];

    // Activation buffers
    buf_input_q = new int8_t[H1_IN * W1_IN];
    buf_conv1   = new int8_t[BUF_CONV1];
    buf_pool1   = new int8_t[BUF_POOL1];
    buf_conv2   = new int8_t[BUF_CONV2];
    buf_pool2   = new int8_t[BUF_POOL2];
    buf_fc1     = new int8_t[FC1_OUT];
    buf_fc2     = new int32_t[FC2_OUT];
}

void QuantizedCNN::free_buffers() {
    delete[] conv1_w_f; delete[] conv1_b_f;
    delete[] conv2_w_f; delete[] conv2_b_f;
    delete[] fc1_w_f;   delete[] fc1_b_f;
    delete[] fc2_w_f;   delete[] fc2_b_f;

    delete[] conv1_w_q; delete[] conv1_b_q;
    delete[] conv2_w_q; delete[] conv2_b_q;
    delete[] fc1_w_q;   delete[] fc1_b_q;
    delete[] fc2_w_q;   delete[] fc2_b_q;

    delete[] buf_input_q;
    delete[] buf_conv1; delete[] buf_pool1;
    delete[] buf_conv2; delete[] buf_pool2;
    delete[] buf_fc1;   delete[] buf_fc2;
}

// ===========================================================================
// Weight loading (float, from common/weights/)
// ===========================================================================
bool QuantizedCNN::load_weights(const std::string& dir) {
    auto path = [&](const char* name) { return dir + "/" + name; };
    bool ok = true;
    ok &= ::load_weights(path("conv1_weights.bin"), conv1_w_f, CONV1_W_SIZE);
    ok &= ::load_weights(path("conv1_bias.bin"),    conv1_b_f, C1_OUT);
    ok &= ::load_weights(path("conv2_weights.bin"), conv2_w_f, CONV2_W_SIZE);
    ok &= ::load_weights(path("conv2_bias.bin"),    conv2_b_f, C2_OUT);
    ok &= ::load_weights(path("fc1_weights.bin"),   fc1_w_f,   FC1_W_SIZE);
    ok &= ::load_weights(path("fc1_bias.bin"),      fc1_b_f,   FC1_OUT);
    ok &= ::load_weights(path("fc2_weights.bin"),   fc2_w_f,   FC2_W_SIZE);
    ok &= ::load_weights(path("fc2_bias.bin"),      fc2_b_f,   FC2_OUT);
    if (ok) std::cout << "[cnn] Float weights loaded from " << dir << "\n";
    return ok;
}

// ===========================================================================
// Calibration: run float forward pass to determine activation ranges
// ===========================================================================
void QuantizedCNN::calibrate(const MNISTData& data, int num_samples) {
    // Temporary float buffers for calibration
    float* cal_conv1 = new float[BUF_CONV1];
    float* cal_pool1 = new float[BUF_POOL1];
    float* cal_conv2 = new float[BUF_CONV2];
    float* cal_pool2 = new float[BUF_POOL2];
    float* cal_fc1   = new float[FC1_OUT];

    // Track max absolute activation values across all calibration images
    float max_conv1 = 0.0f, max_conv2 = 0.0f, max_fc1 = 0.0f;

    int N = std::min(num_samples, data.num_images);

    for (int n = 0; n < N; n++) {
        const float* img = data.images[n].data();

        // CONV1 + ReLU
        float_conv2d(img, conv1_w_f, conv1_b_f, cal_conv1,
                     C1_IN, H1_IN, W1_IN, C1_OUT, K1);
        float_relu(cal_conv1, BUF_CONV1);
        float m = find_max_abs(cal_conv1, BUF_CONV1);
        if (m > max_conv1) max_conv1 = m;

        // Pool1
        float_max_pool2d(cal_conv1, cal_pool1, C1_OUT, H1_OUT, W1_OUT, POOL1);

        // CONV2 + ReLU
        float_conv2d(cal_pool1, conv2_w_f, conv2_b_f, cal_conv2,
                     C2_IN, H2_IN, W2_IN, C2_OUT, K2);
        float_relu(cal_conv2, BUF_CONV2);
        m = find_max_abs(cal_conv2, BUF_CONV2);
        if (m > max_conv2) max_conv2 = m;

        // Pool2
        float_max_pool2d(cal_conv2, cal_pool2, C2_OUT, H2_OUT, W2_OUT, POOL2);

        // FC1 + ReLU
        float_fc(cal_pool2, fc1_w_f, fc1_b_f, cal_fc1, FC1_IN, FC1_OUT);
        float_relu(cal_fc1, FC1_OUT);
        m = find_max_abs(cal_fc1, FC1_OUT);
        if (m > max_fc1) max_fc1 = m;
    }

    // -----------------------------------------------------------------------
    // Compute per-layer scales
    //   scale = max_absolute_value / 127.0
    //   (symmetric quantization: real_value = int8_value × scale)
    // -----------------------------------------------------------------------
    scale_input     = 1.0f / 127.0f;  // input pixels [0,1] → int8 [0,127]

    scale_conv1_w   = find_max_abs(conv1_w_f, CONV1_W_SIZE) / 127.0f;
    scale_conv1_out = max_conv1 / 127.0f;

    scale_conv2_w   = find_max_abs(conv2_w_f, CONV2_W_SIZE) / 127.0f;
    scale_conv2_out = max_conv2 / 127.0f;

    scale_fc1_w     = find_max_abs(fc1_w_f, FC1_W_SIZE) / 127.0f;
    scale_fc1_out   = max_fc1 / 127.0f;

    scale_fc2_w     = find_max_abs(fc2_w_f, FC2_W_SIZE) / 127.0f;

    // -----------------------------------------------------------------------
    // Compute requantization multipliers
    //   M = (scale_weight × scale_input_activation) / scale_output_activation
    //   Each M is stored as RequantParams: multiplier (int32) + shift (int)
    // -----------------------------------------------------------------------
    float M;

    // Conv1: input scale = scale_input (pixels), output scale = scale_conv1_out
    M = (scale_conv1_w * scale_input) / scale_conv1_out;
    rq_conv1 = compute_requant_params(M);

    // Conv2: input scale = scale_conv1_out (pool doesn't change scale)
    M = (scale_conv2_w * scale_conv1_out) / scale_conv2_out;
    rq_conv2 = compute_requant_params(M);

    // FC1: input scale = scale_conv2_out (pool doesn't change scale)
    M = (scale_fc1_w * scale_conv2_out) / scale_fc1_out;
    rq_fc1 = compute_requant_params(M);

    // FC2: no requantization (raw int32 logits)

    delete[] cal_conv1; delete[] cal_pool1;
    delete[] cal_conv2; delete[] cal_pool2;
    delete[] cal_fc1;

    calibrated_ = true;
    std::cout << "[cnn] Calibration complete (" << N << " images). Activation ranges captured.\n";
}

// ===========================================================================
// Quantize model: float weights → int8, float biases → int32
// ===========================================================================
void QuantizedCNN::quantize_model() {
    if (!calibrated_) {
        std::cerr << "[cnn] ERROR: must calibrate() before quantize_model().\n";
        return;
    }

    // --- Quantize weights: float → int8 ---
    auto qw = [](const float* src, int8_t* dst, int size, float scale) {
        for (int i = 0; i < size; i++) {
            int32_t q = (int32_t)roundf(src[i] / scale);
            if (q >  127) q = 127;
            if (q < -128) q = -128;
            dst[i] = (int8_t)q;
        }
    };

    qw(conv1_w_f, conv1_w_q, CONV1_W_SIZE, scale_conv1_w);
    qw(conv2_w_f, conv2_w_q, CONV2_W_SIZE, scale_conv2_w);
    qw(fc1_w_f,   fc1_w_q,   FC1_W_SIZE,   scale_fc1_w);
    qw(fc2_w_f,   fc2_w_q,   FC2_W_SIZE,   scale_fc2_w);

    // --- Quantize biases: float → int32 (in accumulator scale) ---
    // bias_q = round(bias_float / (scale_weight × scale_input_activation))
    auto qb = [](const float* src, int32_t* dst, int size,
                 float scale_w, float scale_in) {
        float acc_scale = scale_w * scale_in;
        for (int i = 0; i < size; i++)
            dst[i] = (int32_t)roundf(src[i] / acc_scale);
    };

    qb(conv1_b_f, conv1_b_q, C1_OUT,   scale_conv1_w, scale_input);
    qb(conv2_b_f, conv2_b_q, C2_OUT,   scale_conv2_w, scale_conv1_out);
    qb(fc1_b_f,   fc1_b_q,   FC1_OUT,  scale_fc1_w,   scale_conv2_out);
    qb(fc2_b_f,   fc2_b_q,   FC2_OUT,  scale_fc2_w,   scale_fc1_out);

    // Free float weights (no longer needed)
    delete[] conv1_w_f; conv1_w_f = nullptr;
    delete[] conv1_b_f; conv1_b_f = nullptr;
    delete[] conv2_w_f; conv2_w_f = nullptr;
    delete[] conv2_b_f; conv2_b_f = nullptr;
    delete[] fc1_w_f;   fc1_w_f = nullptr;
    delete[] fc1_b_f;   fc1_b_f = nullptr;
    delete[] fc2_w_f;   fc2_w_f = nullptr;
    delete[] fc2_b_f;   fc2_b_f = nullptr;

    quantized_ = true;
    std::cout << "[cnn] Model quantized: float32 → int8 weights, int32 biases.\n";
}

// ===========================================================================
// Quantized forward pass — PURE INTEGER arithmetic (FPGA-equivalent)
// ===========================================================================
int QuantizedCNN::forward(const float* input) {
    // Step 0: Quantize input pixels  float [0,1] → int8 [0,127]
    for (int i = 0; i < H1_IN * W1_IN; i++) {
        int32_t q = (int32_t)roundf(input[i] / scale_input);
        if (q > 127) q = 127;
        if (q < 0)   q = 0;
        buf_input_q[i] = (int8_t)q;
    }

    // ===== From here, everything is integer-only =====

    // CONV1 + ReLU (fused into requantization)
    conv2d_q(buf_input_q, conv1_w_q, conv1_b_q, buf_conv1,
             C1_IN, H1_IN, W1_IN, C1_OUT, K1,
             rq_conv1, /*apply_relu=*/true);

    // MaxPool1
    max_pool2d_q(buf_conv1, buf_pool1, C1_OUT, H1_OUT, W1_OUT, POOL1);

    // CONV2 + ReLU (fused)
    conv2d_q(buf_pool1, conv2_w_q, conv2_b_q, buf_conv2,
             C2_IN, H2_IN, W2_IN, C2_OUT, K2,
             rq_conv2, /*apply_relu=*/true);

    // MaxPool2
    max_pool2d_q(buf_conv2, buf_pool2, C2_OUT, H2_OUT, W2_OUT, POOL2);

    // FC1 + ReLU (fused)
    fc_forward_q(buf_pool2, fc1_w_q, fc1_b_q, buf_fc1,
                 FC1_IN, FC1_OUT,
                 rq_fc1, /*apply_relu=*/true);

    // FC2 — raw int32 logits (no requantization)
    fc_forward_q_raw(buf_fc1, fc2_w_q, fc2_b_q, buf_fc2,
                     FC2_IN, FC2_OUT);

    // Argmax on int32 logits
    return argmax_q(buf_fc2, FC2_OUT);
}

// ===========================================================================
// Print quantization parameters (useful for FPGA export / debugging)
// ===========================================================================
void QuantizedCNN::print_quant_params() const {
    std::cout << "\n[quant] === Quantization Parameters ===\n";
    std::cout << "[quant] Input:  scale=" << scale_input << "\n";
    std::cout << "[quant] CONV1:  scale_w=" << scale_conv1_w
              << "  scale_out=" << scale_conv1_out
              << "  M_float=" << (scale_conv1_w * scale_input / scale_conv1_out)
              << "  multiplier=" << rq_conv1.multiplier
              << "  shift=" << rq_conv1.shift << "\n";
    std::cout << "[quant] CONV2:  scale_w=" << scale_conv2_w
              << "  scale_out=" << scale_conv2_out
              << "  M_float=" << (scale_conv2_w * scale_conv1_out / scale_conv2_out)
              << "  multiplier=" << rq_conv2.multiplier
              << "  shift=" << rq_conv2.shift << "\n";
    std::cout << "[quant] FC1:    scale_w=" << scale_fc1_w
              << "  scale_out=" << scale_fc1_out
              << "  M_float=" << (scale_fc1_w * scale_conv2_out / scale_fc1_out)
              << "  multiplier=" << rq_fc1.multiplier
              << "  shift=" << rq_fc1.shift << "\n";
    std::cout << "[quant] FC2:    scale_w=" << scale_fc2_w
              << "  (no requant — raw int32 logits)\n";
    std::cout << "[quant] ===============================\n\n";
}
