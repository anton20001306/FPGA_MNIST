#include "cnn.h"
#include "conv_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include "activations.h"
#include "weights_io.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>

// ---------------------------------------------------------------------------
// Layer dimension constants — derived from main_plan.md architecture table.
// Change these here if the architecture is modified; everywhere else adapts.
// ---------------------------------------------------------------------------

// CONV1: 28×28×1 → 26×26×4 (3×3 kernel, 4 filters)
static const int C1_IN=1,  H1_IN=28, W1_IN=28;
static const int C1_OUT=4, K1=3;
static const int H1_OUT = H1_IN - K1 + 1;  // 26
static const int W1_OUT = W1_IN - K1 + 1;  // 26

// Pool1: 26×26×4 → 13×13×4
static const int POOL1 = 2;
static const int H1P = H1_OUT / POOL1;  // 13
static const int W1P = W1_OUT / POOL1;  // 13

// CONV2: 13×13×4 → 11×11×8 (3×3 kernel, 8 filters)
static const int C2_IN=C1_OUT, H2_IN=H1P, W2_IN=W1P;
static const int C2_OUT=8, K2=3;
static const int H2_OUT = H2_IN - K2 + 1;  // 11
static const int W2_OUT = W2_IN - K2 + 1;  // 11

// Pool2: 11×11×8 → 5×5×8
static const int POOL2 = 2;
static const int H2P = H2_OUT / POOL2;  // 5
static const int W2P = W2_OUT / POOL2;  // 5

// FC1: 200 → 64
static const int FC1_IN  = C2_OUT * H2P * W2P;  // 200
static const int FC1_OUT = 64;

// FC2: 64 → 10
static const int FC2_IN  = FC1_OUT;   // 64
static const int FC2_OUT = 10;

// Weight sizes
static const int CONV1_W_SIZE = C1_OUT * C1_IN * K1 * K1;  // 36
static const int CONV2_W_SIZE = C2_OUT * C2_IN * K2 * K2;  // 288
static const int FC1_W_SIZE   = FC1_OUT * FC1_IN;           // 12800
static const int FC2_W_SIZE   = FC2_OUT * FC2_IN;           // 640

// Buffer sizes
static const int BUF_CONV1 = C1_OUT * H1_OUT * W1_OUT;  // 2704
static const int BUF_POOL1 = C1_OUT * H1P    * W1P;     // 676
static const int BUF_CONV2 = C2_OUT * H2_OUT * W2_OUT;  // 968
static const int BUF_POOL2 = C2_OUT * H2P    * W2P;     // 200

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
TinyCNN::TinyCNN() {
    conv1_w = conv1_b = nullptr;
    conv2_w = conv2_b = nullptr;
    fc1_w   = fc1_b   = nullptr;
    fc2_w   = fc2_b   = nullptr;
    buf_conv1 = buf_pool1 = buf_conv2 = buf_pool2 = nullptr;
    buf_fc1   = buf_fc2   = nullptr;
    allocate_buffers();
}

TinyCNN::~TinyCNN() {
    free_buffers();
}

void TinyCNN::allocate_buffers() {
    // Weights
    conv1_w = new float[CONV1_W_SIZE];
    conv1_b = new float[C1_OUT];
    conv2_w = new float[CONV2_W_SIZE];
    conv2_b = new float[C2_OUT];
    fc1_w   = new float[FC1_W_SIZE];
    fc1_b   = new float[FC1_OUT];
    fc2_w   = new float[FC2_W_SIZE];
    fc2_b   = new float[FC2_OUT];

    // Intermediate activations
    buf_conv1 = new float[BUF_CONV1];
    buf_pool1 = new float[BUF_POOL1];
    buf_conv2 = new float[BUF_CONV2];
    buf_pool2 = new float[BUF_POOL2];
    buf_fc1   = new float[FC1_OUT];
    buf_fc2   = new float[FC2_OUT];
}

void TinyCNN::free_buffers() {
    delete[] conv1_w; delete[] conv1_b;
    delete[] conv2_w; delete[] conv2_b;
    delete[] fc1_w;   delete[] fc1_b;
    delete[] fc2_w;   delete[] fc2_b;
    delete[] buf_conv1; delete[] buf_pool1;
    delete[] buf_conv2; delete[] buf_pool2;
    delete[] buf_fc1;   delete[] buf_fc2;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------
bool TinyCNN::load_weights(const std::string& dir) {
    // Use ::load_weights to call the free function from weights_io.h,
    // not this member function recursively.
    auto path = [&](const char* name) { return dir + "/" + name; };
    bool ok = true;
    ok &= ::load_weights(path("conv1_weights.bin"), conv1_w, CONV1_W_SIZE);
    ok &= ::load_weights(path("conv1_bias.bin"),    conv1_b, C1_OUT);
    ok &= ::load_weights(path("conv2_weights.bin"), conv2_w, CONV2_W_SIZE);
    ok &= ::load_weights(path("conv2_bias.bin"),    conv2_b, C2_OUT);
    ok &= ::load_weights(path("fc1_weights.bin"),   fc1_w,   FC1_W_SIZE);
    ok &= ::load_weights(path("fc1_bias.bin"),      fc1_b,   FC1_OUT);
    ok &= ::load_weights(path("fc2_weights.bin"),   fc2_w,   FC2_W_SIZE);
    ok &= ::load_weights(path("fc2_bias.bin"),      fc2_b,   FC2_OUT);
    if (ok) std::cout << "[cnn] All weights loaded from " << dir << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Random weight initialisation (He initialisation).
// Useful for running a shape smoke-test before trained weights are available.
// He init: weights ~ N(0, sqrt(2 / fan_in))
// ---------------------------------------------------------------------------
static float randn() {
    // Box-Muller transform for a standard normal sample
    float u1 = (float(rand()) + 1.0f) / (float(RAND_MAX) + 1.0f);
    float u2 = (float(rand()) + 1.0f) / (float(RAND_MAX) + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void he_init(float* w, int size, int fan_in) {
    float std_dev = sqrtf(2.0f / float(fan_in));
    for (int i = 0; i < size; i++)
        w[i] = randn() * std_dev;
}

void TinyCNN::init_random_weights() {
    srand(42);  // fixed seed for reproducibility
    he_init(conv1_w, CONV1_W_SIZE, C1_IN * K1 * K1);
    he_init(conv2_w, CONV2_W_SIZE, C2_IN * K2 * K2);
    he_init(fc1_w,   FC1_W_SIZE,   FC1_IN);
    he_init(fc2_w,   FC2_W_SIZE,   FC2_IN);
    // Biases initialised to zero
    memset(conv1_b, 0, C1_OUT  * sizeof(float));
    memset(conv2_b, 0, C2_OUT  * sizeof(float));
    memset(fc1_b,   0, FC1_OUT * sizeof(float));
    memset(fc2_b,   0, FC2_OUT * sizeof(float));
    std::cout << "[cnn] Weights initialised randomly (He init, seed=42)\n";
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
int TinyCNN::forward(const float* input) {
    return forward_verbose(input, nullptr);
}

int TinyCNN::forward_verbose(const float* input, float* logits_out) {
    // CONV1 + ReLU
    conv2d(input,     conv1_w, conv1_b, buf_conv1,
           C1_IN,  H1_IN,  W1_IN,  C1_OUT, K1);
    relu(buf_conv1, BUF_CONV1);

    // MaxPool1
    max_pool2d(buf_conv1, buf_pool1, C1_OUT, H1_OUT, W1_OUT, POOL1);

    // CONV2 + ReLU
    conv2d(buf_pool1, conv2_w, conv2_b, buf_conv2,
           C2_IN,  H2_IN,  W2_IN,  C2_OUT, K2);
    relu(buf_conv2, BUF_CONV2);

    // MaxPool2  →  buf_pool2 is also the flat FC input (200 values)
    max_pool2d(buf_conv2, buf_pool2, C2_OUT, H2_OUT, W2_OUT, POOL2);

    // FC1 + ReLU
    fc_forward(buf_pool2, fc1_w, fc1_b, buf_fc1, FC1_IN, FC1_OUT);
    relu(buf_fc1, FC1_OUT);

    // FC2  (raw logits, no activation)
    fc_forward(buf_fc1, fc2_w, fc2_b, buf_fc2, FC2_IN, FC2_OUT);

    if (logits_out)
        memcpy(logits_out, buf_fc2, FC2_OUT * sizeof(float));

    return argmax(buf_fc2, FC2_OUT);
}
