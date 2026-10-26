// ============================================================================
// cnn_pkg.sv — Parameter package for MNIST Tiny CNN (quantized int8)
//
// All layer dimensions, buffer sizes, and weight sizes are defined here.
// These match the C++ quantized model in sw/quantized/cnn.cpp exactly.
//
// Arithmetic: int8 activations, int8 weights, int32 accumulators, int32 biases
//             Requantization: int32 acc → int8 via (acc * multiplier) >> shift
// ============================================================================

package cnn_pkg;

  // ─── Data types ───
  parameter int ACT_WIDTH   = 8;    // int8 activations and weights
  parameter int ACC_WIDTH   = 32;   // int32 accumulator and biases
  parameter int MULT_WIDTH  = 32;   // requantization multiplier width
  parameter int PROD_WIDTH  = 64;   // acc * multiplier product width

  // ─── Input image ───
  parameter int IMG_H       = 28;
  parameter int IMG_W       = 28;
  parameter int IMG_C       = 1;    // grayscale
  parameter int IMG_SIZE    = IMG_H * IMG_W;  // 784

  // ─── CONV1: 3x3, 1→4 channels ───
  parameter int C1_IN       = IMG_C;          // 1
  parameter int C1_OUT      = 4;              // channels
  parameter int K1          = 3;              // kernel size
  parameter int H1_IN       = IMG_H;          // 28
  parameter int W1_IN       = IMG_W;          // 28
  parameter int H1_OUT      = H1_IN - K1 + 1; // 26
  parameter int W1_OUT      = W1_IN - K1 + 1; // 26
  parameter int CONV1_W_SIZE = C1_OUT * C1_IN * K1 * K1;  // 36

  // ─── MaxPool1: 2x2 ───
  parameter int POOL1       = 2;
  parameter int H1P         = H1_OUT / POOL1; // 13
  parameter int W1P         = W1_OUT / POOL1; // 13

  // ─── CONV2: 3x3, 4→8 channels ───
  parameter int C2_IN       = C1_OUT;         // 4
  parameter int C2_OUT      = 8;              // channels
  parameter int K2          = 3;              // kernel size  
  parameter int H2_IN       = H1P;            // 13
  parameter int W2_IN       = W1P;            // 13
  parameter int H2_OUT      = H2_IN - K2 + 1; // 11
  parameter int W2_OUT      = W2_IN - K2 + 1; // 11
  parameter int CONV2_W_SIZE = C2_OUT * C2_IN * K2 * K2;  // 288

  // ─── MaxPool2: 2x2 ───
  parameter int POOL2       = 2;
  parameter int H2P         = H2_OUT / POOL2; // 5
  parameter int W2P         = W2_OUT / POOL2; // 5

  // ─── FC1: 200→64 ───
  parameter int FC1_IN      = C2_OUT * H2P * W2P;  // 200
  parameter int FC1_OUT     = 64;
  parameter int FC1_W_SIZE  = FC1_OUT * FC1_IN;     // 12800

  // ─── FC2: 64→10 ───
  parameter int FC2_IN      = FC1_OUT;              // 64
  parameter int FC2_OUT     = 10;
  parameter int FC2_W_SIZE  = FC2_OUT * FC2_IN;     // 640

  // ─── Activation buffer sizes (for BRAM allocation) ───
  parameter int BUF_INPUT   = IMG_SIZE;                     // 784
  parameter int BUF_CONV1   = C1_OUT * H1_OUT * W1_OUT;    // 2704
  parameter int BUF_POOL1   = C1_OUT * H1P * W1P;          // 676
  parameter int BUF_CONV2   = C2_OUT * H2_OUT * W2_OUT;    // 968
  parameter int BUF_POOL2   = C2_OUT * H2P * W2P;          // 200
  parameter int BUF_FC1     = FC1_OUT;                      // 64
  parameter int BUF_FC2     = FC2_OUT;                      // 10

  // Largest activation buffer (determines BRAM depth)
  parameter int MAX_ACT_BUF = BUF_CONV1;  // 2704

  // Total weight storage
  parameter int TOTAL_WEIGHTS = CONV1_W_SIZE + CONV2_W_SIZE +
                                FC1_W_SIZE + FC2_W_SIZE;     // 13764
  // Total bias storage (int32)
  parameter int TOTAL_BIASES  = C1_OUT + C2_OUT + FC1_OUT + FC2_OUT;  // 86

  // ─── Number of classes ───
  parameter int NUM_CLASSES = FC2_OUT;  // 10

  // ─── Address width helpers ───
  function automatic int clog2(input int value);
    int result = 0;
    int v = value - 1;
    while (v > 0) begin
      v = v >> 1;
      result++;
    end
    return result;
  endfunction

endpackage
