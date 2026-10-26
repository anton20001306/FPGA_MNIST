// CNN Accelerator Parameter Package
// All modules import parameters from this package to eliminate magic numbers.
// Modify values here to reconfigure the entire accelerator pipeline.

package cnn_pkg;

  // ──────────────────────────────────────────────
  // Fixed-Point Arithmetic
  // ──────────────────────────────────────────────
  parameter int DATA_WIDTH       = 16;   // N: total bit width of activations/weights
  parameter int FRAC_BITS        = 12;   // Q: fractional bits in fixed-point format
                                         // Format: 1 sign + (N-Q-1) integer + Q fractional
                                         // With (16,12): range [-8, +7.999...], precision ~0.000244

  // ──────────────────────────────────────────────
  // Convolution
  // ──────────────────────────────────────────────
  parameter int INPUT_SIZE       = 10;   // n: input activation map dimension (n x n)
  parameter int KERNEL_SIZE      = 3;    // k: convolution kernel dimension (k x k)
  parameter int STRIDE           = 1;    // s: convolution stride (horizontal = vertical)

  // Derived: convolution output size = (n - k + 1) / s
  parameter int CONV_OUT_SIZE    = (INPUT_SIZE - KERNEL_SIZE + 1) / STRIDE;

  // ──────────────────────────────────────────────
  // Activation Function
  // ──────────────────────────────────────────────
  parameter int ACT_TYPE         = 0;    // 0 = ReLU, 1 = Tanh

  // Tanh LUT configuration (only used when ACT_TYPE = 1)
  parameter int TANH_ADDR_WIDTH  = 10;   // AW: address width for tanh ROM (2^AW entries)
  parameter int TANH_DATA_WIDTH  = DATA_WIDTH; // DW: tanh LUT data width (= N)

  // ──────────────────────────────────────────────
  // Pooling
  // ──────────────────────────────────────────────
  parameter int POOL_SIZE        = 2;    // p: pooling window dimension (p x p)
  parameter int POOL_TYPE        = 1;    // 0 = average pooling, 1 = max pooling
                                         // NOTE: .md documentation uses 0 for max pooling — .v design uses 1

  // 1/p^2 in (N,Q) fixed-point format, needed for average pooling division.
  // For p=2, Q=12: 1/4 = 0.25 => 0.25 * 2^12 = 1024 => 16'b0000_0100_0000_0000
  parameter logic [DATA_WIDTH-1:0] POOL_SQR_INV = 16'b0000_0100_0000_0000;

  // Derived: pooler input size = conv output size
  parameter int POOL_IN_SIZE     = CONV_OUT_SIZE;

  // Derived: pooler output size = POOL_IN_SIZE / POOL_SIZE
  parameter int POOL_OUT_SIZE    = POOL_IN_SIZE / POOL_SIZE;

  // ──────────────────────────────────────────────
  // Pipeline
  // ──────────────────────────────────────────────
  // Pipeline fill time for convolver: (k-1)*n + (k-1) clock cycles
  parameter int CONV_PIPE_FILL   = (KERNEL_SIZE - 1) * INPUT_SIZE + (KERNEL_SIZE - 1);

  // Total kernel weight count
  parameter int KERNEL_ELEMENTS  = KERNEL_SIZE * KERNEL_SIZE;

  // Packed weight bus width
  parameter int WEIGHT_BUS_WIDTH = KERNEL_ELEMENTS * DATA_WIDTH;

endpackage
