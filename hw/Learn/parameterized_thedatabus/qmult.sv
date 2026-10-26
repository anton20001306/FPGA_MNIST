// Fixed-Point Multiplier (Sequential — registered output)
// Converts inputs to positive magnitude, multiplies, re-quantizes,
// and converts back to 2's complement if result is negative.
//
// Format: (N,Q) — 1 sign bit + (N-Q-1) integer bits + Q fractional bits
// Default: (16,12) => A(3,12)
//
// NOTE: The .md documentation shows a purely combinational version (assign-only).
//       This design follows the .v implementation which registers the result
//       for better timing closure on FPGA.

module qmult
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH,
  parameter int Q = FRAC_BITS
)(
  input  logic             clk,
  input  logic             rst,
  input  logic [N-1:0]    a,
  input  logic [N-1:0]    b,
  output logic [N-1:0]    q_result,
  output logic             overflow
);

  // Multiplication of two N-bit values produces 2N-bit result
  logic [2*N-1:0] f_result;

  logic [N-1:0] multiplicand;
  logic [N-1:0] multiplier;
  logic [N-1:0] a_2cmp, b_2cmp;
  logic [N-2:0] quantized_result, quantized_result_2cmp;

  // 2's complement of inputs (for handling negative numbers)
  assign a_2cmp = {~a[N-1], ~a[N-2:0] + 1'b1};
  assign b_2cmp = {~b[N-1], ~b[N-2:0] + 1'b1};

  // Convert to positive magnitude for multiplication
  assign multiplicand = a[N-1] ? a_2cmp : a;
  assign multiplier   = b[N-1] ? b_2cmp : b;

  // Multiply magnitudes (excluding sign bits)
  assign f_result = multiplicand[N-2:0] * multiplier[N-2:0];

  // Output sign = XOR of input signs
  assign q_result[N-1] = a[N-1] ^ b[N-1];

  // Extract and quantize result — binary point realignment
  // After multiplication of Q.Q format, binary point is at 2Q; shift right by Q
  assign quantized_result = f_result[N-2+Q:Q];

  // 2's complement of quantized result (for negative output)
  assign quantized_result_2cmp = ~quantized_result[N-2:0] + 1'b1;

  // Select positive or 2's complement based on output sign
  assign q_result[N-2:0] = (a[N-1] ^ b[N-1]) ? quantized_result_2cmp : quantized_result;

  // Overflow detection: if upper bits beyond our format are non-zero
  assign overflow = (f_result[2*N-2:N-1+Q] > 0) ? 1'b1 : 1'b0;

endmodule
