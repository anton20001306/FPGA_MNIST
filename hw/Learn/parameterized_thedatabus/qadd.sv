// Fixed-Point Adder
// Since negative numbers are already in 2's complement form,
// simple addition handles both positive and negative operands.
//
// Format: (N,Q) — 1 sign bit + (N-Q-1) integer bits + Q fractional bits
// Default: (16,12) => A(3,12)

module qadd
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH,
  parameter int Q = FRAC_BITS
)(
  input  logic [N-1:0] a,
  input  logic [N-1:0] b,
  output logic [N-1:0] c
);

  assign c = a + b;

endmodule
