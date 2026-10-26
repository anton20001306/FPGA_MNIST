// MAC (Multiply-Accumulate) Unit — Fixed-Point
// Implements: p = a * b + c (registered output)
//
// Uses qmult for fixed-point multiplication and qadd for accumulation.
// The .v file has ifdef FIXED_POINT selecting between integer and fixed-point modes.
// This SV version implements the fixed-point mode only (the design target).
//
// Additional note: The .md shows a separate mac_manual_fp module for fixed-point;
//                  the .v combines both modes via ifdef in a single mac_manual.

module mac_manual
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH,
  parameter int Q = FRAC_BITS
)(
  input  logic           clk,
  input  logic           sclr,  // synchronous clear
  input  logic           ce,    // clock enable
  input  logic [N-1:0]  a,     // activation
  input  logic [N-1:0]  b,     // weight
  input  logic [N-1:0]  c,     // previous MAC sum
  output logic [N-1:0]  p      // result
);

  logic [N-1:0] mult_out;
  logic [N-1:0] add_out;
  logic         ovr;

  qmult #(.N(N), .Q(Q)) u_mul (
    .clk      (clk),
    .rst      (sclr),
    .a        (a),
    .b        (b),
    .q_result (mult_out),
    .overflow (ovr)
  );

  qadd #(.N(N), .Q(Q)) u_add (
    .a (mult_out),
    .b (c),
    .c (add_out)
  );

  always_ff @(posedge clk or posedge sclr) begin
    if (sclr)
      p <= '0;
    else if (ce)
      p <= add_out;
  end

endmodule
