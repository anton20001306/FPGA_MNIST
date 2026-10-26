// Input Multiplexer for Pooler
// sel=00: pass ip2 (max_reg output)
// sel=01: pass ip1 (shift register output)
// sel=10: output zero (reset path)
//
// NOTE: .md shows sel=00 -> ip2, sel=01 -> ip1, sel=10 -> 0.
//       .v file matches this behavior.

module input_mux
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH
)(
  input  logic [N-1:0]  ip1,  // shift register output (sr_op)
  input  logic [N-1:0]  ip2,  // max register output (max_reg_op)
  input  logic [1:0]    sel,
  output logic [N-1:0]  op
);

  assign op = (sel == 2'b01) ? ip1 :
              (sel == 2'b00) ? ip2 : '0;

endmodule
