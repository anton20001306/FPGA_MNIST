// ReLU Activation Function
// f(x) = max(0, x)
// MSB is sign bit in 2's complement: if negative, output 0; else pass through.
//
// NOTE: .md documentation uses default N=32; .v design uses N=16.
//       This follows the .v value (matches DATA_WIDTH from cnn_pkg).

module relu
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH
)(
  input  logic [N-1:0] din_relu,
  output logic [N-1:0] dout_relu
);

  assign dout_relu = (din_relu[N-1] == 1'b0) ? din_relu : '0;

endmodule
