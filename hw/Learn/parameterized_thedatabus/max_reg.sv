// Max Register
// Holds the current maximum (or running sum for average pooling).
// Resets on master_rst or rst_m (per-neighborhood reset).

module max_reg
  import cnn_pkg::*;
#(
  parameter int N = DATA_WIDTH
)(
  input  logic           clk,
  input  logic           ce,
  input  logic [N-1:0]  din,
  input  logic           rst_m,      // per-neighborhood reset
  input  logic           master_rst, // global reset
  output logic [N-1:0]  reg_op
);

  always_ff @(posedge clk) begin
    if (master_rst)
      reg_op <= '0;
    else if (ce) begin
      if (rst_m)
        reg_op <= '0;
      else
        reg_op <= din;
    end
  end

endmodule
