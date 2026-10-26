// Parameterized Variable-Length Shift Register
// Used for row buffering in both convolver and pooler pipelines.

module variable_shift_reg
  import cnn_pkg::*;
#(
  parameter int WIDTH = DATA_WIDTH,
  parameter int SIZE  = 3
)(
  input  logic               clk,
  input  logic               ce,
  input  logic               rst,
  input  logic [WIDTH-1:0]   d,
  output logic [WIDTH-1:0]   out
);

  logic [WIDTH-1:0] sr [SIZE-1:0];

  for (genvar i = 0; i < SIZE; i++) begin : g_shift
    always_ff @(posedge clk or posedge rst) begin
      if (rst)
        sr[i] <= '0;
      else if (ce) begin
        if (i == 0)
          sr[i] <= d;
        else
          sr[i] <= sr[i-1];
      end
    end
  end

  assign out = sr[SIZE-1];

endmodule
