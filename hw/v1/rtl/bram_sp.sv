// ============================================================================
// bram_sp.sv — Simple single-port memory for activation buffers
//
// v1 uses combinational read (0-cycle read latency) for straightforward
// FSM design and simulation. For synthesis targeting block RAM, change to
// registered output (add always_ff for dout) and adjust engine wait states.
// ============================================================================

module bram_sp #(
  parameter int WIDTH = 8,
  parameter int DEPTH = 1024,
  parameter int ADDR_W = $clog2(DEPTH)
)(
  input  logic               clk,
  input  logic               we,
  input  logic [ADDR_W-1:0]  addr,
  input  logic [WIDTH-1:0]   din,
  output logic [WIDTH-1:0]   dout
);

  logic [WIDTH-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we)
      mem[addr] <= din;
  end

  // Combinational read — engine FSMs assume data available 1 cycle after
  // address is driven (registered addr → combinational dout = 1 effective cycle)
  assign dout = mem[addr];

endmodule
