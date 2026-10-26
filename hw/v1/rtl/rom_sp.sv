// ============================================================================
// rom_sp.sv — Single-port ROM for weight and bias storage
//
// Initialized from a .hex file ($readmemh). Combinational read for v1.
// For synthesis, add registered output and adjust engine wait states.
// ============================================================================

module rom_sp #(
  parameter int    WIDTH    = 8,
  parameter int    DEPTH    = 1024,
  parameter int    ADDR_W   = $clog2(DEPTH),
  parameter string INIT_FILE = ""
)(
  input  logic               clk,
  input  logic               rd_en,
  input  logic [ADDR_W-1:0]  addr,
  output logic [WIDTH-1:0]   dout
);

  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, mem);
    end
  end

  // Combinational read — gated by rd_en, outputs 0 when idle
  assign dout = rd_en ? mem[addr] : '0;

endmodule
