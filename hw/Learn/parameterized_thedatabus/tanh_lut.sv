// Tanh Activation Function — LUT with Linear Interpolation
//
// Exploits tanh properties:
//   - Symmetry: store only positive half; negate for negative inputs
//   - Saturation: for |x| > threshold, directly output +/-1
//
// Uses two LUT reads and linear interpolation for improved precision:
//   f(ai + alpha) ~ alpha * f(ai) + (1-alpha) * f(ai+1)
//
// The .v file combines simple LUT and interpolated modes via `ifdef INTERPOLATE`.
// This SV version implements the interpolated mode (higher precision).
//
// Additional note: .md simple version has no rst input; this design includes it
// following the .v implementation.

module tanh_lut
  import cnn_pkg::*;
#(
  parameter int AW = TANH_ADDR_WIDTH,  // address width (ROM = 2^AW entries)
  parameter int DW = TANH_DATA_WIDTH,  // data width
  parameter int N  = DATA_WIDTH,       // total bit width
  parameter int Q  = FRAC_BITS         // fractional bits
)(
  input  logic             clk,
  input  logic             rst,
  input  logic [N-1:0]    phase,
  output logic [DW-1:0]   tanh
);

  logic [AW-1:0] addra_reg;
  logic [AW-1:0] addrb_reg;
  logic [DW-1:0] tanha;
  logic [DW-1:0] tanhb;
  logic          ovr1, ovr2;

  // ROM — synthesis attribute for block RAM inference
  (* ram_style = "block" *) logic [DW-1:0] mem [0:(1 << AW)-1];

  initial begin
    $readmemb("tanh_data.mem", mem);
  end

  // Address registration
  always_ff @(posedge clk) begin
    addra_reg <= phase[AW-1:0];
    addrb_reg <= phase[AW-1:0] + 1'b1;  // next entry for interpolation
  end

  assign tanha = mem[addra_reg];
  assign tanhb = mem[addrb_reg];

  // Fractional part from LSBs not used in LUT addressing
  localparam int FRAC_LSB_WIDTH = N - AW - 2;  // sign bit + 1 integer bit already consumed
  logic [N-1:0] frac;
  logic [N-1:0] one_minus_frac;
  logic [N-1:0] p1, p2;
  logic [DW-1:0] tanh_temp;

  // 1.0 in (N,Q) format: bit pattern with integer '1' at position Q
  localparam logic [N-1:0] ONE_FP = N'(1 << Q);

  assign frac           = {{(N - FRAC_LSB_WIDTH){1'b0}}, phase[FRAC_LSB_WIDTH-1:0]};
  assign one_minus_frac = ONE_FP - frac;

  // Linear interpolation: alpha * f(Ai) + (1-alpha) * f(Ai+1)
  qmult #(.N(N), .Q(Q)) u_mul1 (
    .clk      (clk),
    .rst      (rst),
    .a        (tanha),
    .b        (frac),
    .q_result (p1),
    .overflow (ovr1)
  );

  qmult #(.N(N), .Q(Q)) u_mul2 (
    .clk      (clk),
    .rst      (rst),
    .a        (tanhb),
    .b        (one_minus_frac),
    .q_result (p2),
    .overflow (ovr2)
  );

  assign tanh_temp = p1 + p2;

  // Saturation and sign handling:
  // phase[N-1] = sign bit
  // phase[N-2] = indicates magnitude >= 4 (saturation region)
  //
  // Positive, magnitude < 4:  output tanh_temp
  // Positive, magnitude >= 4: output +1 (ONE_FP)
  // Negative, magnitude < 4:  output -tanh_temp (2's complement)
  // Negative, magnitude >= 4: output -1
  localparam logic [N-1:0] NEG_ONE_FP = ~ONE_FP + 1'b1;  // -1.0 in (N,Q)

  assign tanh = phase[N-1] ?
                  (phase[N-2] ? NEG_ONE_FP : (~tanh_temp + 1'b1)) :
                  (phase[N-2] ? ONE_FP     : tanh_temp);

endmodule
