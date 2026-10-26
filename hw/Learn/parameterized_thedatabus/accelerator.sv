// CNN Accelerator — Top-Level Module
// Integrates: Convolver -> Activation (ReLU/Tanh) -> Pooler
//
// All parameters sourced from cnn_pkg. No magic numbers.
//
// Additional notes vs .md:
//   - .v has extra debug ports (conv_out, conv_valid, conv_end) — retained here
//   - .md instantiates 'tanh_normal'; .v instantiates 'tanh_lut' — following .v
//   - .v had hardcoded [15:0] for activation/data_out ports — parameterized here

module accelerator
  import cnn_pkg::*;
#(
  parameter int N_IMG       = INPUT_SIZE,
  parameter int K           = KERNEL_SIZE,
  parameter int P           = POOL_SIZE,
  parameter int S           = STRIDE,
  parameter int PTYPE       = POOL_TYPE,
  parameter int ACT         = ACT_TYPE,
  parameter int N           = DATA_WIDTH,
  parameter int Q           = FRAC_BITS,
  parameter int AW          = TANH_ADDR_WIDTH,
  parameter int DW          = TANH_DATA_WIDTH,
  parameter logic [N-1:0] P_SQR_INV = POOL_SQR_INV
)(
  input  logic               clk,
  input  logic               global_rst,
  input  logic               ce,
  input  logic [N-1:0]      activation,
  input  logic [K*K*N-1:0]  weight1,
  output logic [N-1:0]      data_out,
  output logic               valid_op,
  output logic               end_op,
  // Debug / observation ports (not in .md, present in .v)
  output logic [N-1:0]      conv_out,
  output logic               conv_valid,
  output logic               conv_end
);

  logic [N-1:0] conv_op;
  logic         valid_conv_w, end_conv_w;
  logic         valid_ip;
  logic [N-1:0] relu_op;
  logic [N-1:0] tanh_op;
  logic [N-1:0] pooler_ip;

  // Convolution output size
  localparam int CONV_OUT_DIM = N_IMG - K + 1;

  // ══════════════ CONVOLUTION ══════════════
  convolver #(
    .N_IMG (N_IMG),
    .K     (K),
    .S     (S),
    .N     (N),
    .Q     (Q)
  ) u_conv (
    .clk        (clk),
    .ce         (ce),
    .global_rst (global_rst),
    .activation (activation),
    .weight1    (weight1),
    .conv_op    (conv_op),
    .valid_conv (valid_conv_w),
    .end_conv   (end_conv_w)
  );

  // Debug port assignments
  assign conv_out   = conv_op;
  assign conv_valid = valid_conv_w;
  assign conv_end   = end_conv_w;

  assign valid_ip = valid_conv_w && (!end_conv_w);

  // ══════════════ ACTIVATION ══════════════
  // ReLU
  relu #(.N(N)) u_relu (
    .din_relu  (conv_op),
    .dout_relu (relu_op)
  );

  // Tanh (used when ACT=1)
  tanh_lut #(.AW(AW), .DW(DW), .N(N), .Q(Q)) u_tanh (
    .clk   (clk),
    .rst   (global_rst),
    .phase (conv_op),
    .tanh  (tanh_op)
  );

  // Select activation function
  assign pooler_ip = ACT ? tanh_op : relu_op;

  // ══════════════ POOLING ══════════════
  pooler #(
    .M         (CONV_OUT_DIM),
    .P         (P),
    .N         (N),
    .Q         (Q),
    .PTYPE     (PTYPE),
    .P_SQR_INV (P_SQR_INV)
  ) u_pool (
    .clk        (clk),
    .ce         (valid_ip),
    .master_rst (global_rst),
    .data_in    (pooler_ip),
    .data_out   (data_out),
    .valid_op   (valid_op),
    .end_op     (end_op)
  );

endmodule
