// Pooling Unit — Streaming Max/Average Pooler
//
// Performs p x p pooling on an m x m input (from convolver output).
// Output size: (m/p) x (m/p)
//
// ptype=1: Max pooling (default, following .v)
// ptype=0: Average pooling (divides sum by p^2 using p_sqr_inv)
//
// NOTE: .md documentation uses ptype=0 for max pooling — .v design uses ptype=1.

module pooler
  import cnn_pkg::*;
#(
  parameter int M          = POOL_IN_SIZE,  // input size (post-convolution)
  parameter int P          = POOL_SIZE,     // pooling window size
  parameter int N          = DATA_WIDTH,    // total bit width
  parameter int Q          = FRAC_BITS,     // fractional bits
  parameter int PTYPE      = POOL_TYPE,     // 0 = average, 1 = max
  parameter logic [N-1:0] P_SQR_INV = POOL_SQR_INV  // 1/p^2 in (N,Q) format
)(
  input  logic           clk,
  input  logic           ce,
  input  logic           master_rst,
  input  logic [N-1:0]  data_in,
  output logic [N-1:0]  data_out,
  output logic           valid_op,
  output logic           end_op
);

  // Number of pooling neighborhoods per row
  localparam int SR_DEPTH = M / P;

  logic           rst_m, load_sr, global_rst;
  logic [1:0]     sel;
  logic [N-1:0]   comp_op;
  logic [N-1:0]   sr_op;
  logic [N-1:0]   max_reg_op;
  logic [N-1:0]   div_op;
  logic           ovr;
  logic [N-1:0]   mux_out;

  // ── Control logic ──
  control_logic2 #(.M(M), .P(P)) u_ctrl (
    .clk        (clk),
    .master_rst (master_rst),
    .ce         (ce),
    .sel        (sel),
    .rst_m      (rst_m),
    .op_en      (valid_op),
    .load_sr    (load_sr),
    .global_rst (global_rst),
    .end_op     (end_op)
  );

  // ── Comparator: max (ptype=1) or sum (ptype=0) ──
  comparator2 #(.N(N), .PTYPE(PTYPE)) u_cmp (
    .ce      (ce),
    .ip1     (data_in),
    .ip2     (mux_out),
    .comp_op (comp_op)
  );

  // ── Register to hold current maximum/sum ──
  max_reg #(.N(N)) u_max_reg (
    .clk        (clk),
    .ce         (ce),
    .din        (comp_op),
    .rst_m      (rst_m),
    .master_rst (master_rst),
    .reg_op     (max_reg_op)
  );

  // ── Shift register for row buffering ──
  variable_shift_reg #(.WIDTH(N), .SIZE(SR_DEPTH)) u_sr (
    .d   (comp_op),
    .clk (clk),
    .ce  (load_sr),
    .rst (global_rst && master_rst),
    .out (sr_op)
  );

  // ── Input multiplexer ──
  input_mux #(.N(N)) u_mux (
    .ip1 (sr_op),
    .ip2 (max_reg_op),
    .sel (sel),
    .op  (mux_out)
  );

  // ── For average pooling: divide sum by p^2 ──
  qmult #(.N(N), .Q(Q)) u_div (
    .clk      (clk),
    .rst      (rst_m),
    .a        (max_reg_op),
    .b        (P_SQR_INV),
    .q_result (div_op),
    .overflow (ovr)
  );

  // ── Output selection ──
  assign data_out = PTYPE ? max_reg_op : div_op;

endmodule
