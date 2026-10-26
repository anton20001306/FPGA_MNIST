// Convolution Engine — Streaming MAC Array with Shift Registers
//
// Architecture: k*k MAC units arranged in k rows. Each row ends with a
// shift register of depth (n-k) for row buffering. Input activation is
// broadcast to all MACs simultaneously; weights are fixed per MAC position.
//
// Pipeline fill time: (k-1)*n + (k-1) clock cycles before first valid output.
//
// The .v file has `ifdef FIXED_POINT for a 1-cycle valid_conv delay.
// This SV version uses the fixed-point pipeline (1-cycle delay on valid_conv).
//
// Additional note: .v uses hardcoded 16 in weight1 port width and tmp array;
//                  this version parameterizes everything via N.

module convolver
  import cnn_pkg::*;
#(
  parameter int N_IMG = INPUT_SIZE,     // n: activation map size
  parameter int K     = KERNEL_SIZE,    // k: kernel size
  parameter int S     = STRIDE,         // s: stride value
  parameter int N     = DATA_WIDTH,     // total bit width
  parameter int Q     = FRAC_BITS       // fractional bits
)(
  input  logic               clk,
  input  logic               ce,
  input  logic               global_rst,
  input  logic [N-1:0]      activation,
  input  logic [K*K*N-1:0]  weight1,     // packed kernel weights
  output logic [N-1:0]      conv_op,
  output logic               valid_conv,
  output logic               end_conv
);

  // ── Unpack weights ──
  logic [N-1:0] weight [0:K*K-1];
  for (genvar l = 0; l < K*K; l++) begin : g_unpack
    assign weight[l] = weight1[N*l +: N];
  end

  // ── MAC chain with shift registers ──
  logic [N-1:0] tmp [0:K*K+1];
  assign tmp[0] = '0;

  for (genvar i = 0; i < K*K; i++) begin : MAC
    if ((i + 1) % K == 0) begin : g_row_end
      if (i == K*K - 1) begin : g_last_mac
        // Last MAC — output is conv_op
        (* use_dsp = "yes" *)
        mac_manual #(.N(N), .Q(Q)) u_mac (
          .clk  (clk),
          .ce   (ce),
          .sclr (global_rst),
          .a    (activation),
          .b    (weight[i]),
          .c    (tmp[i]),
          .p    (conv_op)
        );
      end
      else begin : g_mid_row_end
        // End of row (not last) — MAC followed by shift register
        logic [N-1:0] tmp2;
        (* use_dsp = "yes" *)
        mac_manual #(.N(N), .Q(Q)) u_mac (
          .clk  (clk),
          .ce   (ce),
          .sclr (global_rst),
          .a    (activation),
          .b    (weight[i]),
          .c    (tmp[i]),
          .p    (tmp2)
        );

        variable_shift_reg #(.WIDTH(N), .SIZE(N_IMG - K)) u_sr (
          .d   (tmp2),
          .clk (clk),
          .ce  (ce),
          .rst (global_rst),
          .out (tmp[i+1])
        );
      end
    end
    else begin : g_mid
      // Mid-row MAC — direct chain
      (* use_dsp = "yes" *)
      mac_manual #(.N(N), .Q(Q)) u_mac (
        .clk  (clk),
        .ce   (ce),
        .sclr (global_rst),
        .a    (activation),
        .b    (weight[i]),
        .c    (tmp[i]),
        .p    (tmp[i+1])
      );
    end
  end

  // ── Valid / End signal generation ──
  localparam int PIPE_FILL = (K - 1) * N_IMG + (K - 1);
  localparam int CONV_OUT  = N_IMG - K;  // n-k valid outputs per row before wrap

  int count;       // master clock counter
  int count2;      // valid output counter within a row
  int count3;      // invalid wrap counter
  int row_count;   // output row counter
  logic en1, en2, en3;
  logic conv_vld;
  logic conv_vld_d1;

  always_ff @(posedge clk) begin
    if (global_rst) begin
      count     <= 0;
      count2    <= 0;
      count3    <= 0;
      row_count <= 0;
      en1       <= 1'b0;
      en2       <= 1'b1;
      en3       <= 1'b0;
    end
    else if (ce) begin
      if (count == PIPE_FILL) begin
        en1   <= 1'b1;
        count <= count + 1;
      end
      else
        count <= count + 1;
    end

    if (en1 && en2) begin
      if (count2 == CONV_OUT) begin
        count2    <= 0;
        en2       <= 1'b0;
        row_count <= row_count + 1;
      end
      else
        count2 <= count2 + 1;
    end

    if (~en2) begin
      if (count3 == K - 2) begin
        count3 <= 0;
        en2    <= 1'b1;
      end
      else
        count3 <= count3 + 1;
    end

    // Stride-based valid enable
    if ((((count2 + 1) % S == 0) && (row_count % S == 0)) ||
        ((count3 == K - 2) && (row_count % S == 0)) ||
        (count == PIPE_FILL))
      en3 <= 1'b1;
    else
      en3 <= 1'b0;
  end

  assign conv_vld = en1 && en2 && en3;

  // Total input elements processed check
  localparam int TOTAL_INPUTS = N_IMG * N_IMG;

  assign end_conv = (count >= TOTAL_INPUTS + 2) ? 1'b1 : 1'b0;

  // 1-cycle delay on valid signal (fixed-point pipeline compensation)
  // Additional note: .md does not mention this delay; .v adds it for FIXED_POINT mode.
  always_ff @(posedge clk or posedge global_rst) begin
    if (global_rst)
      conv_vld_d1 <= 1'b0;
    else
      conv_vld_d1 <= conv_vld;
  end

  assign valid_conv = conv_vld_d1;

endmodule
