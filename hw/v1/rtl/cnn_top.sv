// ============================================================================
// cnn_top.sv — Top-level CNN accelerator for MNIST digit recognition
//
// Architecture: BRAM-based layer-by-layer compute
//   1. Host loads 28x28 int8 image into input BRAM
//   2. Controller sequences: CONV1 → POOL1 → CONV2 → POOL2 → FC1 → FC2 → ARGMAX
//   3. Each layer reads from one BRAM, computes, writes to the next
//   4. Argmax output is the predicted digit (0–9)
//
// Maps directly to the C++ forward() in sw/quantized/cnn.cpp.
//
// Memory layout: separate BRAMs for each activation buffer,
//                separate ROMs for each layer's weights and biases.
//
// Requantization parameters are loaded as configuration registers.
// ============================================================================

module cnn_top
  import cnn_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // ─── Host interface: image loading ───
  input  logic        img_wr_en,
  input  logic [9:0]  img_wr_addr,    // up to 784 pixels
  input  logic signed [7:0] img_wr_data,  // int8 pixel

  // ─── Control ───
  input  logic        start,          // pulse to begin inference
  output logic        done,           // inference complete
  output logic [3:0]  pred_class,     // predicted digit 0–9
  output logic        busy,

  // ─── Requantization config (loaded before start) ───
  input  logic signed [31:0] conv1_rq_mult,
  input  logic        [5:0]  conv1_rq_shift,
  input  logic signed [31:0] conv2_rq_mult,
  input  logic        [5:0]  conv2_rq_shift,
  input  logic signed [31:0] fc1_rq_mult,
  input  logic        [5:0]  fc1_rq_shift
);

  // ═══════════════════════════════════════════════════════════════════════════
  // Layer sequencing FSM
  // ═══════════════════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    S_IDLE,
    S_CONV1,
    S_POOL1,
    S_CONV2,
    S_POOL2,
    S_FC1,
    S_FC2,
    S_ARGMAX,
    S_DONE
  } top_state_t;

  top_state_t top_state;

  logic conv1_start, conv1_done;
  logic pool1_start, pool1_done;
  logic conv2_start, conv2_done;
  logic pool2_start, pool2_done;
  logic fc1_start,   fc1_done;
  logic fc2_start,   fc2_done;
  logic argmax_start, argmax_done;

  assign busy = (top_state != S_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      top_state    <= S_IDLE;
      conv1_start  <= 1'b0;
      pool1_start  <= 1'b0;
      conv2_start  <= 1'b0;
      pool2_start  <= 1'b0;
      fc1_start    <= 1'b0;
      fc2_start    <= 1'b0;
      argmax_start <= 1'b0;
      done         <= 1'b0;
    end else begin
      // Defaults: clear one-shot pulses
      conv1_start  <= 1'b0;
      pool1_start  <= 1'b0;
      conv2_start  <= 1'b0;
      pool2_start  <= 1'b0;
      fc1_start    <= 1'b0;
      fc2_start    <= 1'b0;
      argmax_start <= 1'b0;
      done         <= 1'b0;

      case (top_state)
        S_IDLE: begin
          if (start) begin
            conv1_start <= 1'b1;
            top_state   <= S_CONV1;
          end
        end
        S_CONV1:  if (conv1_done)  begin pool1_start  <= 1'b1; top_state <= S_POOL1;  end
        S_POOL1:  if (pool1_done)  begin conv2_start  <= 1'b1; top_state <= S_CONV2;  end
        S_CONV2:  if (conv2_done)  begin pool2_start  <= 1'b1; top_state <= S_POOL2;  end
        S_POOL2:  if (pool2_done)  begin fc1_start    <= 1'b1; top_state <= S_FC1;    end
        S_FC1:    if (fc1_done)    begin fc2_start    <= 1'b1; top_state <= S_FC2;    end
        S_FC2:    if (fc2_done)    begin argmax_start <= 1'b1; top_state <= S_ARGMAX; end
        S_ARGMAX: if (argmax_done) begin done         <= 1'b1; top_state <= S_DONE;   end
        S_DONE:   top_state <= S_IDLE;
        default:  top_state <= S_IDLE;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════════════════════
  // Activation BRAMs
  // ═══════════════════════════════════════════════════════════════════════════

  // Input image BRAM (int8, 784 deep)
  logic [9:0]           input_addr;
  logic                 input_we;
  logic signed [7:0]    input_din, input_dout;

  // Mux: host write vs conv1 read
  logic [9:0]           conv1_act_addr;
  logic                 conv1_act_en;

  assign input_addr = img_wr_en ? img_wr_addr : conv1_act_addr;
  assign input_we   = img_wr_en;
  assign input_din  = img_wr_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_INPUT)) u_input_bram (
    .clk(clk), .we(input_we), .addr(input_addr), .din(input_din), .dout(input_dout)
  );

  // Conv1 output / Pool1 input BRAM (int8, 2704 deep)
  logic [$clog2(BUF_CONV1)-1:0] buf_conv1_addr;
  logic                          buf_conv1_we;
  logic signed [7:0]             buf_conv1_din, buf_conv1_dout;

  logic [$clog2(BUF_CONV1)-1:0] conv1_out_addr;
  logic                          conv1_out_we;
  logic signed [7:0]             conv1_out_data;
  logic [$clog2(BUF_CONV1)-1:0] pool1_act_addr;
  logic                          pool1_act_en;

  assign buf_conv1_addr = conv1_out_we ? conv1_out_addr : pool1_act_addr;
  assign buf_conv1_we   = conv1_out_we;
  assign buf_conv1_din  = conv1_out_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_CONV1)) u_conv1_bram (
    .clk(clk), .we(buf_conv1_we), .addr(buf_conv1_addr),
    .din(buf_conv1_din), .dout(buf_conv1_dout)
  );

  // Pool1 output / Conv2 input BRAM (int8, 676 deep)
  logic [$clog2(BUF_POOL1)-1:0] buf_pool1_addr;
  logic                          buf_pool1_we;
  logic signed [7:0]             buf_pool1_din, buf_pool1_dout;

  logic [$clog2(BUF_POOL1)-1:0] pool1_out_addr;
  logic                          pool1_out_we;
  logic signed [7:0]             pool1_out_data;
  logic [$clog2(BUF_POOL1)-1:0] conv2_act_addr;
  logic                          conv2_act_en;

  assign buf_pool1_addr = pool1_out_we ? pool1_out_addr : conv2_act_addr;
  assign buf_pool1_we   = pool1_out_we;
  assign buf_pool1_din  = pool1_out_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_POOL1)) u_pool1_bram (
    .clk(clk), .we(buf_pool1_we), .addr(buf_pool1_addr),
    .din(buf_pool1_din), .dout(buf_pool1_dout)
  );

  // Conv2 output / Pool2 input BRAM (int8, 968 deep)
  logic [$clog2(BUF_CONV2)-1:0] buf_conv2_addr;
  logic                          buf_conv2_we;
  logic signed [7:0]             buf_conv2_din, buf_conv2_dout;

  logic [$clog2(BUF_CONV2)-1:0] conv2_out_addr;
  logic                          conv2_out_we;
  logic signed [7:0]             conv2_out_data;
  logic [$clog2(BUF_CONV2)-1:0] pool2_act_addr;
  logic                          pool2_act_en;

  assign buf_conv2_addr = conv2_out_we ? conv2_out_addr : pool2_act_addr;
  assign buf_conv2_we   = conv2_out_we;
  assign buf_conv2_din  = conv2_out_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_CONV2)) u_conv2_bram (
    .clk(clk), .we(buf_conv2_we), .addr(buf_conv2_addr),
    .din(buf_conv2_din), .dout(buf_conv2_dout)
  );

  // Pool2 output / FC1 input BRAM (int8, 200 deep)
  logic [$clog2(BUF_POOL2)-1:0] buf_pool2_addr;
  logic                          buf_pool2_we;
  logic signed [7:0]             buf_pool2_din, buf_pool2_dout;

  logic [$clog2(BUF_POOL2)-1:0] pool2_out_addr;
  logic                          pool2_out_we;
  logic signed [7:0]             pool2_out_data;
  logic [$clog2(BUF_POOL2)-1:0] fc1_act_addr;
  logic                          fc1_act_en;

  assign buf_pool2_addr = pool2_out_we ? pool2_out_addr : fc1_act_addr;
  assign buf_pool2_we   = pool2_out_we;
  assign buf_pool2_din  = pool2_out_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_POOL2)) u_pool2_bram (
    .clk(clk), .we(buf_pool2_we), .addr(buf_pool2_addr),
    .din(buf_pool2_din), .dout(buf_pool2_dout)
  );

  // FC1 output / FC2 input BRAM (int8, 64 deep)
  logic [$clog2(BUF_FC1)-1:0] buf_fc1_addr;
  logic                        buf_fc1_we;
  logic signed [7:0]           buf_fc1_din, buf_fc1_dout;

  logic [$clog2(BUF_FC1)-1:0] fc1_out_addr;
  logic                        fc1_out_we;
  logic signed [7:0]           fc1_out_data;
  logic [$clog2(BUF_FC1)-1:0] fc2_act_addr;
  logic                        fc2_act_en;

  assign buf_fc1_addr = fc1_out_we ? fc1_out_addr : fc2_act_addr;
  assign buf_fc1_we   = fc1_out_we;
  assign buf_fc1_din  = fc1_out_data;

  bram_sp #(.WIDTH(8), .DEPTH(BUF_FC1)) u_fc1_bram (
    .clk(clk), .we(buf_fc1_we), .addr(buf_fc1_addr),
    .din(buf_fc1_din), .dout(buf_fc1_dout)
  );

  // FC2 output BRAM (int32, 10 deep) — raw logits for argmax
  logic [$clog2(BUF_FC2)-1:0] buf_fc2_addr;
  logic                        buf_fc2_we;
  logic signed [31:0]          buf_fc2_din, buf_fc2_dout;

  logic [$clog2(BUF_FC2)-1:0] fc2_out_addr;
  logic                        fc2_out_we;
  logic signed [31:0]          fc2_out_data;
  logic [$clog2(BUF_FC2)-1:0] argmax_rd_addr;
  logic                        argmax_rd_en;

  assign buf_fc2_addr = fc2_out_we ? fc2_out_addr : argmax_rd_addr;
  assign buf_fc2_we   = fc2_out_we;
  assign buf_fc2_din  = fc2_out_data;

  bram_sp #(.WIDTH(32), .DEPTH(BUF_FC2)) u_fc2_bram (
    .clk(clk), .we(buf_fc2_we), .addr(buf_fc2_addr),
    .din(buf_fc2_din), .dout(buf_fc2_dout)
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // Weight ROMs (int8 weights, int32 biases — one per layer)
  // ═══════════════════════════════════════════════════════════════════════════

  // CONV1 weights
  logic [$clog2(CONV1_W_SIZE)-1:0] conv1_wt_addr;
  logic                             conv1_wt_en;
  logic signed [7:0]                conv1_wt_data;

  rom_sp #(.WIDTH(8), .DEPTH(CONV1_W_SIZE), .INIT_FILE("conv1_weights.hex")) u_conv1_wt (
    .clk(clk), .rd_en(conv1_wt_en), .addr(conv1_wt_addr), .dout(conv1_wt_data)
  );

  // CONV1 biases
  logic [$clog2(C1_OUT)-1:0] conv1_bias_addr;
  logic                       conv1_bias_en;
  logic signed [31:0]         conv1_bias_data;

  rom_sp #(.WIDTH(32), .DEPTH(C1_OUT), .INIT_FILE("conv1_bias.hex")) u_conv1_bias (
    .clk(clk), .rd_en(conv1_bias_en), .addr(conv1_bias_addr), .dout(conv1_bias_data)
  );

  // CONV2 weights
  logic [$clog2(CONV2_W_SIZE)-1:0] conv2_wt_addr;
  logic                             conv2_wt_en;
  logic signed [7:0]                conv2_wt_data;

  rom_sp #(.WIDTH(8), .DEPTH(CONV2_W_SIZE), .INIT_FILE("conv2_weights.hex")) u_conv2_wt (
    .clk(clk), .rd_en(conv2_wt_en), .addr(conv2_wt_addr), .dout(conv2_wt_data)
  );

  // CONV2 biases
  logic [$clog2(C2_OUT)-1:0] conv2_bias_addr;
  logic                       conv2_bias_en;
  logic signed [31:0]         conv2_bias_data;

  rom_sp #(.WIDTH(32), .DEPTH(C2_OUT), .INIT_FILE("conv2_bias.hex")) u_conv2_bias (
    .clk(clk), .rd_en(conv2_bias_en), .addr(conv2_bias_addr), .dout(conv2_bias_data)
  );

  // FC1 weights
  logic [$clog2(FC1_W_SIZE)-1:0] fc1_wt_addr;
  logic                           fc1_wt_en;
  logic signed [7:0]              fc1_wt_data;

  rom_sp #(.WIDTH(8), .DEPTH(FC1_W_SIZE), .INIT_FILE("fc1_weights.hex")) u_fc1_wt (
    .clk(clk), .rd_en(fc1_wt_en), .addr(fc1_wt_addr), .dout(fc1_wt_data)
  );

  // FC1 biases
  logic [$clog2(FC1_OUT)-1:0] fc1_bias_addr;
  logic                        fc1_bias_en;
  logic signed [31:0]          fc1_bias_data;

  rom_sp #(.WIDTH(32), .DEPTH(FC1_OUT), .INIT_FILE("fc1_bias.hex")) u_fc1_bias (
    .clk(clk), .rd_en(fc1_bias_en), .addr(fc1_bias_addr), .dout(fc1_bias_data)
  );

  // FC2 weights
  logic [$clog2(FC2_W_SIZE)-1:0] fc2_wt_addr;
  logic                           fc2_wt_en;
  logic signed [7:0]              fc2_wt_data;

  rom_sp #(.WIDTH(8), .DEPTH(FC2_W_SIZE), .INIT_FILE("fc2_weights.hex")) u_fc2_wt (
    .clk(clk), .rd_en(fc2_wt_en), .addr(fc2_wt_addr), .dout(fc2_wt_data)
  );

  // FC2 biases
  logic [$clog2(FC2_OUT)-1:0] fc2_bias_addr;
  logic                        fc2_bias_en;
  logic signed [31:0]          fc2_bias_data;

  rom_sp #(.WIDTH(32), .DEPTH(FC2_OUT), .INIT_FILE("fc2_bias.hex")) u_fc2_bias (
    .clk(clk), .rd_en(fc2_bias_en), .addr(fc2_bias_addr), .dout(fc2_bias_data)
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // Compute engines
  // ═══════════════════════════════════════════════════════════════════════════

  // ─── CONV1 ───
  conv_engine #(
    .C_IN(C1_IN), .H_IN(H1_IN), .W_IN(W1_IN), .C_OUT(C1_OUT), .K(K1)
  ) u_conv1 (
    .clk(clk), .rst_n(rst_n), .start(conv1_start),
    .act_rd_addr(conv1_act_addr), .act_rd_en(conv1_act_en), .act_rd_data(input_dout),
    .wt_rd_addr(conv1_wt_addr), .wt_rd_en(conv1_wt_en), .wt_rd_data(conv1_wt_data),
    .bias_rd_addr(conv1_bias_addr), .bias_rd_en(conv1_bias_en), .bias_rd_data(conv1_bias_data),
    .out_wr_addr(conv1_out_addr), .out_wr_en(conv1_out_we), .out_wr_data(conv1_out_data),
    .rq_multiplier(conv1_rq_mult), .rq_shift(conv1_rq_shift), .apply_relu(1'b1),
    .done(conv1_done)
  );

  // ─── POOL1 ───
  pool_engine #(
    .C(C1_OUT), .H_IN(H1_OUT), .W_IN(W1_OUT), .POOL(POOL1)
  ) u_pool1 (
    .clk(clk), .rst_n(rst_n), .start(pool1_start),
    .act_rd_addr(pool1_act_addr), .act_rd_en(pool1_act_en), .act_rd_data(buf_conv1_dout),
    .out_wr_addr(pool1_out_addr), .out_wr_en(pool1_out_we), .out_wr_data(pool1_out_data),
    .done(pool1_done)
  );

  // ─── CONV2 ───
  conv_engine #(
    .C_IN(C2_IN), .H_IN(H2_IN), .W_IN(W2_IN), .C_OUT(C2_OUT), .K(K2)
  ) u_conv2 (
    .clk(clk), .rst_n(rst_n), .start(conv2_start),
    .act_rd_addr(conv2_act_addr), .act_rd_en(conv2_act_en), .act_rd_data(buf_pool1_dout),
    .wt_rd_addr(conv2_wt_addr), .wt_rd_en(conv2_wt_en), .wt_rd_data(conv2_wt_data),
    .bias_rd_addr(conv2_bias_addr), .bias_rd_en(conv2_bias_en), .bias_rd_data(conv2_bias_data),
    .out_wr_addr(conv2_out_addr), .out_wr_en(conv2_out_we), .out_wr_data(conv2_out_data),
    .rq_multiplier(conv2_rq_mult), .rq_shift(conv2_rq_shift), .apply_relu(1'b1),
    .done(conv2_done)
  );

  // ─── POOL2 ───
  pool_engine #(
    .C(C2_OUT), .H_IN(H2_OUT), .W_IN(W2_OUT), .POOL(POOL2)
  ) u_pool2 (
    .clk(clk), .rst_n(rst_n), .start(pool2_start),
    .act_rd_addr(pool2_act_addr), .act_rd_en(pool2_act_en), .act_rd_data(buf_conv2_dout),
    .out_wr_addr(pool2_out_addr), .out_wr_en(pool2_out_we), .out_wr_data(pool2_out_data),
    .done(pool2_done)
  );

  // ─── FC1: 200→64 with requantization ───
  fc_engine #(
    .N(FC1_IN), .M(FC1_OUT), .RAW_OUT(0)
  ) u_fc1 (
    .clk(clk), .rst_n(rst_n), .start(fc1_start),
    .act_rd_addr(fc1_act_addr), .act_rd_en(fc1_act_en), .act_rd_data(buf_pool2_dout),
    .wt_rd_addr(fc1_wt_addr), .wt_rd_en(fc1_wt_en), .wt_rd_data(fc1_wt_data),
    .bias_rd_addr(fc1_bias_addr), .bias_rd_en(fc1_bias_en), .bias_rd_data(fc1_bias_data),
    .out_wr_addr(fc1_out_addr), .out_wr_en(fc1_out_we),
    .out_wr_data_q(fc1_out_data), .out_wr_data_raw(),
    .rq_multiplier(fc1_rq_mult), .rq_shift(fc1_rq_shift), .apply_relu(1'b1),
    .done(fc1_done)
  );

  // ─── FC2: 64→10 raw int32 output ───
  logic signed [7:0]  fc2_dummy_q;

  fc_engine #(
    .N(FC2_IN), .M(FC2_OUT), .RAW_OUT(1)
  ) u_fc2 (
    .clk(clk), .rst_n(rst_n), .start(fc2_start),
    .act_rd_addr(fc2_act_addr), .act_rd_en(fc2_act_en), .act_rd_data(buf_fc1_dout),
    .wt_rd_addr(fc2_wt_addr), .wt_rd_en(fc2_wt_en), .wt_rd_data(fc2_wt_data),
    .bias_rd_addr(fc2_bias_addr), .bias_rd_en(fc2_bias_en), .bias_rd_data(fc2_bias_data),
    .out_wr_addr(fc2_out_addr), .out_wr_en(fc2_out_we),
    .out_wr_data_q(fc2_dummy_q), .out_wr_data_raw(fc2_out_data),
    .rq_multiplier('0), .rq_shift('0), .apply_relu(1'b0),
    .done(fc2_done)
  );

  // ─── ARGMAX ───
  logic [3:0] argmax_pred;

  argmax #(.N(NUM_CLASSES)) u_argmax (
    .clk(clk), .rst_n(rst_n), .start(argmax_start),
    .rd_addr(argmax_rd_addr), .rd_en(argmax_rd_en), .rd_data(buf_fc2_dout),
    .pred_class(argmax_pred), .done(argmax_done)
  );

  assign pred_class = argmax_pred;

endmodule
