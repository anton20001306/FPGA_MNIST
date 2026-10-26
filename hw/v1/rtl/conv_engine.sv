// ============================================================================
// conv_engine.sv — Parameterized convolution engine (layer-by-layer compute)
//
// Computes one full convolution layer: for each output position (n,x,y),
// accumulates C_in*K*K MAC operations, then requantizes to int8.
//
// C++ equivalent (from conv_layer.cpp):
//   for n in C_out:
//     for x in H_out:
//       for y in W_out:
//         acc = bias[n]
//         for c in C_in:
//           for i in K:
//             for j in K:
//               acc += input[c,x+i,y+j] * kernel[n,c,i,j]
//         output[n,x,y] = requantize(acc)
//
// Interface: reads from input BRAM + weight ROM, writes to output BRAM.
// One MAC per cycle. Total cycles per layer ≈ C_out * H_out * W_out * C_in * K * K
//
// Reference: Structural pattern from convolver.sv in parameterized_thedatabus/,
//            but using BRAM-based compute instead of streaming, and int8 arithmetic.
// ============================================================================

module conv_engine #(
  parameter int C_IN    = 1,
  parameter int H_IN    = 28,
  parameter int W_IN    = 28,
  parameter int C_OUT   = 4,
  parameter int K       = 3,
  parameter int ACT_W   = 8,
  parameter int ACC_W   = 32
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,       // pulse to begin computation

  // Input activation BRAM read port
  output logic [$clog2(C_IN*H_IN*W_IN)-1:0] act_rd_addr,
  output logic                                act_rd_en,
  input  logic signed [ACT_W-1:0]            act_rd_data,

  // Weight ROM read port
  output logic [$clog2(C_OUT*C_IN*K*K)-1:0]  wt_rd_addr,
  output logic                                wt_rd_en,
  input  logic signed [ACT_W-1:0]            wt_rd_data,

  // Bias ROM read port
  output logic [$clog2(C_OUT)-1:0]           bias_rd_addr,
  output logic                                bias_rd_en,
  input  logic signed [ACC_W-1:0]            bias_rd_data,

  // Output activation BRAM write port
  output logic [$clog2(C_OUT*(H_IN-K+1)*(W_IN-K+1))-1:0] out_wr_addr,
  output logic                                              out_wr_en,
  output logic signed [ACT_W-1:0]                          out_wr_data,

  // Requantization parameters (constant for the layer)
  input  logic signed [ACC_W-1:0]   rq_multiplier,
  input  logic        [5:0]         rq_shift,
  input  logic                      apply_relu,

  output logic                      done
);

  localparam int H_OUT     = H_IN - K + 1;
  localparam int W_OUT     = W_IN - K + 1;
  localparam int MAC_COUNT = C_IN * K * K;  // MACs per output pixel

  // ─── FSM states ───
  typedef enum logic [2:0] {
    IDLE,
    LOAD_BIAS,
    WAIT_BIAS,
    MAC_PHASE,
    REQUANT,
    WAIT_RQ,
    WRITE_OUT,
    LAYER_DONE
  } state_t;

  state_t state, state_next;

  // ─── Loop counters (matching C++ loop order) ───
  int n_cnt;    // output channel [0, C_OUT)
  int x_cnt;    // output row     [0, H_OUT)
  int y_cnt;    // output column  [0, W_OUT)
  int c_cnt;    // input channel  [0, C_IN)
  int i_cnt;    // kernel row     [0, K)
  int j_cnt;    // kernel col     [0, K)

  // ─── Accumulator ───
  logic signed [ACC_W-1:0] acc;
  logic signed [ACT_W-1:0] act_delayed, wt_delayed;
  logic mac_valid;

  // ─── Requantization unit ───
  logic               rq_valid_in, rq_valid_out;
  logic signed [7:0]  rq_result;

  requantize u_rq (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_in    (rq_valid_in),
    .acc         (acc),
    .multiplier  (rq_multiplier),
    .shift       (rq_shift),
    .apply_relu  (apply_relu),
    .valid_out   (rq_valid_out),
    .result      (rq_result)
  );

  // ─── BRAM read latency tracking (1-cycle read latency assumed) ───
  logic read_pending;

  // ─── FSM ───
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      n_cnt       <= 0;
      x_cnt       <= 0;
      y_cnt       <= 0;
      c_cnt       <= 0;
      i_cnt       <= 0;
      j_cnt       <= 0;
      acc         <= '0;
      done        <= 1'b0;
      act_rd_en   <= 1'b0;
      wt_rd_en    <= 1'b0;
      bias_rd_en  <= 1'b0;
      out_wr_en   <= 1'b0;
      rq_valid_in <= 1'b0;
      read_pending <= 1'b0;
    end else begin
      // Default de-assertions
      act_rd_en   <= 1'b0;
      wt_rd_en    <= 1'b0;
      bias_rd_en  <= 1'b0;
      out_wr_en   <= 1'b0;
      rq_valid_in <= 1'b0;
      done        <= 1'b0;

      case (state)

        IDLE: begin
          if (start) begin
            n_cnt <= 0; x_cnt <= 0; y_cnt <= 0;
            c_cnt <= 0; i_cnt <= 0; j_cnt <= 0;
            state <= LOAD_BIAS;
          end
        end

        // Read bias for current output channel
        LOAD_BIAS: begin
          bias_rd_addr <= n_cnt[$clog2(C_OUT)-1:0];
          bias_rd_en   <= 1'b1;
          state        <= WAIT_BIAS;
        end

        // Capture bias into accumulator (combinational read — data ready now)
        WAIT_BIAS: begin
          acc   <= bias_rd_data;
          c_cnt <= 0; i_cnt <= 0; j_cnt <= 0;
          // Issue first read: c=0, i=0, j=0 (use literals, not stale counters)
          act_rd_addr <= x_cnt * W_IN + y_cnt;                 // c=0: offset=0
          wt_rd_addr  <= n_cnt * C_IN * K * K;                 // c=0,i=0,j=0
          act_rd_en   <= 1'b1;
          wt_rd_en    <= 1'b1;
          read_pending <= 1'b1;
          state       <= MAC_PHASE;
        end

        // MAC loop: one multiply-accumulate per cycle (after read latency)
        MAC_PHASE: begin
          if (read_pending) begin
            // Data arrived from BRAM — accumulate
            acc <= acc + (ACC_W'(act_rd_data) * ACC_W'(wt_rd_data));
            read_pending <= 1'b0;

            // Advance inner loop counters: j → i → c
            if (j_cnt == K - 1) begin
              if (i_cnt == K - 1) begin
                if (c_cnt == C_IN - 1) begin
                  // All MACs done for this output pixel → requantize
                  state <= REQUANT;
                end else begin
                  // Next input channel
                  c_cnt <= c_cnt + 1;
                  i_cnt <= 0;
                  j_cnt <= 0;
                  // Issue next read
                  act_rd_addr <= (c_cnt + 1) * H_IN * W_IN + (x_cnt + 0) * W_IN + (y_cnt + 0);
                  wt_rd_addr  <= n_cnt * C_IN * K * K + (c_cnt + 1) * K * K + 0;
                  act_rd_en   <= 1'b1;
                  wt_rd_en    <= 1'b1;
                  read_pending <= 1'b1;
                end
              end else begin
                i_cnt <= i_cnt + 1;
                j_cnt <= 0;
                act_rd_addr <= c_cnt * H_IN * W_IN + (x_cnt + i_cnt + 1) * W_IN + (y_cnt + 0);
                wt_rd_addr  <= n_cnt * C_IN * K * K + c_cnt * K * K + (i_cnt + 1) * K + 0;
                act_rd_en   <= 1'b1;
                wt_rd_en    <= 1'b1;
                read_pending <= 1'b1;
              end
            end else begin
              j_cnt <= j_cnt + 1;
              act_rd_addr <= c_cnt * H_IN * W_IN + (x_cnt + i_cnt) * W_IN + (y_cnt + j_cnt + 1);
              wt_rd_addr  <= n_cnt * C_IN * K * K + c_cnt * K * K + i_cnt * K + (j_cnt + 1);
              act_rd_en   <= 1'b1;
              wt_rd_en    <= 1'b1;
              read_pending <= 1'b1;
            end
          end
        end

        // Trigger requantization
        REQUANT: begin
          rq_valid_in <= 1'b1;
          state       <= WAIT_RQ;
        end

        // Wait for requantize pipeline (2 cycles)
        WAIT_RQ: begin
          if (rq_valid_out) begin
            state <= WRITE_OUT;
          end
        end

        // Write requantized result to output BRAM
        WRITE_OUT: begin
          out_wr_addr <= n_cnt * H_OUT * W_OUT + x_cnt * W_OUT + y_cnt;
          out_wr_data <= rq_result;
          out_wr_en   <= 1'b1;

          // Advance outer loop counters: y → x → n
          if (y_cnt == W_OUT - 1) begin
            if (x_cnt == H_OUT - 1) begin
              if (n_cnt == C_OUT - 1) begin
                state <= LAYER_DONE;
              end else begin
                n_cnt <= n_cnt + 1;
                x_cnt <= 0;
                y_cnt <= 0;
                state <= LOAD_BIAS;
              end
            end else begin
              x_cnt <= x_cnt + 1;
              y_cnt <= 0;
              state <= LOAD_BIAS;  // reload bias (same channel, but re-init acc)
            end
          end else begin
            y_cnt <= y_cnt + 1;
            state <= LOAD_BIAS;
          end
        end

        LAYER_DONE: begin
          done  <= 1'b1;
          state <= IDLE;
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule
