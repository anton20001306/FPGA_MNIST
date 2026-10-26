// ============================================================================
// fc_engine.sv — Fully connected layer engine (sequential MAC)
//
// Computes: output[i] = requantize(bias[i] + sum_j(weight[i*N+j] * input[j]))
// Mode: REQUANT (int32→int8 for FC1) or RAW (int32 output for FC2)
//
// C++ equivalent (from fc_layer.cpp):
//   for i in M:
//     acc = bias[i]
//     for j in N:
//       acc += weight[i*N+j] * input[j]
//     output[i] = requantize(acc)   // or raw acc for last layer
//
// Total cycles: M * (N + overhead) per layer
// ============================================================================

module fc_engine #(
  parameter int N       = 200,   // input size
  parameter int M       = 64,    // output size
  parameter int ACT_W   = 8,
  parameter int ACC_W   = 32,
  parameter bit RAW_OUT = 0      // 1 = output raw int32 (FC2), 0 = requantize to int8
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,

  // Input activation read port
  output logic [$clog2(N)-1:0]   act_rd_addr,
  output logic                    act_rd_en,
  input  logic signed [ACT_W-1:0] act_rd_data,

  // Weight ROM read port
  output logic [$clog2(M*N)-1:0] wt_rd_addr,
  output logic                    wt_rd_en,
  input  logic signed [ACT_W-1:0] wt_rd_data,

  // Bias ROM read port
  output logic [$clog2(M)-1:0]   bias_rd_addr,
  output logic                    bias_rd_en,
  input  logic signed [ACC_W-1:0] bias_rd_data,

  // Output write port — int8 if requantized, int32 if RAW_OUT
  output logic [$clog2(M)-1:0]   out_wr_addr,
  output logic                    out_wr_en,
  output logic signed [ACT_W-1:0] out_wr_data_q,   // int8 (used when RAW_OUT=0)
  output logic signed [ACC_W-1:0] out_wr_data_raw,  // int32 (used when RAW_OUT=1)

  // Requantization parameters (ignored when RAW_OUT=1)
  input  logic signed [ACC_W-1:0] rq_multiplier,
  input  logic        [5:0]       rq_shift,
  input  logic                    apply_relu,

  output logic                    done
);

  typedef enum logic [2:0] {
    IDLE,
    LOAD_BIAS,
    WAIT_BIAS,
    MAC_PHASE,
    OUTPUT_PHASE,
    WAIT_RQ,
    WRITE_OUT,
    LAYER_DONE
  } state_t;

  state_t state;

  int i_cnt;  // output neuron [0, M)
  int j_cnt;  // input element [0, N)

  logic signed [ACC_W-1:0] acc;
  logic read_pending;

  // ─── Requantization unit (only used when RAW_OUT=0) ───
  logic               rq_valid_in, rq_valid_out;
  logic signed [7:0]  rq_result;

  generate
    if (!RAW_OUT) begin : gen_rq
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
    end else begin : gen_no_rq
      assign rq_valid_out = 1'b0;
      assign rq_result    = '0;
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= IDLE;
      i_cnt        <= 0;
      j_cnt        <= 0;
      acc          <= '0;
      done         <= 1'b0;
      act_rd_en    <= 1'b0;
      wt_rd_en     <= 1'b0;
      bias_rd_en   <= 1'b0;
      out_wr_en    <= 1'b0;
      rq_valid_in  <= 1'b0;
      read_pending <= 1'b0;
    end else begin
      act_rd_en   <= 1'b0;
      wt_rd_en    <= 1'b0;
      bias_rd_en  <= 1'b0;
      out_wr_en   <= 1'b0;
      rq_valid_in <= 1'b0;
      done        <= 1'b0;

      case (state)

        IDLE: begin
          if (start) begin
            i_cnt <= 0;
            state <= LOAD_BIAS;
          end
        end

        LOAD_BIAS: begin
          bias_rd_addr <= i_cnt[$clog2(M)-1:0];
          bias_rd_en   <= 1'b1;
          state        <= WAIT_BIAS;
        end

        WAIT_BIAS: begin
          acc   <= bias_rd_data;
          j_cnt <= 0;
          // Issue first reads
          act_rd_addr  <= 0;
          wt_rd_addr   <= i_cnt * N;
          act_rd_en    <= 1'b1;
          wt_rd_en     <= 1'b1;
          read_pending <= 1'b1;
          state        <= MAC_PHASE;
        end

        MAC_PHASE: begin
          if (read_pending) begin
            acc <= acc + (ACC_W'(act_rd_data) * ACC_W'(wt_rd_data));
            read_pending <= 1'b0;

            if (j_cnt == N - 1) begin
              // All MACs done for this neuron
              state <= OUTPUT_PHASE;
            end else begin
              j_cnt <= j_cnt + 1;
              act_rd_addr  <= j_cnt + 1;
              wt_rd_addr   <= i_cnt * N + j_cnt + 1;
              act_rd_en    <= 1'b1;
              wt_rd_en     <= 1'b1;
              read_pending <= 1'b1;
            end
          end
        end

        OUTPUT_PHASE: begin
          if (RAW_OUT) begin
            // Direct int32 output (FC2: no requantization)
            state <= WRITE_OUT;
          end else begin
            // Trigger requantization
            rq_valid_in <= 1'b1;
            state       <= WAIT_RQ;
          end
        end

        WAIT_RQ: begin
          if (rq_valid_out)
            state <= WRITE_OUT;
        end

        WRITE_OUT: begin
          out_wr_addr     <= i_cnt[$clog2(M)-1:0];
          out_wr_en       <= 1'b1;
          out_wr_data_q   <= rq_result;
          out_wr_data_raw <= acc;

          if (i_cnt == M - 1) begin
            state <= LAYER_DONE;
          end else begin
            i_cnt <= i_cnt + 1;
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
