// ============================================================================
// pool_engine.sv — Max pooling engine (2x2 non-overlapping)
//
// Reads from input BRAM, finds max in each pool_size x pool_size window,
// writes result to output BRAM. Int8 signed comparison.
//
// C++ equivalent (from pool_layer.cpp):
//   for c in C:
//     for x in H_out:
//       for y in W_out:
//         max_val = -128
//         for i in pool_size:
//           for j in pool_size:
//             v = input[c, x*pool_size+i, y*pool_size+j]
//             if v > max_val: max_val = v
//         output[c,x,y] = max_val
//
// Reference: comparator2.sv from parameterized_thedatabus/ inspired the signed
//            comparison. This engine uses BRAM reads instead of streaming.
// ============================================================================

module pool_engine #(
  parameter int C       = 4,
  parameter int H_IN    = 26,
  parameter int W_IN    = 26,
  parameter int POOL    = 2,
  parameter int ACT_W   = 8
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,

  // Input BRAM read port
  output logic [$clog2(C*H_IN*W_IN)-1:0] act_rd_addr,
  output logic                             act_rd_en,
  input  logic signed [ACT_W-1:0]         act_rd_data,

  // Output BRAM write port
  output logic [$clog2(C*(H_IN/POOL)*(W_IN/POOL))-1:0] out_wr_addr,
  output logic                                           out_wr_en,
  output logic signed [ACT_W-1:0]                       out_wr_data,

  output logic                             done
);

  localparam int H_OUT = H_IN / POOL;
  localparam int W_OUT = W_IN / POOL;

  typedef enum logic [2:0] {
    IDLE,
    INIT_MAX,
    READ_ELEM,
    WAIT_READ,
    COMPARE,
    WRITE_OUT,
    LAYER_DONE
  } state_t;

  state_t state;

  int c_cnt, x_cnt, y_cnt;  // output position
  int i_cnt, j_cnt;          // pool window position
  logic signed [ACT_W-1:0] max_val;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= IDLE;
      c_cnt      <= 0;
      x_cnt      <= 0;
      y_cnt      <= 0;
      i_cnt      <= 0;
      j_cnt      <= 0;
      max_val    <= '0;
      act_rd_en  <= 1'b0;
      out_wr_en  <= 1'b0;
      done       <= 1'b0;
    end else begin
      act_rd_en <= 1'b0;
      out_wr_en <= 1'b0;
      done      <= 1'b0;

      case (state)

        IDLE: begin
          if (start) begin
            c_cnt <= 0; x_cnt <= 0; y_cnt <= 0;
            state <= INIT_MAX;
          end
        end

        // Initialize max to minimum int8
        INIT_MAX: begin
          max_val <= -8'sd128;
          i_cnt   <= 0;
          j_cnt   <= 0;
          state   <= READ_ELEM;
        end

        // Read one element from pool window
        READ_ELEM: begin
          act_rd_addr <= c_cnt * H_IN * W_IN
                       + (x_cnt * POOL + i_cnt) * W_IN
                       + (y_cnt * POOL + j_cnt);
          act_rd_en   <= 1'b1;
          state       <= WAIT_READ;
        end

        // Wait for BRAM read latency
        WAIT_READ: begin
          state <= COMPARE;
        end

        // Compare and update max
        COMPARE: begin
          if (act_rd_data > max_val)
            max_val <= act_rd_data;

          // Advance pool window: j → i
          if (j_cnt == POOL - 1) begin
            if (i_cnt == POOL - 1) begin
              // Pool window complete → write output
              state <= WRITE_OUT;
            end else begin
              i_cnt <= i_cnt + 1;
              j_cnt <= 0;
              state <= READ_ELEM;
            end
          end else begin
            j_cnt <= j_cnt + 1;
            state <= READ_ELEM;
          end
        end

        // Write max value to output
        WRITE_OUT: begin
          // Need to use the just-updated max_val (from COMPARE)
          // But if the last compare updated it, it's already in max_val
          out_wr_addr <= c_cnt * H_OUT * W_OUT + x_cnt * W_OUT + y_cnt;
          out_wr_data <= max_val;
          out_wr_en   <= 1'b1;

          // Advance output counters: y → x → c
          if (y_cnt == W_OUT - 1) begin
            if (x_cnt == H_OUT - 1) begin
              if (c_cnt == C - 1) begin
                state <= LAYER_DONE;
              end else begin
                c_cnt <= c_cnt + 1;
                x_cnt <= 0;
                y_cnt <= 0;
                state <= INIT_MAX;
              end
            end else begin
              x_cnt <= x_cnt + 1;
              y_cnt <= 0;
              state <= INIT_MAX;
            end
          end else begin
            y_cnt <= y_cnt + 1;
            state <= INIT_MAX;
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
