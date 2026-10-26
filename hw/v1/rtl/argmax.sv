// ============================================================================
// argmax.sv — Find index of maximum value among N int32 logits
//
// Reads N int32 values from a BRAM port, outputs the index of the largest.
// Used after FC2 to determine predicted digit (0–9).
//
// C++ equivalent (from activations.h):
//   int best = 0;
//   for (int i = 1; i < size; i++)
//     if (logits[i] > logits[best]) best = i;
//   return best;
//
// Reference: softmax_argmax.md describes a combinational comparator chain.
//            This version reads sequentially from BRAM (N cycles).
// ============================================================================

module argmax #(
  parameter int N     = 10,
  parameter int ACC_W = 32
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,

  // Logits BRAM read port
  output logic [$clog2(N)-1:0]   rd_addr,
  output logic                    rd_en,
  input  logic signed [ACC_W-1:0] rd_data,

  output logic [$clog2(N)-1:0]   pred_class,  // predicted digit
  output logic                    done
);

  typedef enum logic [1:0] {
    IDLE,
    READ,
    WAIT_CMP,
    DONE_ST
  } state_t;

  state_t state;

  int idx;
  logic signed [ACC_W-1:0] best_val;
  logic [$clog2(N)-1:0]    best_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= IDLE;
      idx        <= 0;
      best_val   <= '0;
      best_idx   <= '0;
      rd_en      <= 1'b0;
      done       <= 1'b0;
      pred_class <= '0;
    end else begin
      rd_en <= 1'b0;
      done  <= 1'b0;

      case (state)

        IDLE: begin
          if (start) begin
            idx      <= 0;
            best_val <= {1'b1, {(ACC_W-1){1'b0}}};  // minimum int32
            best_idx <= '0;
            rd_addr  <= '0;
            rd_en    <= 1'b1;
            state    <= WAIT_CMP;
          end
        end

        // Wait for read data, then compare
        WAIT_CMP: begin
          state <= READ;
        end

        READ: begin
          // Compare current value with best
          if (rd_data > best_val) begin
            best_val <= rd_data;
            best_idx <= idx[$clog2(N)-1:0];
          end

          if (idx == N - 1) begin
            // All values compared
            pred_class <= (rd_data > best_val) ? idx[$clog2(N)-1:0] : best_idx;
            state      <= DONE_ST;
          end else begin
            idx     <= idx + 1;
            rd_addr <= idx[$clog2(N)-1:0] + 1'b1;
            rd_en   <= 1'b1;
            state   <= WAIT_CMP;
          end
        end

        DONE_ST: begin
          done  <= 1'b1;
          state <= IDLE;
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule
