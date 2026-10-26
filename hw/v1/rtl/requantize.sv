// ============================================================================
// requantize.sv — Int32 accumulator → Int8 requantization unit
//
// Implements the FPGA-equivalent of fixed_point.h requantize() / requantize_relu():
//   int64 v = acc * multiplier;
//   v = (v + (1 << (shift-1))) >> shift;   // round-half-up
//   clamp to [-128,127] or [0,127] if ReLU
//
// On FPGA: one 32x32→64 DSP multiply, add rounding constant, barrel shift, saturate.
//
// Pipeline: 2-cycle latency (multiply in cycle 1, shift+clamp in cycle 2)
// ============================================================================

module requantize (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               valid_in,
  input  logic signed [31:0] acc,         // int32 accumulator value
  input  logic signed [31:0] multiplier,  // requantization multiplier (Q0.31)
  input  logic        [5:0]  shift,       // right-shift amount (0–63)
  input  logic               apply_relu,  // 1 = clamp to [0,127], 0 = clamp to [-128,127]
  output logic               valid_out,
  output logic signed [7:0]  result       // int8 output
);

  // ─── Stage 1: Multiply ───
  logic signed [63:0] product_r;
  logic        [5:0]  shift_r;
  logic               relu_r;
  logic               valid_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_r <= '0;
      shift_r   <= '0;
      relu_r    <= '0;
      valid_r   <= '0;
    end else begin
      product_r <= 64'(acc) * 64'(multiplier);
      shift_r   <= shift;
      relu_r    <= apply_relu;
      valid_r   <= valid_in;
    end
  end

  // ─── Stage 2: Round + Shift + Clamp ───
  logic signed [63:0] rounded;
  logic signed [63:0] shifted;

  always_comb begin
    // Add rounding constant: 2^(shift-1)
    if (shift_r > 0)
      rounded = product_r + (64'sd1 <<< (shift_r - 1));
    else
      rounded = product_r;

    // Arithmetic right shift
    shifted = rounded >>> shift_r;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      result    <= '0;
      valid_out <= '0;
    end else begin
      valid_out <= valid_r;

      // Saturate to int8 range
      if (relu_r) begin
        // Fused ReLU: clamp to [0, 127]
        if (shifted > 64'sd127)
          result <= 8'sd127;
        else if (shifted < 64'sd0)
          result <= 8'sd0;
        else
          result <= shifted[7:0];
      end else begin
        // No ReLU: clamp to [-128, 127]
        if (shifted > 64'sd127)
          result <= 8'sd127;
        else if (shifted < -64'sd128)
          result <= -8'sd128;
        else
          result <= shifted[7:0];
      end
    end
  end

endmodule
