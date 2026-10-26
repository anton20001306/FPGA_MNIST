// ============================================================================
// mac_int8.sv — Int8 Multiply-Accumulate Unit
//
// Computes: acc_out = acc_in + (a * b)
//   - a, b: signed int8 (activation and weight)
//   - acc_in, acc_out: signed int32 (accumulator)
//   - Single-cycle combinational (DSP-inferred on FPGA)
//
// C++ equivalent (from conv_layer.cpp):
//   acc += (int32_t)input[idx] * (int32_t)kernel[idx];
//
// Reference: Adapted from mac_manual.sv in parameterized_thedatabus/,
//            but using int8 symmetric quantization instead of (N,Q) fixed-point.
// ============================================================================

module mac_int8 (
  input  logic signed [7:0]  a,        // activation (int8)
  input  logic signed [7:0]  b,        // weight     (int8)
  input  logic signed [31:0] acc_in,   // accumulator input
  output logic signed [31:0] acc_out   // accumulator output
);

  // int8 * int8 produces at most 16-bit signed result
  // Adding to int32 accumulator — no overflow for typical conv depths
  // Max accumulation: K*K*C_in * 127*127 = 9*4*16129 = 580,644 (fits int32)
  assign acc_out = acc_in + (32'(a) * 32'(b));

endmodule
