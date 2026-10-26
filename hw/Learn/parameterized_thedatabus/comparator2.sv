// Comparator / Adder for Pooler
// ptype=1 (max pooling):  outputs max of two signed fixed-point inputs
// ptype=0 (average pooling): outputs sum of two inputs
//
// NOTE: .md documentation uses a simplified unsigned comparison (ip1 > ip2).
//       This .v-based design handles signed 2's complement correctly by
//       checking sign bits and comparing magnitudes for negative numbers.

module comparator2
  import cnn_pkg::*;
#(
  parameter int N     = DATA_WIDTH,
  parameter int Q     = FRAC_BITS,
  parameter int PTYPE = POOL_TYPE  // 0 = average (sum), 1 = max pooling
)(
  input  logic           ce,
  input  logic [N-1:0]  ip1,
  input  logic [N-1:0]  ip2,
  output logic [N-1:0]  comp_op
);

  logic [N-1:0] ip1_2cmp, ip2_2cmp;
  logic [N-1:0] temp;

  // 2's complement for magnitude comparison of negative numbers
  assign ip1_2cmp = {~ip1[N-1], ~ip1[N-2:0] + 1'b1};
  assign ip2_2cmp = {~ip2[N-1], ~ip2[N-2:0] + 1'b1};

  assign comp_op = ce ? temp : '0;

  always_comb begin
    if (PTYPE == 0) begin
      // Average pooling mode: accumulate sum
      temp = ip1 + ip2;
    end
    else begin
      // Max pooling mode: full signed comparison
      if (!ip1[N-1] && !ip2[N-1]) begin
        // Both positive: larger magnitude wins
        temp = (ip1 > ip2) ? ip1 : ip2;
      end
      else if (ip1[N-1] && ip2[N-1]) begin
        // Both negative: higher magnitude (2cmp) means more negative => pick smaller 2cmp
        temp = (ip1_2cmp > ip2_2cmp) ? ip2 : ip1;
      end
      else if (ip1[N-1] && !ip2[N-1]) begin
        // ip1 negative, ip2 positive
        temp = ip2;
      end
      else begin
        // ip1 positive, ip2 negative
        temp = ip1;
      end
    end
  end

endmodule
