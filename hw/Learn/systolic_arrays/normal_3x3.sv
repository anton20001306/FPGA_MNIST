// normal_3x3.sv
// Standard combinational 3x3 matrix multiplier: C = A * B
// All 9 dot products computed in parallel using multipliers and adders.
// Each element: C[i][j] = sum_k( A[i][k] * B[k][j] ), k = 0..2
//
// Data format: signed 8-bit inputs, signed 24-bit outputs (enough to hold max value)
// Max value of one product: 127*127 = 16129 (fits in 16 bits)
// Max value of accumulated sum of 3 products: 3 * 16129 = 48387 (fits in 17 bits)
// Using 24-bit output for clarity and safety.

module normal_3x3 #(
    parameter int DATA_W  = 8,   // input bit-width (signed)
    parameter int ACCUM_W = 24   // output bit-width (signed)
)(
    // Matrix A: row-major, A[row][col]
    input  logic signed [DATA_W-1:0]  A [0:2][0:2],
    // Matrix B: row-major, B[row][col]
    input  logic signed [DATA_W-1:0]  B [0:2][0:2],
    // Output matrix C = A*B
    output logic signed [ACCUM_W-1:0] C [0:2][0:2]
);

    // Product width: 2*DATA_W bits for full-precision intermediate multiply
    localparam PROD_W = 2 * DATA_W;

    logic signed [PROD_W-1:0] prod [0:2][0:2][0:2]; // prod[i][j][k] = A[i][k]*B[k][j]

    // Compute all products and accumulate combinationally
    always_comb begin
        for (int i = 0; i < 3; i++) begin
            for (int j = 0; j < 3; j++) begin
                for (int k = 0; k < 3; k++) begin
                    prod[i][j][k] = A[i][k] * B[k][j];
                end
                C[i][j] = ACCUM_W'(prod[i][j][0]) +
                           ACCUM_W'(prod[i][j][1]) +
                           ACCUM_W'(prod[i][j][2]);
            end
        end
    end

endmodule
