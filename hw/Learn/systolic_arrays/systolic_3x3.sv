// systolic_3x3.sv
// Output-stationary 3x3 systolic array for matrix multiplication C = A * B
//
// Architecture: 3x3 grid of Processing Elements (PEs).
// Each PE accumulates partial products over N=3 clock cycles.
// A is fed row-wise from the left (with diagonal skew so columns arrive in sync).
// B is fed column-wise from the top (with diagonal skew).
// After 3+2=5 pipeline stages the full result is valid (skew + fill).
//
// Skewing: to align data so PE[i][j] receives A[i][k] and B[k][j] on the same cycle,
// A rows are delayed by i cycles and B columns are delayed by j cycles.
// Total latency = 2*(N-1) + N = N + 2*(N-1) fill cycles = 3+4 = 7 cycles for N=3.
// valid_out pulses high exactly once when all C[i][j] are stable.
//
// Data format: signed 8-bit inputs, signed 24-bit accumulators.

module systolic_3x3 #(
    parameter int N       = 3,
    parameter int DATA_W  = 8,
    parameter int ACCUM_W = 24
)(
    input  logic                      clk,
    input  logic                      rst_n,   // active-low sync reset
    input  logic                      start,   // pulse high for 1 cycle to begin feeding
    // Flat row-major input matrices (feed all at once; module manages skewing)
    input  logic signed [DATA_W-1:0]  A [0:2][0:2],
    input  logic signed [DATA_W-1:0]  B [0:2][0:2],
    output logic signed [ACCUM_W-1:0] C [0:2][0:2],
    output logic                      valid_out  // high when C is fully computed
);

    // Latched copies of A and B (captured on start)
    logic signed [DATA_W-1:0] A_reg [0:2][0:2];
    logic signed [DATA_W-1:0] B_reg [0:2][0:2];

    // -------------------------------------------------------------------------
    // PE: accumulates A*B partial products
    // -------------------------------------------------------------------------
    // PE[i][j] receives one A value and one B value per cycle for N cycles.
    logic signed [ACCUM_W-1:0] pe_accum [0:2][0:2];

    // -------------------------------------------------------------------------
    // Control: step counter
    // -------------------------------------------------------------------------
    // Total cycles needed: (N-1) skew + N accumulation = 2*(N-1)+N = 2+2+3 = ...
    // Actually: the last PE[2][2] first receives data on cycle start+2+2=start+4,
    // and needs N=3 cycles, so it finishes on cycle start+4+2=start+6.
    // We count 7 cycles total (0..6) and assert valid_out after cycle 6.
    localparam int TOTAL_CYCLES = 3*N - 3; // = 2*(N-1)+N-1 = 3N-3 for zero-indexed last

    logic [3:0] cnt;   // counts up to TOTAL_CYCLES
    logic        running;

    // capture start
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running   <= 0;
            cnt       <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= 0;
            if (start) begin
                running <= 1;
                cnt     <= 0;
                // Latch matrices
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < N; k++) begin
                        A_reg[i][k] <= A[i][k];
                        B_reg[k][i] <= B[k][i];
                    end
            end else if (running) begin
                if (cnt == TOTAL_CYCLES[3:0]) begin
                    running   <= 0;
                    valid_out <= 1;
                end else begin
                    cnt <= cnt + 1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Skew logic: present A[i][k] to the array on cycle (i+k) relative to start.
    // We do this by reading from A_reg column-by-column and gating with cnt.
    // A_feed[i] = A[i][col_index] where col_index = cnt - i (if in range, else 0)
    // B_feed[j] = B[row_index][j] where row_index = cnt - j (if in range, else 0)
    // -------------------------------------------------------------------------
    logic signed [DATA_W-1:0] A_feed [0:2]; // one value per row, fed left-to-right
    logic signed [DATA_W-1:0] B_feed [0:2]; // one value per col, fed top-to-bottom

    always_comb begin
        int col, row;
        for (int i = 0; i < N; i++) begin
            // col index for row i at current cycle cnt is (cnt - i)
            // valid when 0 <= cnt-i <= N-1
            col = int'(cnt) - i;
            if (running && col >= 0 && col < N)
                A_feed[i] = A_reg[i][col];
            else
                A_feed[i] = '0;
        end
        for (int j = 0; j < N; j++) begin
            row = int'(cnt) - j;
            if (running && row >= 0 && row < N)
                B_feed[j] = B_reg[row][j];
            else
                B_feed[j] = '0;
        end
    end

    // -------------------------------------------------------------------------
    // PE array: each PE[i][j] accumulates A_feed[i] * B_feed[j]
    // Reset accumulators on start.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    pe_accum[i][j] <= '0;
        end else if (start) begin
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    pe_accum[i][j] <= '0;
        end else if (running) begin
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    pe_accum[i][j] <= pe_accum[i][j] +
                                      ACCUM_W'(A_feed[i]) * ACCUM_W'(B_feed[j]);
        end
    end

    // Output: drive C from accumulators (valid only when valid_out)
    always_comb begin
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = pe_accum[i][j];
    end

endmodule
