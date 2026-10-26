// tb_matmul_3x3.sv
// Testbench for normal_3x3 and systolic_3x3 matrix multipliers.
//
// Test scenarios:
//   1. Identity matrix  : A * I = A
//   2. Zero matrix      : A * 0 = 0
//   3. Ones matrix      : all-ones * all-ones (each element = N = 3)
//   4. Known values     : manually pre-computed result
//   5. Negative values  : signed arithmetic check
//   6. Max values       : saturate near int8 max (127)
//
// For each scenario the testbench:
//   (a) drives normal_3x3 (combinational) and checks C immediately
//   (b) drives systolic_3x3, waits for valid_out, then checks C
//   (c) cross-checks that both produce the same result
//
// Pass/fail summary printed at end.

`timescale 1ns/1ps

module tb_matmul_3x3;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int DATA_W     = 8;
    localparam int ACCUM_W    = 24;
    localparam int N          = 3;
    localparam int CLK_PERIOD = 10; // ns

    // -----------------------------------------------------------------------
    // Clock & reset
    // -----------------------------------------------------------------------
    logic clk   = 0;
    logic rst_n = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic signed [DATA_W-1:0]  A      [0:2][0:2];
    logic signed [DATA_W-1:0]  B      [0:2][0:2];

    logic signed [ACCUM_W-1:0] C_norm [0:2][0:2];  // normal_3x3 output
    logic signed [ACCUM_W-1:0] C_sys  [0:2][0:2];  // systolic_3x3 output
    logic                      valid_sys;
    logic                      start_sys = 0;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    normal_3x3 #(.DATA_W(DATA_W), .ACCUM_W(ACCUM_W)) u_normal (
        .A (A),
        .B (B),
        .C (C_norm)
    );

    systolic_3x3 #(.N(N), .DATA_W(DATA_W), .ACCUM_W(ACCUM_W)) u_systolic (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start_sys),
        .A         (A),
        .B         (B),
        .C         (C_sys),
        .valid_out (valid_sys)
    );

    // -----------------------------------------------------------------------
    // Test counters (module-scope so tasks can update them)
    // -----------------------------------------------------------------------
    int pass_count = 0;
    int fail_count = 0;

    // Expected result buffer
    logic signed [ACCUM_W-1:0] C_exp [0:2][0:2];

    // -----------------------------------------------------------------------
    // Tasks (all automatic so local variables are per-call)
    // -----------------------------------------------------------------------

    // Reference model: C = A*B computed in pure SV
    task automatic compute_expected(
        input  logic signed [DATA_W-1:0]  tA [0:2][0:2],
        input  logic signed [DATA_W-1:0]  tB [0:2][0:2],
        output logic signed [ACCUM_W-1:0] tC [0:2][0:2]
    );
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                tC[i][j] = 0;
                for (int k = 0; k < N; k++)
                    tC[i][j] += ACCUM_W'(tA[i][k]) * ACCUM_W'(tB[k][j]);
            end
    endtask

    // Compare got vs expected, report PASS/FAIL
    task automatic check_result(
        input string                       label,
        input string                       dut_name,
        input logic signed [ACCUM_W-1:0]  got [0:2][0:2],
        input logic signed [ACCUM_W-1:0]  exp [0:2][0:2]
    );
        bit ok = 1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (got[i][j] !== exp[i][j]) ok = 0;

        if (ok) begin
            $display("[PASS] %-30s | %s", label, dut_name);
            pass_count++;
        end else begin
            $display("[FAIL] %-30s | %s", label, dut_name);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (got[i][j] !== exp[i][j])
                        $display("       C[%0d][%0d]: got=%0d  exp=%0d",
                                 i, j, got[i][j], exp[i][j]);
            fail_count++;
        end
    endtask

    // Cross-check: normal_3x3 output == systolic_3x3 output
    task automatic cross_check(input string label);
        bit ok = 1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (C_norm[i][j] !== C_sys[i][j]) ok = 0;
        if (ok) begin
            $display("[PASS] %-30s | normal == systolic", label);
            pass_count++;
        end else begin
            $display("[FAIL] %-30s | normal != systolic", label);
            fail_count++;
        end
    endtask

    // Pulse start, wait for valid_out (with timeout)
    task automatic run_systolic(input string label);
        int timeout;
        timeout = 0;
        @(posedge clk); #1;
        start_sys = 1;
        @(posedge clk); #1;
        start_sys = 0;
        while (!valid_sys && timeout < 30) begin
            @(posedge clk); #1;
            timeout++;
        end
        if (timeout >= 30) begin
            $display("[FAIL] %-30s | systolic TIMEOUT", label);
            fail_count++;
        end
    endtask

    // Full scenario: load matrices, drive both DUTs, verify
    task automatic run_scenario(
        input string                      label,
        input logic signed [DATA_W-1:0]  tA [0:2][0:2],
        input logic signed [DATA_W-1:0]  tB [0:2][0:2]
    );
        logic signed [ACCUM_W-1:0] tC_exp [0:2][0:2];

        // Load inputs (visible to both DUTs)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                A[i][j] = tA[i][j];
                B[i][j] = tB[i][j];
            end

        // Compute expected via reference model
        compute_expected(tA, tB, tC_exp);

        // --- normal_3x3: combinational, check after propagation ---
        #2;
        check_result(label, "normal_3x3",   C_norm, tC_exp);

        // --- systolic_3x3: clocked ---
        run_systolic(label);
        check_result(label, "systolic_3x3", C_sys,  tC_exp);

        // --- cross-check ---
        cross_check(label);

        repeat(2) @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Reset helper
    // -----------------------------------------------------------------------
    task automatic do_reset();
        rst_n     = 0;
        start_sys = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin A[i][j] = 0; B[i][j] = 0; end
        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
    endtask

    // -----------------------------------------------------------------------
    // Scenario data (declared at module scope to avoid iverilog lifetime issues)
    // -----------------------------------------------------------------------
    logic signed [DATA_W-1:0] sA [0:2][0:2];
    logic signed [DATA_W-1:0] sB [0:2][0:2];

    // -----------------------------------------------------------------------
    // Main test sequence
    // -----------------------------------------------------------------------
    initial begin
        $display("=== 3x3 Matrix Multiplier Testbench ===");
        $display("");
        do_reset();

        // ------------------------------------------------------------------
        // Scenario 1: A * Identity = A
        //   A = [[1,2,3],[4,5,6],[7,8,9]]   B = I
        // ------------------------------------------------------------------
        sA = '{ '{8'sd1,8'sd2,8'sd3}, '{8'sd4,8'sd5,8'sd6}, '{8'sd7,8'sd8,8'sd9} };
        sB = '{ '{8'sd1,8'sd0,8'sd0}, '{8'sd0,8'sd1,8'sd0}, '{8'sd0,8'sd0,8'sd1} };
        run_scenario("Sc1: A*I=A", sA, sB);

        // ------------------------------------------------------------------
        // Scenario 2: A * Zero = 0
        // ------------------------------------------------------------------
        sA = '{ '{8'sd1,8'sd2,8'sd3}, '{8'sd4,8'sd5,8'sd6}, '{8'sd7,8'sd8,8'sd9} };
        sB = '{ '{8'sd0,8'sd0,8'sd0}, '{8'sd0,8'sd0,8'sd0}, '{8'sd0,8'sd0,8'sd0} };
        run_scenario("Sc2: A*0=0", sA, sB);

        // ------------------------------------------------------------------
        // Scenario 3: Ones * Ones — each element = 3
        // ------------------------------------------------------------------
        sA = '{ '{8'sd1,8'sd1,8'sd1}, '{8'sd1,8'sd1,8'sd1}, '{8'sd1,8'sd1,8'sd1} };
        sB = '{ '{8'sd1,8'sd1,8'sd1}, '{8'sd1,8'sd1,8'sd1}, '{8'sd1,8'sd1,8'sd1} };
        run_scenario("Sc3: ones*ones=3", sA, sB);

        // ------------------------------------------------------------------
        // Scenario 4: Known sparse values
        //   A = [[1,0,2],[0,3,0],[4,0,5]]
        //   B = [[1,1,0],[0,1,1],[1,0,1]]
        //   C = [[3,1,2],[0,3,3],[9,4,5]]
        // ------------------------------------------------------------------
        sA = '{ '{8'sd1,8'sd0,8'sd2}, '{8'sd0,8'sd3,8'sd0}, '{8'sd4,8'sd0,8'sd5} };
        sB = '{ '{8'sd1,8'sd1,8'sd0}, '{8'sd0,8'sd1,8'sd1}, '{8'sd1,8'sd0,8'sd1} };
        run_scenario("Sc4: known sparse", sA, sB);

        // ------------------------------------------------------------------
        // Scenario 5: Negative values (signed arithmetic)
        //   A = [[-1,2,-3],[4,-5,6],[-7,8,-9]]
        //   B = [[1,-2,3],[-4,5,-6],[7,-8,9]]
        // ------------------------------------------------------------------
        sA = '{ '{-8'sd1, 8'sd2,-8'sd3}, '{ 8'sd4,-8'sd5, 8'sd6}, '{-8'sd7, 8'sd8,-8'sd9} };
        sB = '{ '{ 8'sd1,-8'sd2, 8'sd3}, '{-8'sd4, 8'sd5,-8'sd6}, '{ 8'sd7,-8'sd8, 8'sd9} };
        run_scenario("Sc5: negatives", sA, sB);

        // ------------------------------------------------------------------
        // Scenario 6: Max values — A = B = all 127
        //   Each element = 3 * 127 * 127 = 48387
        // ------------------------------------------------------------------
        sA = '{ '{8'sd127,8'sd127,8'sd127}, '{8'sd127,8'sd127,8'sd127}, '{8'sd127,8'sd127,8'sd127} };
        sB = '{ '{8'sd127,8'sd127,8'sd127}, '{8'sd127,8'sd127,8'sd127}, '{8'sd127,8'sd127,8'sd127} };
        run_scenario("Sc6: max +127", sA, sB);

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        $display("");
        $display("========================================");
        $display("Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED - see above");
        $display("========================================");
        $finish;
    end

    // Watchdog
    initial begin
        #200000;
        $display("[ERROR] Global timeout");
        $finish;
    end

endmodule
