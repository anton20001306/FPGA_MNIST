// Pooler Control Logic
// Generates control signals for the pooling unit as the pooling window moves
// across the input feature map.
//
// 5 distinct cases:
// Case 1 (normal):                     store max in register
// Case 2 (end of neighborhood):        store max to shift register
// Case 3 (end of row):                 store to SR, then load max_reg from SR
// Case 4 (end of neighborhood, last row): output valid, store max
// Case 5 (end of last neighborhood):   output valid, reset module
//
// Signal table:
// Case               1     2     3     4     5
// load_sr           low   high  high  high  low
// sel               low   low   high  high  low
// rst_m             low   high  low   low   low
// op_en (valid_op)  low   low   low   high  high
// global_rst        low   low   low   low   high

module control_logic2
  import cnn_pkg::*;
#(
  parameter int M = POOL_IN_SIZE,  // input matrix size to pooler (m x m)
  parameter int P = POOL_SIZE      // pooling window size (p x p)
)(
  input  logic       clk,
  input  logic       master_rst,
  input  logic       ce,
  output logic [1:0] sel,
  output logic       rst_m,
  output logic       op_en,       // referred to as valid_op in .md
  output logic       load_sr,
  output logic       global_rst,
  output logic       end_op
);

  // Number of neighborhoods per row/column
  localparam int NUM_NEIGHBORHOODS = M / P;

  int row_count;
  int col_count;
  int count;           // neighborhood counter within a row
  int nbgh_row_count;  // completed neighborhood-row counter

  // ── Control signal generation ──
  always_ff @(posedge clk) begin
    if (master_rst) begin
      sel        <= 2'b00;
      load_sr    <= 1'b0;
      rst_m      <= 1'b0;
      op_en      <= 1'b0;
      global_rst <= 1'b0;
      end_op     <= 1'b0;
    end
    else begin
      // op_en generation
      if (((col_count + 1) % P != 0) && (row_count == P - 1) &&
          (col_count == P * count + (P - 2)) && ce)
        op_en <= 1'b1;
      else
        op_en <= 1'b0;

      if (ce) begin
        // end_op generation
        if (nbgh_row_count == NUM_NEIGHBORHOODS)
          end_op <= 1'b1;
        else
          end_op <= 1'b0;

        // global_rst generation
        if (((col_count + 1) % P != 0) && (col_count == M - 2) &&
            (row_count == P - 1))
          global_rst <= 1'b1;
        else
          global_rst <= 1'b0;

        // rst_m generation
        if ((((col_count + 1) % P == 0) && (count != NUM_NEIGHBORHOODS - 1) &&
             (row_count != P - 1)) ||
            ((col_count == M - 1) && (row_count == P - 1)))
          rst_m <= 1'b1;
        else
          rst_m <= 1'b0;

        // sel generation
        if (((col_count + 1) % P != 0) && (col_count == M - 2) &&
            (row_count == P - 1))
          sel <= 2'b10;
        else begin
          if ((((col_count) % P == 0) && (count == NUM_NEIGHBORHOODS - 1) &&
               (row_count != P - 1)) ||
              (((col_count) % P == 0) && (count != NUM_NEIGHBORHOODS - 1) &&
               (row_count == P - 1)))
            sel <= 2'b01;
          else
            sel <= 2'b00;
        end

        // load_sr generation
        if (((col_count + 1) % P == 0) && (count == NUM_NEIGHBORHOODS - 1) ||
            ((col_count + 1) % P == 0) && (count != NUM_NEIGHBORHOODS - 1))
          load_sr <= 1'b1;
        else
          load_sr <= 1'b0;
      end
    end
  end

  // ── Counter logic ──
  always_ff @(posedge clk) begin
    if (master_rst) begin
      row_count      <= 0;
      col_count      <= 32'hFFFF_FFFF;
      count          <= 32'hFFFF_FFFF;
      nbgh_row_count <= 0;
    end
    else if (ce) begin
      if (global_rst) begin
        row_count      <= 0;
        col_count      <= 0;
        count          <= 0;
        nbgh_row_count <= nbgh_row_count + 1;
      end
      else begin
        if (((col_count + 1) % P == 0) && (count == NUM_NEIGHBORHOODS - 1) &&
            (row_count != P - 1)) begin
          col_count <= 0;
          row_count <= row_count + 1;
          count     <= 0;
        end
        else begin
          col_count <= col_count + 1;
          if (((col_count + 1) % P == 0) && (count != NUM_NEIGHBORHOODS - 1))
            count <= count + 1;
        end
      end
    end
  end

endmodule
