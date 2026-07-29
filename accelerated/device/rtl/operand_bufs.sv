// operand_bufs.sv -- Operand buffers
//
// - Buffers two operand matrices

module operand_bufs #(
    parameter ARRAY_DIM = 4,
    parameter MUL_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         load,
    input  logic [MUL_WIDTH-1:0]         new_A_values            [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    input  logic [MUL_WIDTH-1:0]         new_B_values            [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    input  logic [$clog2(ARRAY_DIM)-1:0] read_operands_rowcol_num,
    output logic [MUL_WIDTH-1:0]         A_out_col               [0:ARRAY_DIM-1],
    output logic [MUL_WIDTH-1:0]         B_out_row               [0:ARRAY_DIM-1]
);

    logic [MUL_WIDTH-1:0] A_buffer [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] B_buffer [0:ARRAY_DIM-1][0:ARRAY_DIM-1];

    ////////////////////////////////////////////////////////////////////////////////
    // REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    // A buffer
    generate
        for (genvar i = 0; i < ARRAY_DIM; i++) begin // row
            for (genvar j = 0; j < ARRAY_DIM; j++) begin // column
                flop_en #(.WIDTH(MUL_WIDTH)) element(.clk,
                                                     .en(load),
                                                     .d(new_A_values[i][j]),
                                                     .q(A_buffer[i][j]));
            end
        end
    endgenerate

    // B buffer
    generate
        for (genvar i = 0; i < ARRAY_DIM; i++) begin // row
            for (genvar j = 0; j < ARRAY_DIM; j++) begin // column
                flop_en #(.WIDTH(MUL_WIDTH)) element(.clk,
                                                     .en(load),
                                                     .d(new_B_values[i][j]),
                                                     .q(B_buffer[i][j]));
            end
        end
    endgenerate

    ////////////////////////////////////////////////////////////////////////////////
    // OUTPUT ROW/COL
    ////////////////////////////////////////////////////////////////////////////////

    // select column for A, select row for B

    always_comb begin
        // select column of A
        for (int i = 0; i < ARRAY_DIM; i++) begin
            A_out_col[i] = A_buffer[i][read_operands_rowcol_num];
        end

        // select row of B
        B_out_row = B_buffer[read_operands_rowcol_num];
    end

endmodule
