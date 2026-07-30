// small_bufs.sv -- Small buffers (immediate operands)
//
// - Buffers two operand tile matrices and one output tile matrix to be fed
//   directly into skew registers
// - Lives in logic-based flip flops

module small_bufs #(
    parameter ARRAY_DIM,
    parameter MUL_WIDTH,
    parameter ACC_WIDTH
) (
    input  logic                         clk,
    input  logic                         opr_write_en,
    input  logic [$clog2(ARRAY_DIM)-1:0] opr_write_addrs     [0:1][0:1],
    input  logic [MUL_WIDTH-1:0]         opr_write_data      [0:1],
    input  logic                         out_write_en,
    input  logic [ACC_WIDTH-1:0]         out_write_data      [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    input  logic [$clog2(ARRAY_DIM)-1:0] opr_read_rowcol_idx,
    input  logic [$clog2(ARRAY_DIM)-1:0] out_read_addr       [0:1],
    output logic [MUL_WIDTH-1:0]         opr_read_data       [0:1][0:ARRAY_DIM-1],
    output logic [ACC_WIDTH-1:0]         out_read_data
);

    // lives in logic-based flip flops
    logic [MUL_WIDTH-1:0] A_buffer [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] B_buffer [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [ACC_WIDTH-1:0] C_buffer [0:ARRAY_DIM-1][0:ARRAY_DIM-1];

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND BUFFERS (ROW/COL)
    ////////////////////////////////////////////////////////////////////////////////

    // both operand buffers are read a whole row at a time; control stores the A
    // tile transposed (element (r,c) written to A_buffer[c][r]) so that row
    // 'opr_read_rowcol_idx' of A_buffer is column 'opr_read_rowcol_idx' of the A
    // tile, which is what the horizontal PE stream needs
    always_ff @(posedge clk) begin
        if (opr_write_en) begin
            A_buffer[opr_write_addrs[0][0]][opr_write_addrs[0][1]] <= opr_write_data[0];
            B_buffer[opr_write_addrs[1][0]][opr_write_addrs[1][1]] <= opr_write_data[1];
        end

        if (out_write_en)
            C_buffer <= out_write_data;

        opr_read_data[0] <= A_buffer[opr_read_rowcol_idx];
        opr_read_data[1] <= B_buffer[opr_read_rowcol_idx];

        out_read_data <= C_buffer[out_read_addr[0]][out_read_addr[1]];
    end

endmodule
