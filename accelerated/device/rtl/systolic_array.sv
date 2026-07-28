// systolic_array.sv -- Systolic array (top module)

module systolic_array (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 load,
    input  logic [MUL_WIDTH-1:0] new_A_values [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    input  logic [MUL_WIDTH-1:0] new_B_values [0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    input  logic                 start,
    output logic                 done,
    output logic [ACC_WIDTH-1:0] acc_values   [0:ARRAY_DIM-1][0:ARRAY_DIM-1]
);

    localparam ARRAY_DIM = 4;
    localparam MUL_WIDTH = 16;
    localparam ACC_WIDTH = 48;

    ////////////////////////////////////////////////////////////////////////////////

    // operand buffers
    logic                         read_operands;
    logic [$clog2(ARRAY_DIM)-1:0] read_operands_rowcol_num;
    logic                         zero_data;

    // skew registers
    logic [MUL_WIDTH-1:0] skew_reg_A_in_values [0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] skew_reg_B_in_values [0:ARRAY_DIM-1];

    // PE array
    logic [MUL_WIDTH-1:0] pe_array_A_in_values [0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] pe_array_B_in_values [0:ARRAY_DIM-1];

    ////////////////////////////////////////////////////////////////////////////////
    // CONTROL
    ////////////////////////////////////////////////////////////////////////////////

    control #(.ARRAY_DIM(ARRAY_DIM)) control_0(.clk,
                                               .rst,
                                               .start,
                                               .done,
                                               .read_operands,
                                               .read_operands_rowcol_num,
                                               .zero_data);

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND BUFFERS
    ////////////////////////////////////////////////////////////////////////////////

    operand_bufs #(.ARRAY_DIM(ARRAY_DIM), .MUL_WIDTH(MUL_WIDTH)) operand_bufs_0(.clk,
                                                                                .load,
                                                                                .new_A_values,
                                                                                .new_B_values,
                                                                                .read_operands_rowcol_num,
                                                                                .A_out_col(skew_reg_A_in_values),
                                                                                .B_out_row(skew_reg_B_in_values));

    ////////////////////////////////////////////////////////////////////////////////
    // SKEW REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    skew_regs #(.ARRAY_DIM(ARRAY_DIM), .MUL_WIDTH(MUL_WIDTH)) skew_regs_A(.clk,
                                                                          .zero_data,
                                                                          .read_operands,
                                                                          .in_values(skew_reg_A_in_values),
                                                                          .out_values(pe_array_A_in_values));

    skew_regs #(.ARRAY_DIM(ARRAY_DIM), .MUL_WIDTH(MUL_WIDTH)) skew_regs_B(.clk,
                                                                          .zero_data,
                                                                          .read_operands,
                                                                          .in_values(skew_reg_B_in_values),
                                                                          .out_values(pe_array_B_in_values));

    ////////////////////////////////////////////////////////////////////////////////
    // PE ARRAY
    ////////////////////////////////////////////////////////////////////////////////

    pe_array #(.ARRAY_DIM(ARRAY_DIM), .MUL_WIDTH(MUL_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_array_0(.clk,
                                                                                               .zero_data,
                                                                                               .a_in_values(pe_array_A_in_values),
                                                                                               .b_in_values(pe_array_B_in_values),
                                                                                               .acc_values);

endmodule
