// operand_buffers.sv -- Operand buffers
//
// - Buffers two operand matrices

module operand_buffers #(
    parameter ARRAY_DIM = 4,
    parameter MUL_WIDTH = 16
) (
    input  logic                 clk,
    input  logic                 load,
    input  logic [MUL_WIDTH-1:0] in_values  [0:1][0:ARRAY_DIM-1][0:ARRAY_DIM-1],
    output logic [MUL_WIDTH-1:0] out_values [0:1][0:ARRAY_DIM-1][0:ARRAY_DIM-1]
);

    ////////////////////////////////////////////////////////////////////////////////
    // REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    generate
        for (genvar m = 0; m < 2; m++) begin  // matrix
            for (genvar i = 0; i < ARRAY_DIM; i++) begin // row
                for (genvar j = 0; j < ARRAY_DIM; j++) begin // column
                    flop_en #(.WIDTH(MUL_WIDTH)) element(.clk,
                                                         .en(load),
                                                         .d(in_values[m][i][j]),
                                                         .q(out_values[m][i][j]));
                end
            end
        end
    endgenerate

endmodule
