// pe_array.sv -- PE array
//
// - Output-stationary dataflow
// - 'a_values' refers to the values streaming horizontally, 'b_values' refers
//   to the values streaming vertically
// - Fully square array of PEs

module pe_array #(
    parameter ARRAY_DIM = 4,
    parameter MUL_WIDTH = 16,
    parameter ACC_WIDTH = 48
) (
    input  logic                 clk,
    input  logic                 zero_data,
    input  logic [MUL_WIDTH-1:0] a_in_values [0:ARRAY_DIM-1],
    input  logic [MUL_WIDTH-1:0] b_in_values [0:ARRAY_DIM-1],
    output logic [ACC_WIDTH-1:0] acc_values  [0:ARRAY_DIM-1][0:ARRAY_DIM-1]
);

    // each individual bus is indexed by the PE its data comes from
    logic [MUL_WIDTH-1:0] a_stream [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] b_stream [0:ARRAY_DIM-1][0:ARRAY_DIM-1];

    ////////////////////////////////////////////////////////////////////////////////
    // ARRAY
    ////////////////////////////////////////////////////////////////////////////////

    generate
        // PE 0,0 is the top-left-most PE
        for (genvar i = 0; i < ARRAY_DIM; i++) begin
            for (genvar j = 0; j < ARRAY_DIM; j++) begin
                // output stream values of last row and column are left unconnected
                pe #(.MUL_WIDTH, .ACC_WIDTH) pe_instance(.clk,
                                                         .zero_data(zero_data),
                                                         .a_in((j == 0) ? a_in_values[i] : a_stream[i][j-1]),
                                                         .b_in((i == 0) ? b_in_values[j] : b_stream[i-1][j]),
                                                         .a_out(a_stream[i][j]),
                                                         .b_out(b_stream[i][j]),
                                                         .acc(acc_values[i][j]));
            end
        end
    endgenerate

endmodule
