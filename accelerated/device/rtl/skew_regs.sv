// skew_regs.sv -- Skew registers
//
// - One set of skew registers for one side of input to PE array

module skew_regs #(
    parameter ARRAY_DIM = 4,
    parameter MUL_WIDTH = 32
) (
    input  logic clk,
    input  logic zero_data,
    input  logic read_operands,
    input  logic [MUL_WIDTH-1:0] in_values  [0:ARRAY_DIM-1],
    output logic [MUL_WIDTH-1:0] out_values [0:ARRAY_DIM-1]
);

    logic [MUL_WIDTH-1:0] inputs [0:ARRAY_DIM-1];

    // each individual bus is indexed by the register its data comes from
    logic [MUL_WIDTH-1:0] data_stream [1:ARRAY_DIM-1][0:ARRAY_DIM-2];

    ////////////////////////////////////////////////////////////////////////////////
    // REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    // input gate
    always_comb begin
        for (int i = 0; i < ARRAY_DIM; i++) begin
            inputs[i] = read_operands ? in_values[i] : '0;
        end
    end

    // no skew for first pipeline
    assign out_values[0] = inputs[0];
    generate
        // register 1,0 is the register closest to PE 0,0
        for (genvar i = 1; i < ARRAY_DIM; i++) begin  // width
            for (genvar j = 0; j < i; j++) begin  // depth; excludes over-declared registers
                flop_r #(.WIDTH(MUL_WIDTH), .RST(0)) reg_instance(.clk,
                                                                  .rst(zero_data),
                                                                  .d((j == (i - 1)) ? inputs[i] : data_stream[i][j+1]),
                                                                  .q(data_stream[i][j]));
            end

            assign out_values[i] = data_stream[i][0];
        end
    endgenerate

endmodule
