// pe.sv -- Processing element
//
// - Output-stationary dataflow
// - 'a' refers to the value streaming horizontally, 'b' refers to the value
//   streaming vertically
// - On a positive clock edge, the PE latches the new accumulated value and
//   makes `a` and `b` availables to downstream PEs
// - Default multiplication and accumulation widths designed to fit in one
//   DSP48E2 slice

module pe #(
    parameter MUL_WIDTH = 16,
    parameter ACC_WIDTH = 48
) (
    input  logic                 clk,
    input  logic                 zero_data,
    input  logic [MUL_WIDTH-1:0] a_in,
    input  logic [MUL_WIDTH-1:0] b_in,
    output logic [MUL_WIDTH-1:0] a_out,
    output logic [MUL_WIDTH-1:0] b_out,
    output logic [ACC_WIDTH-1:0] acc
);

    logic [ACC_WIDTH-1:0] prod;
    logic [ACC_WIDTH-1:0] new_acc;

    ////////////////////////////////////////////////////////////////////////////////
    // STREAMING DATA REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    flop_r #(.WIDTH(MUL_WIDTH), .RST(0)) a_reg(.clk,
                                               .rst(zero_data),
                                               .d(a_in),
                                               .q(a_out));

    flop_r #(.WIDTH(MUL_WIDTH), .RST(0)) b_reg(.clk,
                                               .rst(zero_data),
                                               .d(b_in),
                                               .q(b_out));

    ////////////////////////////////////////////////////////////////////////////////
    // ACCUMULATOR REGISTER
    ////////////////////////////////////////////////////////////////////////////////

    flop_r #(.WIDTH(ACC_WIDTH), .RST(0)) acc_reg(.clk,
                                                 .rst(zero_data),
                                                 .d(new_acc),
                                                 .q(acc));

    assign prod = $signed(a_in) * $signed(b_in);
    assign new_acc = acc + prod;

endmodule
