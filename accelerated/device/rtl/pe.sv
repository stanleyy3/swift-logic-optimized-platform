// pe.sv -- Processing element
//
// - Output-stationary dataflow
// - 'a' refers to the value streaming horizontally, 'b' refers to the value
//   streaming vertically
// - On a positive clock edge, the PE latches the new accumulated value and
//   makes `a` and `b` availables to downstream PEs
// - Operands are IEEE-754 binary32, the accumulated value is fixed-point with
//   an LSB of weight 2**ACC_LSB (see mac.sv)
//
// *Datatype described here may have changed

module pe #(
    parameter MUL_WIDTH = 32,
    parameter EXP_WIDTH = 8,
    parameter ACC_WIDTH = 64,
    parameter ACC_LSB   = -52
) (
    input  logic                 clk,
    input  logic                 zero_data,
    input  logic [MUL_WIDTH-1:0] a_in,
    input  logic [MUL_WIDTH-1:0] b_in,
    output logic [MUL_WIDTH-1:0] a_out,
    output logic [MUL_WIDTH-1:0] b_out,
    output logic [ACC_WIDTH-1:0] acc
);

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

    ////////////////////////////////////////////////////////////////////////////////
    // MULTIPLY-ACCUMULATE
    ////////////////////////////////////////////////////////////////////////////////

    mac #(.MUL_WIDTH(MUL_WIDTH),
          .EXP_WIDTH(EXP_WIDTH),
          .ACC_WIDTH(ACC_WIDTH),
          .ACC_LSB(ACC_LSB)) mac_unit(.a(a_in),
                                      .b(b_in),
                                      .acc_in(acc),
                                      .acc_out(new_acc));

endmodule
