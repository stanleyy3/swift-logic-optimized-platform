// mac.sv -- Floating-point multiply-accumulate unit
//
// - Purely combinational; the accumulator register lives in the parent
// - Operands are IEEE-754 binary32, the accumulator is a fixed-point value
//   whose LSB has a weight of 2**ACC_LSB
// - The mantissa product is exact (24x24 -> 48 bits, two DSP48E2 slices); only
//   the alignment shift discards bits, and only below 2**ACC_LSB
// - Subnormal operands are flushed to zero, matching the Vivado FP Operator
// - Infinities and NaNs are not handled
//
// *Datatype described here may have changed

module mac #(
    parameter MUL_WIDTH = 32,
    parameter EXP_WIDTH = 8,
    parameter ACC_WIDTH = 64,
    parameter ACC_LSB   = -52
) (
    input  logic [MUL_WIDTH-1:0] a,
    input  logic [MUL_WIDTH-1:0] b,
    input  logic [ACC_WIDTH-1:0] acc_in,
    output logic [ACC_WIDTH-1:0] acc_out
);

    localparam FRAC_WIDTH = MUL_WIDTH - EXP_WIDTH - 1;   // 23
    localparam MANT_WIDTH = FRAC_WIDTH + 1;              // 24, including the hidden bit
    localparam PROD_WIDTH = 2 * MANT_WIDTH;              // 48, exact
    localparam BIAS       = (1 << (EXP_WIDTH - 1)) - 1;  // 127

    // the mantissa product's LSB has a weight of 2**(exp_a + exp_b - SHIFT_BIAS - ACC_LSB)
    localparam signed [15:0] SHIFT_BIAS = 2 * BIAS + 2 * FRAC_WIDTH + ACC_LSB;

    logic                        sign_a, sign_b;
    logic [EXP_WIDTH-1:0]        exp_a, exp_b;
    logic [FRAC_WIDTH-1:0]       frac_a, frac_b;
    logic [MANT_WIDTH-1:0]       mant_a, mant_b;
    logic [PROD_WIDTH-1:0]       mant_prod;
    logic                        prod_sign, is_zero;
    logic signed [15:0]          shift_amt;
    logic [ACC_WIDTH-1:0]        aligned;
    logic signed [ACC_WIDTH-1:0] addend;

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND UNPACKING
    ////////////////////////////////////////////////////////////////////////////////

    assign {sign_a, exp_a, frac_a} = a;
    assign {sign_b, exp_b, frac_b} = b;

    // a zero exponent covers both zero and subnormals, and both flush to zero
    assign is_zero = (exp_a == '0) || (exp_b == '0);

    assign mant_a = {1'b1, frac_a};
    assign mant_b = {1'b1, frac_b};

    ////////////////////////////////////////////////////////////////////////////////
    // EXACT MANTISSA PRODUCT
    ////////////////////////////////////////////////////////////////////////////////

    assign mant_prod = mant_a * mant_b;
    assign prod_sign = sign_a ^ sign_b;

    ////////////////////////////////////////////////////////////////////////////////
    // ALIGNMENT
    ////////////////////////////////////////////////////////////////////////////////

    // shift the product so that its LSB lines up with the accumulator's LSB; a
    // product lying entirely outside the accumulator's window contributes nothing
    assign shift_amt = signed'(16'(exp_a) + 16'(exp_b)) - SHIFT_BIAS;

    assign aligned = (is_zero || (shift_amt <= -PROD_WIDTH) || (shift_amt >= ACC_WIDTH)) ? '0
                                                                                         : ((shift_amt < 0) ? (ACC_WIDTH'(mant_prod) >> unsigned'(-shift_amt))
                                                                                                            : (ACC_WIDTH'(mant_prod) << unsigned'(shift_amt)));

    ////////////////////////////////////////////////////////////////////////////////
    // ACCUMULATE
    ////////////////////////////////////////////////////////////////////////////////

    assign addend  = prod_sign ? -signed'(aligned) : signed'(aligned);
    assign acc_out = signed'(acc_in) + addend;

endmodule
