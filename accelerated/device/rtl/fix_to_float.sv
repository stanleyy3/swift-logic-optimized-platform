// fix_to_float.sv -- Fixed-point to floating-point converter
//
// - Purely combinational; the inverse of the packing done by the MAC (see
//   mac.sv)
// - The input is a fixed-point value whose LSB has a weight of 2**ACC_LSB, the
//   output is a floating-point value in the same format as the MAC's operands
// - Rounds to nearest, ties to even
// - Values too large to represent clamp to the largest finite magnitude rather
//   than to infinity, since the MAC does not handle infinities
// - Values too small to represent flush to zero, matching the MAC's treatment
//   of subnormals

module fix_to_float #(
    parameter MUL_WIDTH,
    parameter EXP_WIDTH,
    parameter ACC_WIDTH,
    parameter ACC_LSB
) (
    input  logic [ACC_WIDTH-1:0] fixed_in,
    output logic [MUL_WIDTH-1:0] float_out
);

    localparam FRAC_WIDTH = MUL_WIDTH - EXP_WIDTH - 1;   // 10 (fp16)
    localparam BIAS       = (1 << (EXP_WIDTH - 1)) - 1;  // 15 (fp16)
    localparam EXP_MAX    = (1 << EXP_WIDTH) - 1;        // 31 (fp16)

    // one bit wider than the accumulator so that negating the most negative
    // input does not alias back onto itself
    localparam MAG_WIDTH = ACC_WIDTH + 1;                // 49 (fp16)
    localparam IDX_WIDTH = $clog2(MAG_WIDTH);            // 6  (fp16)

    // a magnitude whose MSB sits at bit index 'msb_idx' has a value of
    // 2**(msb_idx + ACC_LSB), so this is what turns that index into an exponent
    localparam signed [15:0] EXP_OFFSET = ACC_LSB + BIAS;  // -9 (fp16)

    logic                        sign, is_zero;
    logic signed [MAG_WIDTH-1:0] extended;
    logic        [MAG_WIDTH-1:0] mag, norm;
    logic        [IDX_WIDTH-1:0] msb_idx;
    logic        [FRAC_WIDTH-1:0] frac;
    logic                        round_bit, sticky, round_up;
    logic        [FRAC_WIDTH:0]  frac_sum;
    logic signed [15:0]          exp_biased, exp_rounded;
    logic                        overflow, underflow;

    ////////////////////////////////////////////////////////////////////////////////
    // MAGNITUDE
    ////////////////////////////////////////////////////////////////////////////////

    assign sign     = fixed_in[ACC_WIDTH-1];
    assign extended = MAG_WIDTH'(signed'(fixed_in));
    assign mag      = sign ? unsigned'(-extended) : unsigned'(extended);
    assign is_zero  = (mag == '0);

    ////////////////////////////////////////////////////////////////////////////////
    // NORMALIZATION
    ////////////////////////////////////////////////////////////////////////////////

    // index of the most significant set bit; the last iteration to see a set bit
    // wins, so this settles on the highest one
    always_comb begin
        msb_idx = '0;
        for (int i = 0; i < MAG_WIDTH; i++) begin
            if (mag[i]) msb_idx = IDX_WIDTH'(i);
        end
    end

    // left-align the magnitude so its leading one lands in the top bit, putting
    // the fraction and the rounding bits at fixed positions
    assign norm = mag << (MAG_WIDTH - 1 - msb_idx);

    assign frac      = norm[MAG_WIDTH-2 -: FRAC_WIDTH];
    assign round_bit = norm[MAG_WIDTH-2-FRAC_WIDTH];
    assign sticky    = |norm[MAG_WIDTH-3-FRAC_WIDTH:0];

    assign exp_biased = signed'(16'(msb_idx)) + EXP_OFFSET;

    ////////////////////////////////////////////////////////////////////////////////
    // ROUNDING
    ////////////////////////////////////////////////////////////////////////////////

    assign round_up = round_bit & (sticky | frac[0]);

    // a carry out means the fraction was all ones and has wrapped to all zeros,
    // which is the correct mantissa for the incremented exponent
    assign frac_sum    = {1'b0, frac} + round_up;
    assign exp_rounded = exp_biased + signed'(16'(frac_sum[FRAC_WIDTH]));

    // an exponent field of all ones is reserved, and a field of zero means
    // zero or subnormal
    assign overflow  = (exp_rounded >= signed'(16'(EXP_MAX)));
    assign underflow = (exp_rounded <= 16'sd0);

    ////////////////////////////////////////////////////////////////////////////////
    // PACKING
    ////////////////////////////////////////////////////////////////////////////////

    always_comb begin
        if (is_zero | underflow)
            float_out = {sign, {(MUL_WIDTH-1){1'b0}}};
        else if (overflow)
            float_out = {sign, EXP_WIDTH'(EXP_MAX - 1), {FRAC_WIDTH{1'b1}}};
        else
            float_out = {sign, EXP_WIDTH'(exp_rounded), frac_sum[FRAC_WIDTH-1:0]};
    end

endmodule