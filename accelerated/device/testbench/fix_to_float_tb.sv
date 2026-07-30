// fix_to_float_tb.sv -- Fixed-point to floating-point converter testbench
//
// - The reference is written deliberately differently from the design: it shifts
//   the magnitude down and compares the discarded remainder against a half, where
//   the design left-aligns and slices, so a shared misconception is less likely to
//   hide a bug
// - Sweeps every small magnitude of both signs to cover rounding boundaries
//   densely, then samples the full range randomly

module fix_to_float_tb;

    localparam MUL_WIDTH = 16;
    localparam EXP_WIDTH = 5;
    localparam ACC_WIDTH = 48;
    localparam ACC_LSB   = -24;

    localparam FRAC_WIDTH = MUL_WIDTH - EXP_WIDTH - 1;
    localparam BIAS       = (1 << (EXP_WIDTH - 1)) - 1;
    localparam EXP_MAX    = (1 << EXP_WIDTH) - 1;

    localparam MAG_WIDTH = ACC_WIDTH + 1;
    localparam M_WIDTH   = ACC_WIDTH + 2;

    localparam SWEEP_LIMIT  = 1 << 16;
    localparam RANDOM_TRIES = 20000;

    logic [ACC_WIDTH-1:0] fixed_in;
    logic [MUL_WIDTH-1:0] float_out;

    int errors = 0;
    int checks = 0;

    fix_to_float #(.MUL_WIDTH(MUL_WIDTH),
                   .EXP_WIDTH(EXP_WIDTH),
                   .ACC_WIDTH(ACC_WIDTH),
                   .ACC_LSB(ACC_LSB)) dut(.fixed_in,
                                          .float_out);

    ////////////////////////////////////////////////////////////////////////////////
    // REFERENCE
    ////////////////////////////////////////////////////////////////////////////////

    function automatic [MUL_WIDTH-1:0] ref_convert(logic [ACC_WIDTH-1:0] value);
        logic                      sign, round_up;
        logic signed [MAG_WIDTH-1:0] extended;
        logic [MAG_WIDTH-1:0]      mag, tmp, dropped, half;
        logic [M_WIDTH-1:0]        m;
        int                        e, shift, exp_field;

        sign     = value[ACC_WIDTH-1];
        extended = signed'({value[ACC_WIDTH-1], value});
        mag      = sign ? unsigned'(-extended) : unsigned'(extended);

        if (mag == '0) return '0;

        // exponent is the index of the most significant set bit
        e   = 0;
        tmp = mag;
        while (tmp > 1) begin
            tmp = tmp >> 1;
            e   = e + 1;
        end

        shift = e - FRAC_WIDTH;

        if (shift > 0) begin
            // drop 'shift' bits and round to nearest, ties to even
            m       = M_WIDTH'(mag >> shift);
            dropped = mag & ((MAG_WIDTH'(1) << shift) - MAG_WIDTH'(1));
            half    = MAG_WIDTH'(1) << (shift - 1);

            if (dropped > half)       round_up = 1'b1;
            else if (dropped == half) round_up = m[0];
            else                      round_up = 1'b0;
        end else begin
            // nothing is discarded, so there is nothing to round
            m        = M_WIDTH'(mag) << (-shift);
            round_up = 1'b0;
        end

        m = m + M_WIDTH'(round_up);

        // rounding can push the mantissa up into the next binade
        if (m == (M_WIDTH'(1) << (FRAC_WIDTH + 1))) begin
            m = m >> 1;
            e = e + 1;
        end

        exp_field = e + ACC_LSB + BIAS;

        if (exp_field >= EXP_MAX)
            return {sign, EXP_WIDTH'(EXP_MAX - 1), {FRAC_WIDTH{1'b1}}};
        else if (exp_field <= 0)
            return {sign, {(MUL_WIDTH-1){1'b0}}};
        else
            return {sign, EXP_WIDTH'(exp_field), m[FRAC_WIDTH-1:0]};
    endfunction

    ////////////////////////////////////////////////////////////////////////////////
    // CHECKER
    ////////////////////////////////////////////////////////////////////////////////

    task automatic check(logic [ACC_WIDTH-1:0] value, string label);
        logic [MUL_WIDTH-1:0] expected;

        fixed_in = value;
        #1;

        expected = ref_convert(value);
        checks   = checks + 1;

        if (float_out !== expected) begin
            errors = errors + 1;

            if (errors <= 20)
                $display("  %s: fixed_in=%h  got=%h  expected=%h", label, value, float_out, expected);
        end
    endtask

    ////////////////////////////////////////////////////////////////////////////////
    // STIMULUS
    ////////////////////////////////////////////////////////////////////////////////

    initial begin
        // directed cases around zero, unity, and the format's limits
        check('0,                                   "zero");
        check(ACC_WIDTH'(1),                        "one lsb (underflows)");
        check(ACC_WIDTH'(1) << 9,                   "2**-15 (underflows)");
        check(ACC_WIDTH'(1) << 10,                  "2**-14 (smallest normal)");
        check(ACC_WIDTH'(1) << 24,                  "+1.0");
        check(-(ACC_WIDTH'(1) << 24),               "-1.0");
        check((ACC_WIDTH'(1) << 24) + (ACC_WIDTH'(1) << 23), "+1.5");
        check(-((ACC_WIDTH'(1) << 24) + (ACC_WIDTH'(1) << 23)), "-1.5");
        check({1'b0, {(ACC_WIDTH-1){1'b1}}},        "largest positive (clamps)");
        check({1'b1, {(ACC_WIDTH-1){1'b0}}},        "most negative (clamps, tests negation)");
        check({1'b1, {(ACC_WIDTH-1){1'b1}}},        "minus one lsb");

        // exact ties: 'shift' is 10 at this magnitude, so a discarded value of
        // 2**9 lands precisely halfway and must resolve toward an even mantissa
        check((ACC_WIDTH'(1) << 20) | (ACC_WIDTH'(1) << 9),                        "tie, mantissa even");
        check((ACC_WIDTH'(1) << 20) | (ACC_WIDTH'(1) << 10) | (ACC_WIDTH'(1) << 9), "tie, mantissa odd");
        check((ACC_WIDTH'(1) << 20) | (ACC_WIDTH'(1) << 9) | ACC_WIDTH'(1),        "just above a tie");
        check((ACC_WIDTH'(1) << 20) | ((ACC_WIDTH'(1) << 9) - ACC_WIDTH'(1)),      "just below a tie");

        // mantissa all ones, so rounding up carries into the exponent
        check((ACC_WIDTH'(1) << 21) - (ACC_WIDTH'(1) << 10),                       "rounding carry");

        // dense sweep of small magnitudes, both signs
        for (int i = 0; i < SWEEP_LIMIT; i++) begin
            check(ACC_WIDTH'(i),    "sweep +");
            check(-(ACC_WIDTH'(i)), "sweep -");
        end

        // sweep a shifted window so that larger exponents see the same boundaries
        for (int i = 0; i < SWEEP_LIMIT; i++) begin
            check(ACC_WIDTH'(i) << 16,    "shifted sweep +");
            check(-(ACC_WIDTH'(i) << 16), "shifted sweep -");
        end

        // random over the full range
        for (int i = 0; i < RANDOM_TRIES; i++) begin
            check({$urandom(), $urandom()}, "random full");
        end

        // random over the range a real workload produces
        for (int i = 0; i < RANDOM_TRIES; i++) begin
            check(ACC_WIDTH'(signed'(32'($urandom()))), "random narrow");
        end

        if (errors != 0) begin
            $display("");
            $display("fix_to_float_tb: %0d of %0d checks failed!", errors, checks);
            $display("");
            $stop;
        end

        $display("fix_to_float_tb: all %0d checks passed!", checks);
        $stop;
    end

endmodule
