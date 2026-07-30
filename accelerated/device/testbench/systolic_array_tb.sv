// systolic_array_tb.sv -- Systolic array testbench
//
// - Drives the design against a behavioral AXI DataMover (see
//   axi_datamover_model.sv) so no block design is needed
// - Runs at reduced dimensions so that a full block case is still quick
// - The expected result is computed with a bit-exact model of the MAC, because the
//   accumulator is fixed-point and a real-arithmetic reference would not match;
//   the conversion back to floating point is checked exhaustively by
//   fix_to_float_tb, so a second converter instance is used as the reference here
// - Because the accumulator is fixed-point, every product is truncated
//   independently before being added, so the sum is order-independent and a plain
//   k loop matches the array's slice-by-slice accumulation exactly

module systolic_array_tb;

    localparam LARGE_BUFFER_DIM = 16;
    localparam ARRAY_DIM        = 4;
    localparam MUL_WIDTH        = 16;
    localparam EXP_WIDTH        = 5;
    localparam ACC_WIDTH        = 48;
    localparam ACC_LSB          = -24;

    localparam CMD_WIDTH  = 104;
    localparam ADDR_WIDTH = 64;
    localparam BLK_WIDTH  = $clog2(LARGE_BUFFER_DIM) + 1;

    localparam FRAC_WIDTH = MUL_WIDTH - EXP_WIDTH - 1;   // 10
    localparam MANT_WIDTH = FRAC_WIDTH + 1;              // 11
    localparam PROD_WIDTH = 2 * MANT_WIDTH;              // 22
    localparam BIAS       = (1 << (EXP_WIDTH - 1)) - 1;  // 15

    localparam signed [15:0] SHIFT_BIAS = 2*BIAS + 2*FRAC_WIDTH + ACC_LSB;  // 26

    localparam TILE_ELEMS = ARRAY_DIM * ARRAY_DIM;
    localparam ELEM_BYTES = MUL_WIDTH / 8;
    localparam MAX_ELEMS  = LARGE_BUFFER_DIM * LARGE_BUFFER_DIM;

    // a base with a non-zero upper word so the 64-bit address arithmetic is exercised
    localparam [63:0] MEM_BASE  = 64'h0000_0002_0000_0000;
    localparam        MEM_BYTES = 16384;

    localparam [63:0] A_ADDR = MEM_BASE + 64'd0;
    localparam [63:0] B_ADDR = MEM_BASE + 64'd4096;
    localparam [63:0] C_ADDR = MEM_BASE + 64'd8192;

    ////////////////////////////////////////////////////////////////////////////////
    // SIGNALS
    ////////////////////////////////////////////////////////////////////////////////

    logic clk = 0;
    logic rst;

    logic [7:0] stall_pct;

    logic [ADDR_WIDTH-1:0]  a_block_addr, b_block_addr, c_block_addr;
    logic [BLK_WIDTH-1:0]   blk_m, blk_k, blk_n;
    logic [1:0]             load_block_en;
    logic                   start_blk_comp, done_blk_comp, error_blk_comp;

    logic [CMD_WIDTH-1:0]   mm2s_cmd_tdata;
    logic                   mm2s_cmd_tvalid, mm2s_cmd_tready;
    logic [MUL_WIDTH-1:0]   mm2s_tdata;
    logic                   mm2s_tvalid, mm2s_tready, mm2s_tlast;
    logic [7:0]             mm2s_sts_tdata;
    logic                   mm2s_sts_tvalid, mm2s_sts_tready;

    logic [CMD_WIDTH-1:0]   s2mm_cmd_tdata;
    logic                   s2mm_cmd_tvalid, s2mm_cmd_tready;
    logic [MUL_WIDTH-1:0]   s2mm_tdata;
    logic [MUL_WIDTH/8-1:0] s2mm_tkeep;
    logic                   s2mm_tvalid, s2mm_tready, s2mm_tlast;
    logic [7:0]             s2mm_sts_tdata;
    logic                   s2mm_sts_tvalid, s2mm_sts_tready;

    // operands held for the reference computation
    logic [MUL_WIDTH-1:0] A [0:LARGE_BUFFER_DIM-1][0:LARGE_BUFFER_DIM-1];
    logic [MUL_WIDTH-1:0] B [0:LARGE_BUFFER_DIM-1][0:LARGE_BUFFER_DIM-1];
    logic [MUL_WIDTH-1:0] expected_C [0:MAX_ELEMS-1];

    // reference converter, driven directly rather than called as a function
    logic [ACC_WIDTH-1:0] ref_fixed;
    logic [MUL_WIDTH-1:0] ref_float;

    int errors = 0;
    int cases  = 0;

    ////////////////////////////////////////////////////////////////////////////////
    // INSTANCES
    ////////////////////////////////////////////////////////////////////////////////

    systolic_array #(.LARGE_BUFFER_DIM(LARGE_BUFFER_DIM),
                     .ARRAY_DIM(ARRAY_DIM),
                     .MUL_WIDTH(MUL_WIDTH),
                     .EXP_WIDTH(EXP_WIDTH),
                     .ACC_WIDTH(ACC_WIDTH),
                     .ACC_LSB(ACC_LSB)) dut(.*);

    axi_datamover_model #(.CMD_WIDTH(CMD_WIDTH),
                          .ADDR_WIDTH(ADDR_WIDTH),
                          .STREAM_WIDTH(MUL_WIDTH),
                          .MEM_BYTES(MEM_BYTES),
                          .MEM_BASE(MEM_BASE)) dm(.*);

    fix_to_float #(.MUL_WIDTH(MUL_WIDTH),
                   .EXP_WIDTH(EXP_WIDTH),
                   .ACC_WIDTH(ACC_WIDTH),
                   .ACC_LSB(ACC_LSB)) ref_converter(.fixed_in(ref_fixed),
                                                    .float_out(ref_float));

    always #5 clk = ~clk;

    ////////////////////////////////////////////////////////////////////////////////
    // BIT-EXACT MAC MODEL (mirrors mac.sv)
    ////////////////////////////////////////////////////////////////////////////////

    function automatic [ACC_WIDTH-1:0] ref_mac(logic [MUL_WIDTH-1:0] a,
                                               logic [MUL_WIDTH-1:0] b,
                                               logic [ACC_WIDTH-1:0] acc_in);
        logic                        sign_a, sign_b, prod_sign, is_zero;
        logic [EXP_WIDTH-1:0]        exp_a, exp_b;
        logic [FRAC_WIDTH-1:0]       frac_a, frac_b;
        logic [MANT_WIDTH-1:0]       mant_a, mant_b;
        logic [PROD_WIDTH-1:0]       mant_prod;
        logic signed [15:0]          shift_amt;
        logic [ACC_WIDTH-1:0]        aligned;
        logic signed [ACC_WIDTH-1:0] addend;

        {sign_a, exp_a, frac_a} = a;
        {sign_b, exp_b, frac_b} = b;

        is_zero = (exp_a == '0) || (exp_b == '0);

        mant_a    = {1'b1, frac_a};
        mant_b    = {1'b1, frac_b};
        mant_prod = mant_a * mant_b;
        prod_sign = sign_a ^ sign_b;

        shift_amt = signed'(16'(exp_a) + 16'(exp_b)) - SHIFT_BIAS;

        if (is_zero || (shift_amt <= -PROD_WIDTH) || (shift_amt >= ACC_WIDTH))
            aligned = '0;
        else if (shift_amt < 0)
            aligned = ACC_WIDTH'(mant_prod) >> unsigned'(-shift_amt);
        else
            aligned = ACC_WIDTH'(mant_prod) << unsigned'(shift_amt);

        addend = prod_sign ? -signed'(aligned) : signed'(aligned);

        return ACC_WIDTH'(signed'(acc_in) + addend);
    endfunction

    ////////////////////////////////////////////////////////////////////////////////
    // STIMULUS HELPERS
    ////////////////////////////////////////////////////////////////////////////////

    // exponents are kept near unity so that products stay well inside the
    // accumulator's range and the result never clamps on the way back out
    // (note $urandom_range takes the maximum first)
    function automatic [MUL_WIDTH-1:0] rand_operand();
        logic                  sign;
        logic [EXP_WIDTH-1:0]  exp;
        logic [FRAC_WIDTH-1:0] frac;

        sign = $urandom_range(1);
        frac = FRAC_WIDTH'($urandom());

        // occasionally emit a zero to exercise the MAC's flush-to-zero path
        if ($urandom_range(7) == 0) exp = '0;
        else                        exp = EXP_WIDTH'($urandom_range(BIAS + 3, BIAS - 4));

        return {sign, exp, frac};
    endfunction

    task automatic stage_operands(input int m, input int k, input int n, input bit fresh_a);
        for (int r = 0; r < k; r++) begin
            for (int c = 0; c < n; c++) begin
                B[r][c] = rand_operand();
                dm.poke_elem(B_ADDR + 64'((r*n + c) * ELEM_BYTES), B[r][c]);
            end
        end

        if (fresh_a) begin
            for (int r = 0; r < m; r++) begin
                for (int c = 0; c < k; c++) begin
                    A[r][c] = rand_operand();
                    dm.poke_elem(A_ADDR + 64'((r*k + c) * ELEM_BYTES), A[r][c]);
                end
            end
        end
    endtask

    // fills the A region with a recognizable pattern so that a run which is not
    // supposed to reload A fails loudly if it does
    task automatic corrupt_a_memory(input int m, input int k);
        for (int r = 0; r < m; r++) begin
            for (int c = 0; c < k; c++) begin
                dm.poke_elem(A_ADDR + 64'((r*k + c) * ELEM_BYTES), 16'h7BFF);
            end
        end
    endtask

    task automatic compute_expected(input int m, input int k, input int n);
        logic [ACC_WIDTH-1:0] acc;
        int                   tiles_j, idx;

        tiles_j = n / ARRAY_DIM;

        for (int ti = 0; ti < (m / ARRAY_DIM); ti++) begin
            for (int tj = 0; tj < tiles_j; tj++) begin
                for (int r = 0; r < ARRAY_DIM; r++) begin
                    for (int c = 0; c < ARRAY_DIM; c++) begin
                        acc = '0;

                        for (int kk = 0; kk < k; kk++) begin
                            acc = ref_mac(A[ti*ARRAY_DIM + r][kk], B[kk][tj*ARRAY_DIM + c], acc);
                        end

                        ref_fixed = acc;
                        #1;

                        idx = (ti*tiles_j + tj)*TILE_ELEMS + r*ARRAY_DIM + c;
                        expected_C[idx] = ref_float;
                    end
                end
            end
        end
    endtask

    task automatic check_result(input int m, input int n, input string label);
        logic [MUL_WIDTH-1:0] got;
        int                   tiles_j, idx, case_errors;

        tiles_j     = n / ARRAY_DIM;
        case_errors = 0;

        for (int ti = 0; ti < (m / ARRAY_DIM); ti++) begin
            for (int tj = 0; tj < tiles_j; tj++) begin
                for (int r = 0; r < ARRAY_DIM; r++) begin
                    for (int c = 0; c < ARRAY_DIM; c++) begin
                        idx = (ti*tiles_j + tj)*TILE_ELEMS + r*ARRAY_DIM + c;
                        got = dm.peek_elem(C_ADDR + 64'(idx * ELEM_BYTES));

                        if (got !== expected_C[idx]) begin
                            case_errors = case_errors + 1;

                            if (case_errors <= 8)
                                $display("  %s: tile (%0d,%0d) element (%0d,%0d): got=%h expected=%h",
                                         label, ti, tj, r, c, got, expected_C[idx]);
                        end
                    end
                end
            end
        end

        if (error_blk_comp !== 1'b0) begin
            $display("  %s: error_blk_comp asserted", label);
            case_errors = case_errors + 1;
        end

        if (case_errors != 0) begin
            $display("  %s: FAILED (%0d mismatches)", label, case_errors);
            errors = errors + case_errors;
        end else begin
            $display("  %s: passed", label);
        end
    endtask

    task automatic run_block(input int m, input int k, input int n, input bit reuse_a);
        blk_m            = BLK_WIDTH'(m);
        blk_k            = BLK_WIDTH'(k);
        blk_n            = BLK_WIDTH'(n);
        load_block_en[0] = ~reuse_a;
        load_block_en[1] = 1'b1;

        start_blk_comp = 1'b1;
        @(posedge clk);
        #1;

        wait (done_blk_comp === 1'b1);

        start_blk_comp = 1'b0;
        @(posedge clk);
        #1;
    endtask

    task automatic run_case(input int m, input int k, input int n, input logic [7:0] stall,
                            input string label);
        cases     = cases + 1;
        stall_pct = stall;

        stage_operands(m, k, n, 1'b1);
        compute_expected(m, k, n);
        run_block(m, k, n, 1'b0);
        check_result(m, n, label);
    endtask

    ////////////////////////////////////////////////////////////////////////////////
    // STIMULUS
    ////////////////////////////////////////////////////////////////////////////////

    initial begin
        stall_pct        = 8'd0;
        start_blk_comp   = 1'b0;
        load_block_en[0] = 1'b1;
        load_block_en[1] = 1'b1;
        a_block_addr     = A_ADDR;
        b_block_addr     = B_ADDR;
        c_block_addr     = C_ADDR;
        blk_m            = '0;
        blk_k            = '0;
        blk_n            = '0;

        rst = 1'b1;
        @(posedge clk);
        #1;
        rst = 1'b0;

        $display("systolic_array_tb: ARRAY_DIM=%0d LARGE_BUFFER_DIM=%0d", ARRAY_DIM, LARGE_BUFFER_DIM);

        run_case(ARRAY_DIM, ARRAY_DIM, ARRAY_DIM, 8'd0,  "single tile");
        run_case(8,  8,  8,  8'd0,  "multi tile");
        run_case(8,  4,  12, 8'd0,  "non-square");
        run_case(16, 16, 16, 8'd0,  "full block");
        run_case(8,  8,  8,  8'd30, "multi tile, backpressure");
        run_case(16, 16, 16, 8'd20, "full block, backpressure");

        // A stays resident across the second run; the memory it came from is
        // scribbled over first, so a spurious reload cannot go unnoticed
        cases     = cases + 1;
        stall_pct = 8'd0;
        stage_operands(8, 8, 8, 1'b1);
        compute_expected(8, 8, 8);
        run_block(8, 8, 8, 1'b0);
        check_result(8, 8, "resident A, first run");

        cases     = cases + 1;
        stage_operands(8, 8, 8, 1'b0);
        corrupt_a_memory(8, 8);
        compute_expected(8, 8, 8);
        run_block(8, 8, 8, 1'b1);
        check_result(8, 8, "resident A, reused");

        if (dm.access_errors != 0) begin
            $display("systolic_array_tb: %0d out-of-range memory accesses", dm.access_errors);
            errors = errors + dm.access_errors;
        end

        $display("");

        if (errors != 0) begin
            $display("systolic_array_tb: FAILED -- %0d mismatches across %0d cases", errors, cases);
            $display("");
            $stop;
        end

        $display("systolic_array_tb: all %0d cases passed!", cases);
        $stop;
    end

    ////////////////////////////////////////////////////////////////////////////////
    // WATCHDOG
    ////////////////////////////////////////////////////////////////////////////////

    initial begin
        #2_000_000;
        $display("systolic_array_tb: TIMEOUT -- the design is not making progress");
        $stop;
    end

endmodule
