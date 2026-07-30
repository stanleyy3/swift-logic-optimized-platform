// axi_ctrl_wrapper_tb.sv -- AXI4-Lite control wrapper testbench
//
// Drives real AXI4-Lite write/read transactions (not direct signal pokes)
// against axi_ctrl_wrapper standalone, with done_blk_comp/error_blk_comp
// driven by this testbench rather than a real systolic_array, so the
// busy/shadow/sticky-done logic can be checked in isolation.

module axi_ctrl_wrapper_tb;

    localparam LARGE_BUFFER_DIM = 256;
    localparam ADDR_WIDTH       = 64;
    localparam BLK_WIDTH        = $clog2(LARGE_BUFFER_DIM) + 1;

    logic clk = 0;
    logic rst;

    logic [11:0] s_axi_awaddr;
    logic        s_axi_awvalid, s_axi_awready;
    logic [31:0] s_axi_wdata;
    logic [3:0]  s_axi_wstrb;
    logic        s_axi_wvalid, s_axi_wready;
    logic [1:0]  s_axi_bresp;
    logic        s_axi_bvalid, s_axi_bready;
    logic [11:0] s_axi_araddr;
    logic        s_axi_arvalid, s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic [1:0]  s_axi_rresp;
    logic        s_axi_rvalid, s_axi_rready;

    logic [ADDR_WIDTH-1:0] a_block_addr, b_block_addr, c_block_addr;
    logic [BLK_WIDTH-1:0]  blk_m, blk_k, blk_n;
    logic [1:0]            load_block_en;
    logic                  start_blk_comp, done_blk_comp, error_blk_comp;

    int errors = 0;
    int checks = 0;

    always #5 clk = ~clk;

    axi_ctrl_wrapper #(.LARGE_BUFFER_DIM(LARGE_BUFFER_DIM),
                       .ADDR_WIDTH(ADDR_WIDTH)) dut(.*);

    ////////////////////////////////////////////////////////////////////////////////
    // AXI4-LITE DRIVER TASKS
    ////////////////////////////////////////////////////////////////////////////////

    task automatic axi_write(input logic [11:0] addr, input logic [31:0] data);
        s_axi_awaddr  = addr;
        s_axi_awvalid = 1'b1;
        s_axi_wdata   = data;
        s_axi_wstrb   = 4'hF;
        s_axi_wvalid  = 1'b1;
        s_axi_bready  = 1'b1;

        // wait for both address and data to be accepted
        do @(posedge clk); while (!(s_axi_awready && s_axi_wready));
        #1;
        s_axi_awvalid = 1'b0;
        s_axi_wvalid  = 1'b0;

        do @(posedge clk); while (!s_axi_bvalid);
        #1;
        s_axi_bready = 1'b0;
    endtask

    task automatic axi_read(input logic [11:0] addr, output logic [31:0] data);
        s_axi_araddr  = addr;
        s_axi_arvalid = 1'b1;
        s_axi_rready  = 1'b1;

        do @(posedge clk); while (!s_axi_arready);
        #1;
        s_axi_arvalid = 1'b0;

        do @(posedge clk); while (!s_axi_rvalid);
        data = s_axi_rdata;
        #1;
        s_axi_rready = 1'b0;
    endtask

    task automatic check_eq(input logic [63:0] got, input logic [63:0] expected, input string label);
        checks = checks + 1;
        if (got !== expected) begin
            errors = errors + 1;
            $display("  %s: FAILED got=%h expected=%h", label, got, expected);
        end else begin
            $display("  %s: passed", label);
        end
    endtask

    ////////////////////////////////////////////////////////////////////////////////
    // STIMULUS
    ////////////////////////////////////////////////////////////////////////////////

    logic [31:0] rdata;

    initial begin
        s_axi_awaddr = '0; s_axi_awvalid = 0;
        s_axi_wdata  = '0; s_axi_wstrb = '0; s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_araddr = '0; s_axi_arvalid = 0;
        s_axi_rready = 0;
        done_blk_comp = 0;
        error_blk_comp = 0;

        rst = 1'b1;
        @(posedge clk); @(posedge clk);
        #1;
        rst = 1'b0;
        @(posedge clk);

        // --- program the argument registers ---
        axi_write(12'h10, 32'h1000_0000); // a_block[31:0]
        axi_write(12'h14, 32'h0000_0002); // a_block[63:32]
        axi_write(12'h1C, 32'h2000_0000); // b_block[31:0]
        axi_write(12'h20, 32'h0000_0002); // b_block[63:32]
        axi_write(12'h28, 32'h3000_0000); // c_block[31:0]
        axi_write(12'h2C, 32'h0000_0002); // c_block[63:32]
        axi_write(12'h34, 32'd16);        // blk_m
        axi_write(12'h3C, 32'd8);         // blk_k
        axi_write(12'h44, 32'd24);        // blk_n
        axi_write(12'h4C, 32'd3);         // load_block_en

        // arguments must not reach the shadow outputs before ap_start
        check_eq(a_block_addr, 64'h0, "shadow not captured before ap_start");
        check_eq(start_blk_comp, 1'b0, "start_blk_comp idle before ap_start");

        axi_read(12'h00, rdata);
        check_eq(rdata[2], 1'b1, "ap_idle asserted before first run");

        // --- ap_start: shadow capture + level start_blk_comp ---
        axi_write(12'h00, 32'h1);
        #1;
        check_eq(start_blk_comp, 1'b1, "start_blk_comp asserted after ap_start");
        check_eq(a_block_addr, 64'h0000_0002_1000_0000, "a_block_addr shadow captured");
        check_eq(b_block_addr, 64'h0000_0002_2000_0000, "b_block_addr shadow captured");
        check_eq(c_block_addr, 64'h0000_0002_3000_0000, "c_block_addr shadow captured");
        check_eq(blk_m, 9'd16, "blk_m shadow captured");
        check_eq(blk_k, 9'd8,  "blk_k shadow captured");
        check_eq(blk_n, 9'd24, "blk_n shadow captured");
        check_eq(load_block_en, 2'd3, "load_block_en shadow captured");

        axi_read(12'h00, rdata);
        check_eq(rdata[2], 1'b0, "ap_idle deasserted while busy");
        check_eq(rdata[1], 1'b0, "ap_done not yet asserted");

        // a write mid-run must not disturb the shadow registers
        axi_write(12'h34, 32'd99);
        check_eq(blk_m, 9'd16, "shadow blk_m unaffected by mid-run write");

        // --- done_blk_comp: start_blk_comp deasserts, ap_done becomes sticky ---
        @(posedge clk);
        done_blk_comp = 1'b1;
        @(posedge clk);
        #1;
        done_blk_comp = 1'b0;

        check_eq(start_blk_comp, 1'b0, "start_blk_comp deasserted after done");

        axi_read(12'h00, rdata);
        check_eq(rdata[2], 1'b1, "ap_idle asserted after done");
        check_eq(rdata[1], 1'b1, "ap_done asserted after done");
        check_eq(rdata[3], 1'b1, "ap_ready asserted after done");

        // ap_done is read-to-clear
        axi_read(12'h00, rdata);
        check_eq(rdata[1], 1'b0, "ap_done cleared after read");

        // the deferred mid-run write is visible now that the run is over
        axi_write(12'h00, 32'h1);
        #1;
        check_eq(blk_m, 9'd99, "shadow blk_m updated by next ap_start");

        @(posedge clk);
        done_blk_comp = 1'b1;
        @(posedge clk);
        #1;
        done_blk_comp = 1'b0;

        // --- error register readback ---
        error_blk_comp = 1'b1;
        axi_read(12'h54, rdata);
        check_eq(rdata, 32'h1, "error register reflects error_blk_comp");
        error_blk_comp = 1'b0;

        $display("");
        if (errors != 0) begin
            $display("axi_ctrl_wrapper_tb: FAILED -- %0d/%0d checks failed", errors, checks);
            $stop;
        end
        $display("axi_ctrl_wrapper_tb: all %0d checks passed!", checks);
        $stop;
    end

    initial begin
        #100_000;
        $display("axi_ctrl_wrapper_tb: TIMEOUT -- the design is not making progress");
        $stop;
    end

endmodule
