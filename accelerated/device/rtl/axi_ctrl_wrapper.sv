// axi_ctrl_wrapper.sv -- AXI4-Lite control-register slave for the systolic
// array kernel
//
// Implements the register map and ap_ctrl_hs semantics pinned down in
// accelerated/device/CONTROL_INTERFACE.md:
// - offsets 0x00-0x0C are the standard ap_ctrl_hs bank (GIER/IP_IER/IP_ISR
//   are decoded but unused, since the host polls ap_done rather than using
//   interrupts)
// - offsets 0x10-0x54 are the seven argument registers, captured into shadow
//   flops on the ap_start edge so a write that lands mid-run cannot move the
//   block geometry under a running FSM (control.sv requires the descriptors
//   stay stable from start to done)
// - 'busy' turns the self-clearing ap_start pulse into the level
//   'start_blk_comp' that control.sv requires held until 'done_blk_comp'
// - ap_done is sticky (set by done_blk_comp) and clears on a read of CTRL,
//   which is how a polling driver (no interrupts enabled) observes it exactly
//   once per run

module axi_ctrl_wrapper #(
    parameter LARGE_BUFFER_DIM = 256,
    parameter ADDR_WIDTH       = 64,
    parameter C_S_AXI_ADDR_WIDTH = 12,
    parameter C_S_AXI_DATA_WIDTH = 32
) (
    input  logic clk,
    input  logic rst,

    // AXI4-Lite slave (s_axi_control)
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                          s_axi_awvalid,
    output logic                          s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [C_S_AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  logic                          s_axi_wvalid,
    output logic                          s_axi_wready,
    output logic [1:0]                    s_axi_bresp,
    output logic                          s_axi_bvalid,
    input  logic                          s_axi_bready,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                          s_axi_arvalid,
    output logic                          s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]                    s_axi_rresp,
    output logic                          s_axi_rvalid,
    input  logic                          s_axi_rready,

    // shadow outputs to systolic_array
    output logic [ADDR_WIDTH-1:0]             a_block_addr,
    output logic [ADDR_WIDTH-1:0]             b_block_addr,
    output logic [ADDR_WIDTH-1:0]             c_block_addr,
    output logic [$clog2(LARGE_BUFFER_DIM):0] blk_m,
    output logic [$clog2(LARGE_BUFFER_DIM):0] blk_k,
    output logic [$clog2(LARGE_BUFFER_DIM):0] blk_n,
    output logic [1:0]                        load_block_en,
    output logic                              start_blk_comp,
    input  logic                              done_blk_comp,
    input  logic                              error_blk_comp
);

    localparam BLK_WIDTH = $clog2(LARGE_BUFFER_DIM) + 1;

    // register offsets (byte addresses)
    localparam ADDR_CTRL         = 12'h00;
    localparam ADDR_GIER         = 12'h04;
    localparam ADDR_IP_IER       = 12'h08;
    localparam ADDR_IP_ISR       = 12'h0C;
    localparam ADDR_A_BLOCK_LO   = 12'h10;
    localparam ADDR_A_BLOCK_HI   = 12'h14;
    localparam ADDR_B_BLOCK_LO   = 12'h1C;
    localparam ADDR_B_BLOCK_HI   = 12'h20;
    localparam ADDR_C_BLOCK_LO   = 12'h28;
    localparam ADDR_C_BLOCK_HI   = 12'h2C;
    localparam ADDR_BLK_M        = 12'h34;
    localparam ADDR_BLK_K        = 12'h3C;
    localparam ADDR_BLK_N        = 12'h44;
    localparam ADDR_LOAD_BLK_EN  = 12'h4C;
    localparam ADDR_ERROR        = 12'h54;

    ////////////////////////////////////////////////////////////////////////////////
    // LIVE ARGUMENT REGISTERS (written directly by AXI)
    ////////////////////////////////////////////////////////////////////////////////

    logic [31:0] a_block_lo_reg, a_block_hi_reg;
    logic [31:0] b_block_lo_reg, b_block_hi_reg;
    logic [31:0] c_block_lo_reg, c_block_hi_reg;
    logic [31:0] blk_m_reg, blk_k_reg, blk_n_reg;
    logic [31:0] load_block_en_reg;

    ////////////////////////////////////////////////////////////////////////////////
    // ap_ctrl_hs STATE: busy / sticky done
    ////////////////////////////////////////////////////////////////////////////////

    logic busy;
    logic ap_done;
    logic ap_start_pulse;
    logic ctrl_read_fire;

    assign start_blk_comp = busy;

    always_ff @(posedge clk) begin
        if (rst) busy <= 1'b0;
        else if (ap_start_pulse) busy <= 1'b1;
        else if (done_blk_comp) busy <= 1'b0;
    end

    always_ff @(posedge clk) begin
        if (rst) ap_done <= 1'b0;
        else if (done_blk_comp) ap_done <= 1'b1;
        else if (ctrl_read_fire) ap_done <= 1'b0;
    end

    ////////////////////////////////////////////////////////////////////////////////
    // SHADOW ARGUMENT REGISTERS (captured on the ap_start edge, fed to the array)
    ////////////////////////////////////////////////////////////////////////////////

    always_ff @(posedge clk) begin
        if (rst) begin
            a_block_addr   <= '0;
            b_block_addr   <= '0;
            c_block_addr   <= '0;
            blk_m          <= '0;
            blk_k          <= '0;
            blk_n          <= '0;
            load_block_en  <= '0;
        end else if (ap_start_pulse) begin
            a_block_addr   <= {a_block_hi_reg, a_block_lo_reg};
            b_block_addr   <= {b_block_hi_reg, b_block_lo_reg};
            c_block_addr   <= {c_block_hi_reg, c_block_lo_reg};
            blk_m          <= blk_m_reg[BLK_WIDTH-1:0];
            blk_k          <= blk_k_reg[BLK_WIDTH-1:0];
            blk_n          <= blk_n_reg[BLK_WIDTH-1:0];
            load_block_en  <= load_block_en_reg[1:0];
        end
    end

    ////////////////////////////////////////////////////////////////////////////////
    // AXI4-LITE WRITE CHANNEL
    ////////////////////////////////////////////////////////////////////////////////

    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    logic                          axi_awready;
    logic                          axi_wready;
    logic                          axi_bvalid;
    logic                          slv_reg_wren;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = axi_bvalid;

    always_ff @(posedge clk) begin
        if (rst) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= '0;
        end else if (~axi_awready && s_axi_awvalid && s_axi_wvalid) begin
            axi_awready <= 1'b1;
            axi_awaddr  <= s_axi_awaddr;
        end else begin
            axi_awready <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) axi_wready <= 1'b0;
        else if (~axi_wready && s_axi_wvalid && s_axi_awvalid) axi_wready <= 1'b1;
        else axi_wready <= 1'b0;
    end

    assign slv_reg_wren = axi_wready & s_axi_wvalid & axi_awready & s_axi_awvalid;

    always_ff @(posedge clk) begin
        if (rst) axi_bvalid <= 1'b0;
        else if (slv_reg_wren && ~axi_bvalid) axi_bvalid <= 1'b1;
        else if (s_axi_bready && axi_bvalid) axi_bvalid <= 1'b0;
    end

    // ap_start is a one-cycle pulse: a write to CTRL bit 0 while not busy: the
    // shadow capture and 'busy' set both happen off this same pulse, and it is
    // not latched anywhere as a persistent "ap_start register" bit
    assign ap_start_pulse = slv_reg_wren && (axi_awaddr == ADDR_CTRL) &&
                             s_axi_wstrb[0] && s_axi_wdata[0] && ~busy;

    always_ff @(posedge clk) begin
        if (rst) begin
            a_block_lo_reg     <= '0;
            a_block_hi_reg     <= '0;
            b_block_lo_reg     <= '0;
            b_block_hi_reg     <= '0;
            c_block_lo_reg     <= '0;
            c_block_hi_reg     <= '0;
            blk_m_reg          <= '0;
            blk_k_reg          <= '0;
            blk_n_reg          <= '0;
            load_block_en_reg  <= '0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr)
                ADDR_A_BLOCK_LO:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) a_block_lo_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_A_BLOCK_HI:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) a_block_hi_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_B_BLOCK_LO:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) b_block_lo_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_B_BLOCK_HI:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) b_block_hi_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_C_BLOCK_LO:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) c_block_lo_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_C_BLOCK_HI:  for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) c_block_hi_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_BLK_M:       for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) blk_m_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_BLK_K:       for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) blk_k_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_BLK_N:       for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) blk_n_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                ADDR_LOAD_BLK_EN: for (int b = 0; b < 4; b++) if (s_axi_wstrb[b]) load_block_en_reg[8*b +: 8] <= s_axi_wdata[8*b +: 8];
                default: ; // CTRL/GIER/IP_IER/IP_ISR/ERROR writes are no-ops
            endcase
        end
    end

    ////////////////////////////////////////////////////////////////////////////////
    // AXI4-LITE READ CHANNEL
    ////////////////////////////////////////////////////////////////////////////////

    logic                          axi_arready;
    logic [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    logic                          axi_rvalid;
    logic [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    logic                          slv_reg_rden;

    assign s_axi_arready = axi_arready;
    assign s_axi_rvalid  = axi_rvalid;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rdata   = axi_rdata;

    always_ff @(posedge clk) begin
        if (rst) begin
            axi_arready <= 1'b0;
            axi_araddr  <= '0;
        end else if (~axi_arready && s_axi_arvalid) begin
            axi_arready <= 1'b1;
            axi_araddr  <= s_axi_araddr;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) axi_rvalid <= 1'b0;
        else if (slv_reg_rden && ~axi_rvalid) axi_rvalid <= 1'b1;
        else if (s_axi_rready && axi_rvalid) axi_rvalid <= 1'b0;
    end

    assign slv_reg_rden = axi_arready & s_axi_arvalid & ~axi_rvalid;

    // ap_done clears on the cycle CTRL is actually driven onto the read data
    // bus, i.e. when the read that exposes it fires
    assign ctrl_read_fire = slv_reg_rden && (axi_araddr == ADDR_CTRL);

    always_ff @(posedge clk) begin
        if (slv_reg_rden) begin
            case (axi_araddr)
                // [3]=ap_ready (mirrors ap_done: not pipelined/auto-restarting),
                // [2]=ap_idle, [1]=ap_done, [0]=ap_start (write-only, reads back 0)
                ADDR_CTRL:        axi_rdata <= {28'b0, ap_done, ~busy, ap_done, 1'b0};
                ADDR_GIER:        axi_rdata <= '0;
                ADDR_IP_IER:      axi_rdata <= '0;
                ADDR_IP_ISR:      axi_rdata <= '0;
                ADDR_A_BLOCK_LO:  axi_rdata <= a_block_lo_reg;
                ADDR_A_BLOCK_HI:  axi_rdata <= a_block_hi_reg;
                ADDR_B_BLOCK_LO:  axi_rdata <= b_block_lo_reg;
                ADDR_B_BLOCK_HI:  axi_rdata <= b_block_hi_reg;
                ADDR_C_BLOCK_LO:  axi_rdata <= c_block_lo_reg;
                ADDR_C_BLOCK_HI:  axi_rdata <= c_block_hi_reg;
                ADDR_BLK_M:       axi_rdata <= blk_m_reg;
                ADDR_BLK_K:       axi_rdata <= blk_k_reg;
                ADDR_BLK_N:       axi_rdata <= blk_n_reg;
                ADDR_LOAD_BLK_EN: axi_rdata <= load_block_en_reg;
                ADDR_ERROR:       axi_rdata <= {31'b0, error_blk_comp};
                default:          axi_rdata <= '0;
            endcase
        end
    end

endmodule
