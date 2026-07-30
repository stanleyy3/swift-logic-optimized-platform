// axi_datamover_model.sv -- Behavioral AXI DataMover (simulation only)
//
// - Stands in for the AXI DataMover IP so the RTL can be exercised before the
//   block design exists; not synthesizable
// - Implements the MM2S read channel, the S2MM write channel, and both status
//   channels against a flat byte-addressable memory
// - Only the command fields the design actually uses are decoded (SADDR and BTT);
//   TYPE, EOF, DSA, DRR, and TAG are ignored
// - 'stall_pct' injects idle cycles on both channels so that the design's stall
//   paths get exercised; zero means never stall
// - Flags an error on any access outside the modeled memory, which is the quickest
//   way to catch an address generation bug
//
// Handshake sampling is deliberately arranged to be race-free: a value driven by
// the design is only ever read while the design is being held stalled, so it is
// guaranteed stable rather than being sampled in the same time step it changes.
//
// Every wait compares against 1'b1 with '!==' rather than negating the signal.
// Before the first clock edge the design's state registers are uninitialized, so its
// handshake outputs are X, and '!X' evaluates to X, which counts as false. A
// negating wait would therefore fall straight through at time zero and act on a
// command that was never issued: the channel would go on to block forever on a
// status handshake the design never reaches, while the design blocked on a command
// the model never accepted.

module axi_datamover_model #(
    parameter CMD_WIDTH,
    parameter ADDR_WIDTH,
    parameter STREAM_WIDTH,
    parameter MEM_BYTES,
    parameter [63:0] MEM_BASE
) (
    input  logic                      clk,
    input  logic [7:0]                stall_pct,
    input  logic [CMD_WIDTH-1:0]      mm2s_cmd_tdata,
    input  logic                      mm2s_cmd_tvalid,
    output logic                      mm2s_cmd_tready,
    output logic [STREAM_WIDTH-1:0]   mm2s_tdata,
    output logic                      mm2s_tvalid,
    input  logic                      mm2s_tready,
    output logic                      mm2s_tlast,
    output logic [7:0]                mm2s_sts_tdata,
    output logic                      mm2s_sts_tvalid,
    input  logic                      mm2s_sts_tready,
    input  logic [CMD_WIDTH-1:0]      s2mm_cmd_tdata,
    input  logic                      s2mm_cmd_tvalid,
    output logic                      s2mm_cmd_tready,
    input  logic [STREAM_WIDTH-1:0]   s2mm_tdata,
    input  logic [STREAM_WIDTH/8-1:0] s2mm_tkeep,
    input  logic                      s2mm_tvalid,
    input  logic                      s2mm_tlast,
    output logic                      s2mm_tready,
    output logic [7:0]                s2mm_sts_tdata,
    output logic                      s2mm_sts_tvalid,
    input  logic                      s2mm_sts_tready
);

    localparam ELEM_BYTES   = STREAM_WIDTH / 8;
    localparam CMD_ADDR_LSB = 32;        // SADDR position in the command word
    localparam BTT_WIDTH    = 23;
    localparam STS_OKAY     = 8'h80;     // OKAY set, no error bits

    logic [7:0] mem [0:MEM_BYTES-1];

    logic [ADDR_WIDTH-1:0]  mm2s_addr, s2mm_addr;
    logic [BTT_WIDTH-1:0]   mm2s_btt, s2mm_btt;
    int                     mm2s_beats, s2mm_beats;
    int                     access_errors;

    ////////////////////////////////////////////////////////////////////////////////
    // MEMORY ACCESS
    ////////////////////////////////////////////////////////////////////////////////

    function automatic int mem_index(logic [ADDR_WIDTH-1:0] addr);
        logic [63:0] offset;

        offset = 64'(addr) - MEM_BASE;

        if ((64'(addr) < MEM_BASE) || (offset > (MEM_BYTES - ELEM_BYTES))) begin
            $error("datamover model: address %h is outside the modeled memory", addr);
            access_errors = access_errors + 1;
            return 0;
        end

        return int'(offset);
    endfunction

    function automatic [STREAM_WIDTH-1:0] mem_read_elem(logic [ADDR_WIDTH-1:0] addr);
        logic [STREAM_WIDTH-1:0] value;
        int                      base;

        base = mem_index(addr);

        for (int b = 0; b < ELEM_BYTES; b++) begin
            value[b*8 +: 8] = mem[base + b];
        end

        return value;
    endfunction

    task automatic mem_write_elem(logic [ADDR_WIDTH-1:0] addr, logic [STREAM_WIDTH-1:0] value);
        int base;

        base = mem_index(addr);

        for (int b = 0; b < ELEM_BYTES; b++) begin
            mem[base + b] = value[b*8 +: 8];
        end
    endtask

    // element-granularity helpers for the testbench to stage operands and check
    // results without reaching into the byte array itself
    task automatic poke_elem(logic [ADDR_WIDTH-1:0] addr, logic [STREAM_WIDTH-1:0] value);
        mem_write_elem(addr, value);
    endtask

    function automatic [STREAM_WIDTH-1:0] peek_elem(logic [ADDR_WIDTH-1:0] addr);
        return mem_read_elem(addr);
    endfunction

    ////////////////////////////////////////////////////////////////////////////////
    // STALL INJECTION
    ////////////////////////////////////////////////////////////////////////////////

    task automatic maybe_stall();
        while ((stall_pct != 0) && ($urandom_range(99) < int'(stall_pct))) begin
            @(posedge clk);
            #1;
        end
    endtask

    ////////////////////////////////////////////////////////////////////////////////
    // MM2S (MEMORY TO STREAM)
    ////////////////////////////////////////////////////////////////////////////////

    initial begin
        access_errors    = 0;
        mm2s_cmd_tready  = 1'b0;
        mm2s_tvalid      = 1'b0;
        mm2s_tlast       = 1'b0;
        mm2s_tdata       = '0;
        mm2s_sts_tvalid  = 1'b0;
        mm2s_sts_tdata   = '0;

        forever begin
            // the command is sampled while the design is still stalled, so its
            // fields cannot change underneath us
            while (mm2s_cmd_tvalid !== 1'b1) begin
                @(posedge clk);
                #1;
            end

            mm2s_addr  = mm2s_cmd_tdata[CMD_ADDR_LSB +: ADDR_WIDTH];
            mm2s_btt   = mm2s_cmd_tdata[BTT_WIDTH-1:0];
            mm2s_beats = int'(mm2s_btt) / ELEM_BYTES;

            mm2s_cmd_tready = 1'b1;
            @(posedge clk);
            #1;
            mm2s_cmd_tready = 1'b0;

            for (int i = 0; i < mm2s_beats; i++) begin
                mm2s_tvalid = 1'b0;
                maybe_stall();

                mm2s_tdata  = mem_read_elem(mm2s_addr + ADDR_WIDTH'(i * ELEM_BYTES));
                mm2s_tvalid = 1'b1;
                mm2s_tlast  = (i == (mm2s_beats - 1));

                // the beat transfers on the edge ending the first cycle in which
                // the consumer is ready
                while (mm2s_tready !== 1'b1) begin
                    @(posedge clk);
                    #1;
                end

                @(posedge clk);
                #1;
            end

            mm2s_tvalid = 1'b0;
            mm2s_tlast  = 1'b0;

            mm2s_sts_tdata  = STS_OKAY;
            mm2s_sts_tvalid = 1'b1;

            while (mm2s_sts_tready !== 1'b1) begin
                @(posedge clk);
                #1;
            end

            @(posedge clk);
            #1;
            mm2s_sts_tvalid = 1'b0;
        end
    end

    ////////////////////////////////////////////////////////////////////////////////
    // S2MM (STREAM TO MEMORY)
    ////////////////////////////////////////////////////////////////////////////////

    initial begin
        s2mm_cmd_tready = 1'b0;
        s2mm_tready     = 1'b0;
        s2mm_sts_tvalid = 1'b0;
        s2mm_sts_tdata  = '0;

        forever begin
            while (s2mm_cmd_tvalid !== 1'b1) begin
                @(posedge clk);
                #1;
            end

            s2mm_addr  = s2mm_cmd_tdata[CMD_ADDR_LSB +: ADDR_WIDTH];
            s2mm_btt   = s2mm_cmd_tdata[BTT_WIDTH-1:0];
            s2mm_beats = int'(s2mm_btt) / ELEM_BYTES;

            s2mm_cmd_tready = 1'b1;
            @(posedge clk);
            #1;
            s2mm_cmd_tready = 1'b0;

            for (int i = 0; i < s2mm_beats; i++) begin
                s2mm_tready = 1'b0;
                maybe_stall();

                s2mm_tready = 1'b1;

                // the producer holds its beat until we are ready, so the data is
                // stable to sample during this cycle
                while (s2mm_tvalid !== 1'b1) begin
                    @(posedge clk);
                    #1;
                end

                mem_write_elem(s2mm_addr + ADDR_WIDTH'(i * ELEM_BYTES), s2mm_tdata);

                if ((i == (s2mm_beats - 1)) && !s2mm_tlast)
                    $error("datamover model: expected tlast on the final beat of a command");

                @(posedge clk);
                #1;
            end

            s2mm_tready = 1'b0;

            s2mm_sts_tdata  = STS_OKAY;
            s2mm_sts_tvalid = 1'b1;

            while (s2mm_sts_tready !== 1'b1) begin
                @(posedge clk);
                #1;
            end

            @(posedge clk);
            #1;
            s2mm_sts_tvalid = 1'b0;
        end
    end

endmodule