// systolic_array.sv -- Systolic array (top module)
//
// - Floating-point MACs
//
// Terminology:
// - overall matrix: entire matrix that needs the operated on from host
// - block: sub-matrix of overall matrix that fits at once in large buffers
// - tile: output sub-matrix that is produced within PEs across slice passes
// - slice: a partial sum of a tile
//
// - One block matmul per 'start_ovr_comp'; the host schedules blocks and
//   accumulates across the overall K dimension (see control.sv for the memory
//   layout the host has to produce)
// - The memory-side interface is an AXI DataMover (PG022), which is instantiated
//   outside this module; everything here is the command, stream, and status
//   plumbing it expects
//
// (DO NOT DELETE)
// - Configuration for fp32:
//   - X, 32, 8, 64, -52
// - Configuration for fp16:
//   - X, 16, 5, 48, -24

`define PHYSICAL_ADDR_WIDTH 64

// DataMover command word; a 64-bit memory-mapped address widens it from 72 to 104
// bits, and the reserved field pads it out to a multiple of 8 for AXI4-Stream
`define DATAMOVER_CMD_WIDTH 104

module systolic_array #(
    parameter LARGE_BUFFER_DIM = 256, // dimension of 1 of two regions (for each operand block)
    parameter ARRAY_DIM = 8,
    parameter MUL_WIDTH = 16,
    parameter EXP_WIDTH = 5,
    parameter ACC_WIDTH = 48,
    parameter ACC_LSB = -24
) (
    input  logic                              clk,
    input  logic                              rst,
    input  logic [`PHYSICAL_ADDR_WIDTH-1:0]   a_block_addr,
    input  logic [`PHYSICAL_ADDR_WIDTH-1:0]   b_block_addr,
    input  logic [`PHYSICAL_ADDR_WIDTH-1:0]   c_block_addr,
    input  logic [$clog2(LARGE_BUFFER_DIM):0] blk_m,
    input  logic [$clog2(LARGE_BUFFER_DIM):0] blk_k,
    input  logic [$clog2(LARGE_BUFFER_DIM):0] blk_n,
    input  logic [1:0]                        load_block_en,
    input  logic                              start_ovr_comp,
    output logic                              done_ovr_comp,
    output logic                              error_ovr_comp,
    output logic [`DATAMOVER_CMD_WIDTH-1:0]   mm2s_cmd_tdata,
    output logic                              mm2s_cmd_tvalid,
    input  logic                              mm2s_cmd_tready,
    input  logic [MUL_WIDTH-1:0]              mm2s_tdata,
    input  logic                              mm2s_tvalid,
    output logic                              mm2s_tready,
    input  logic [7:0]                        mm2s_sts_tdata,
    input  logic                              mm2s_sts_tvalid,
    output logic                              mm2s_sts_tready,
    output logic [`DATAMOVER_CMD_WIDTH-1:0]   s2mm_cmd_tdata,
    output logic                              s2mm_cmd_tvalid,
    input  logic                              s2mm_cmd_tready,
    output logic [MUL_WIDTH-1:0]              s2mm_tdata,
    output logic [MUL_WIDTH/8-1:0]            s2mm_tkeep,
    output logic                              s2mm_tvalid,
    input  logic                              s2mm_tready,
    output logic                              s2mm_tlast,
    input  logic [7:0]                        s2mm_sts_tdata,
    input  logic                              s2mm_sts_tvalid,
    output logic                              s2mm_sts_tready
);

    // axi interface
    logic [ACC_WIDTH-1:0]            small_bufs_out_read_data;
    logic [`PHYSICAL_ADDR_WIDTH-1:0] mm2s_cmd_addr;
    logic [22:0]                     mm2s_cmd_bytes;
    logic [`PHYSICAL_ADDR_WIDTH-1:0] s2mm_cmd_addr;
    logic [22:0]                     s2mm_cmd_bytes;

    // large buffers
    logic large_bufs_write_en [0:1];
    logic [$clog2(LARGE_BUFFER_DIM)-1:0] large_bufs_write_addrs [0:1][0:1];
    logic [MUL_WIDTH-1:0]                large_bufs_write_data  [0:1];
    logic [$clog2(LARGE_BUFFER_DIM)-1:0] large_bufs_read_addrs  [0:1][0:1];

    // small buffers
    logic                         small_bufs_opr_write_en;
    logic [$clog2(ARRAY_DIM)-1:0] small_bufs_opr_write_addrs     [0:1][0:1];
    logic [MUL_WIDTH-1:0]         small_bufs_opr_write_data      [0:1];
    logic                         small_bufs_out_write_en;
    logic [ACC_WIDTH-1:0]         small_bufs_out_write_data      [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [$clog2(ARRAY_DIM)-1:0] small_bufs_opr_read_rowcol_idx;
    logic [$clog2(ARRAY_DIM)-1:0] small_bufs_out_read_addr       [0:1];

    // skew registers
    logic                 zero_skew_regs;
    logic                 read_imm_operands;
    logic [MUL_WIDTH-1:0] skew_regs_in_data [0:1][0:ARRAY_DIM-1];

    // PE array
    logic                 zero_pe_array;
    logic [MUL_WIDTH-1:0] pe_array_A_in_values [0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] pe_array_B_in_values [0:ARRAY_DIM-1];

    ////////////////////////////////////////////////////////////////////////////////
    // CONTROL
    ////////////////////////////////////////////////////////////////////////////////

    control #(.LARGE_BUFFER_DIM(LARGE_BUFFER_DIM),
              .ARRAY_DIM(ARRAY_DIM),
              .MUL_WIDTH(MUL_WIDTH),
              .ADDR_WIDTH(`PHYSICAL_ADDR_WIDTH)) control_0(.clk,
                                                           .rst,
                                                           .a_block_addr,
                                                           .b_block_addr,
                                                           .c_block_addr,
                                                           .blk_m,
                                                           .blk_k,
                                                           .blk_n,
                                                           .load_block_en,
                                                           .start_ovr_comp,
                                                           .done_ovr_comp,
                                                           .error_ovr_comp,
                                                           .large_bufs_write_en,
                                                           .large_bufs_write_addrs,
                                                           .large_bufs_read_addrs,
                                                           .small_bufs_opr_write_en,
                                                           .small_bufs_opr_write_addrs,
                                                           .small_bufs_opr_read_rowcol_idx,
                                                           .small_bufs_out_write_en,
                                                           .small_bufs_out_read_addr,
                                                           .zero_skew_regs,
                                                           .read_imm_operands,
                                                           .zero_pe_array,
                                                           .mm2s_cmd_addr,
                                                           .mm2s_cmd_bytes,
                                                           .mm2s_cmd_valid(mm2s_cmd_tvalid),
                                                           .mm2s_cmd_ready(mm2s_cmd_tready),
                                                           .mm2s_tvalid,
                                                           .mm2s_tready,
                                                           .mm2s_sts_tdata,
                                                           .mm2s_sts_tvalid,
                                                           .mm2s_sts_tready,
                                                           .s2mm_cmd_addr,
                                                           .s2mm_cmd_bytes,
                                                           .s2mm_cmd_valid(s2mm_cmd_tvalid),
                                                           .s2mm_cmd_ready(s2mm_cmd_tready),
                                                           .s2mm_tvalid,
                                                           .s2mm_tready,
                                                           .s2mm_tlast,
                                                           .s2mm_sts_tdata,
                                                           .s2mm_sts_tvalid,
                                                           .s2mm_sts_tready);

    ////////////////////////////////////////////////////////////////////////////////
    // AXI INTERFACE
    ////////////////////////////////////////////////////////////////////////////////

    // DataMover command word layout (PG022):
    //   [22:0] BTT, [23] TYPE, [29:24] DSA, [30] EOF, [31] DRR,
    //   [95:32] SADDR, [99:96] TAG, [103:100] RSVD
    // TYPE is always incrementing, EOF is always set so that the read master
    // asserts TLAST at the end of a command and the write master expects it, and
    // DSA/DRR stay clear because the data realignment engine is not used
    assign mm2s_cmd_tdata = {4'b0,             // RSVD
                             4'b0,             // TAG
                             mm2s_cmd_addr,    // SADDR
                             1'b0,             // DRR
                             1'b1,             // EOF
                             6'b0,             // DSA
                             1'b1,             // TYPE
                             mm2s_cmd_bytes};  // BTT

    assign s2mm_cmd_tdata = {4'b0,             // RSVD
                             4'b0,             // TAG
                             s2mm_cmd_addr,    // SADDR
                             1'b0,             // DRR
                             1'b1,             // EOF
                             6'b0,             // DSA
                             1'b1,             // TYPE
                             s2mm_cmd_bytes};  // BTT

    // a block load is a single command per operand, so the incoming stream is just
    // written straight into whichever region control has enabled
    assign large_bufs_write_data[0] = mm2s_tdata;
    assign large_bufs_write_data[1] = mm2s_tdata;

    assign s2mm_tkeep = '1;

    ////////////////////////////////////////////////////////////////////////////////
    // LARGE BUFFERS
    ////////////////////////////////////////////////////////////////////////////////

    large_bufs #(.BUFFER_DIM(LARGE_BUFFER_DIM),
                 .ELEM_SIZE(MUL_WIDTH),
                 .NUM_BANKS(ARRAY_DIM)) large_bufs(.clk,
                                                   .write_en(large_bufs_write_en),
                                                   .write_addrs(large_bufs_write_addrs),
                                                   .write_data(large_bufs_write_data),
                                                   .read_addrs(large_bufs_read_addrs),
                                                   .read_data(small_bufs_opr_write_data));

    ////////////////////////////////////////////////////////////////////////////////
    // SMALL BUFFERS (IMMEDIATE OPERANDS)
    ////////////////////////////////////////////////////////////////////////////////

    small_bufs #(.ARRAY_DIM(ARRAY_DIM),
                 .MUL_WIDTH(MUL_WIDTH),
                 .ACC_WIDTH(ACC_WIDTH)) operand_bufs_0(.clk,
                                                       .opr_write_en(small_bufs_opr_write_en),
                                                       .opr_write_addrs(small_bufs_opr_write_addrs),
                                                       .opr_write_data(small_bufs_opr_write_data),
                                                       .out_write_en(small_bufs_out_write_en),
                                                       .out_write_data(small_bufs_out_write_data),
                                                       .opr_read_rowcol_idx(small_bufs_opr_read_rowcol_idx),
                                                       .opr_read_data(skew_regs_in_data),
                                                       .out_read_addr(small_bufs_out_read_addr),
                                                       .out_read_data(small_bufs_out_read_data));

    ////////////////////////////////////////////////////////////////////////////////
    // SKEW REGISTERS
    ////////////////////////////////////////////////////////////////////////////////

    // skew registers for operand A
    skew_regs #(.ARRAY_DIM(ARRAY_DIM),
                .MUL_WIDTH(MUL_WIDTH)) skew_regs_A(.clk,
                                                   .zero_data(zero_skew_regs),
                                                   .read_operands(read_imm_operands),
                                                   .in_values(skew_regs_in_data[0]),
                                                   .out_values(pe_array_A_in_values));

    // skew registers for operand B
    skew_regs #(.ARRAY_DIM(ARRAY_DIM),
                .MUL_WIDTH(MUL_WIDTH)) skew_regs_B(.clk,
                                                   .zero_data(zero_skew_regs),
                                                   .read_operands(read_imm_operands),
                                                   .in_values(skew_regs_in_data[1]),
                                                   .out_values(pe_array_B_in_values));

    ////////////////////////////////////////////////////////////////////////////////
    // PE ARRAY
    ////////////////////////////////////////////////////////////////////////////////

    pe_array #(.ARRAY_DIM(ARRAY_DIM),
               .MUL_WIDTH(MUL_WIDTH),
               .EXP_WIDTH(EXP_WIDTH),
               .ACC_WIDTH(ACC_WIDTH),
               .ACC_LSB(ACC_LSB)) pe_array_0(.clk,
                                             .zero_data(zero_pe_array),
                                             .a_in_values(pe_array_A_in_values),
                                             .b_in_values(pe_array_B_in_values),
                                             .acc_values(small_bufs_out_write_data));

    ////////////////////////////////////////////////////////////////////////////////
    // OUTPUT TILE BUFFER
    ////////////////////////////////////////////////////////////////////////////////

    // the completed tile is held in the small buffers; converting on the way out
    // means the writeback stream is the same width as the operand stream
    fix_to_float #(.MUL_WIDTH(MUL_WIDTH),
                   .EXP_WIDTH(EXP_WIDTH),
                   .ACC_WIDTH(ACC_WIDTH),
                   .ACC_LSB(ACC_LSB)) out_converter(.fixed_in(small_bufs_out_read_data),
                                                    .float_out(s2mm_tdata));

endmodule