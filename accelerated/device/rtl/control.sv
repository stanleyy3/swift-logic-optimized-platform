// control.sv -- Control
//
// - One block matmul per 'start_blk_comp': C := A @ B, where A is blk_m x blk_k
//   and B is blk_k x blk_n, all of which must be multiples of ARRAY_DIM and no
//   greater than LARGE_BUFFER_DIM
// - The host schedules blocks and accumulates across the overall K dimension;
//   this module never accumulates across two starts
// - Strictly sequential by design (this is the unoptimized baseline): a slice
//   fully loads its operand tiles, feeds them, and drains the array before the
//   next slice begins
// - The PE accumulators are zeroed only at the start of a tile, so every slice
//   of a tile accumulates into them consecutively
// - The only overlap is the output writeback, which runs against the next tile's
//   computation off the snapshot held in the small buffers
//
// Operand block layout in the large buffers:
// - A row stride and B row stride are both LARGE_BUFFER_DIM, so every large
//   buffer address is a plain concatenation of a row and a column counter
//
// Host memory layout:
// - A block: contiguous, blk_m x blk_k, row-major
// - B block: contiguous, blk_k x blk_n, row-major
// - C block: tile-major, tile (ti,tj) at element offset
//   (ti*tiles_j + tj)*ARRAY_DIM**2, each tile row-major
//
// - The block descriptor inputs must be held stable from 'start_blk_comp' until
//   'done_blk_comp'

module control #(
    parameter LARGE_BUFFER_DIM,
    parameter ARRAY_DIM,
    parameter MUL_WIDTH,
    parameter ADDR_WIDTH
) (
    input  logic                                clk,
    input  logic                                rst,
    input  logic [ADDR_WIDTH-1:0]               a_block_addr,
    input  logic [ADDR_WIDTH-1:0]               b_block_addr,
    input  logic [ADDR_WIDTH-1:0]               c_block_addr,
    input  logic [$clog2(LARGE_BUFFER_DIM):0]   blk_m,
    input  logic [$clog2(LARGE_BUFFER_DIM):0]   blk_k,
    input  logic [$clog2(LARGE_BUFFER_DIM):0]   blk_n,
    input  logic [1:0]                          load_block_en,
    input  logic                                start_blk_comp,
    output logic                                done_blk_comp,
    output logic                                error_blk_comp,
    output logic                                large_bufs_write_en            [0:1],
    output logic [$clog2(LARGE_BUFFER_DIM)-1:0] large_bufs_write_addrs         [0:1][0:1],
    output logic [$clog2(LARGE_BUFFER_DIM)-1:0] large_bufs_read_addrs          [0:1][0:1],
    output logic                                small_bufs_opr_write_en,
    output logic [$clog2(ARRAY_DIM)-1:0]        small_bufs_opr_write_addrs     [0:1][0:1],
    output logic [$clog2(ARRAY_DIM)-1:0]        small_bufs_opr_read_rowcol_idx,
    output logic                                small_bufs_out_write_en,
    output logic [$clog2(ARRAY_DIM)-1:0]        small_bufs_out_read_addr       [0:1],
    output logic                                zero_skew_regs,
    output logic                                read_imm_operands,
    output logic                                zero_pe_array,
    output logic [ADDR_WIDTH-1:0]               mm2s_cmd_addr,
    output logic [22:0]                         mm2s_cmd_bytes,
    output logic                                mm2s_cmd_valid,
    input  logic                                mm2s_cmd_ready,
    input  logic                                mm2s_tvalid,
    output logic                                mm2s_tready,
    input  logic [7:0]                          mm2s_sts_tdata,
    input  logic                                mm2s_sts_tvalid,
    output logic                                mm2s_sts_tready,
    output logic [ADDR_WIDTH-1:0]               s2mm_cmd_addr,
    output logic [22:0]                         s2mm_cmd_bytes,
    output logic                                s2mm_cmd_valid,
    input  logic                                s2mm_cmd_ready,
    output logic                                s2mm_tvalid,
    input  logic                                s2mm_tready,
    output logic                                s2mm_tlast,
    input  logic [7:0]                          s2mm_sts_tdata,
    input  logic                                s2mm_sts_tvalid,
    output logic                                s2mm_sts_tready
);

    localparam LB_ADDR_WIDTH  = $clog2(LARGE_BUFFER_DIM);       // 8
    localparam SB_ADDR_WIDTH  = $clog2(ARRAY_DIM);              // 3
    localparam TILE_IDX_WIDTH = LB_ADDR_WIDTH - SB_ADDR_WIDTH;  // 5
    localparam TILE_CNT_WIDTH = TILE_IDX_WIDTH + 1;             // 6, holds the count itself
    localparam ELEM_IDX_WIDTH = 2 * SB_ADDR_WIDTH;              // 6, indexes one tile
    localparam LOAD_WIDTH     = ELEM_IDX_WIDTH + 1;             // 7, counts one past a tile
    localparam CYCLE_WIDTH    = SB_ADDR_WIDTH + 2;              // 5
    localparam BYTES_WIDTH    = 23;                             // DataMover BTT field

    localparam ELEM_BYTES   = MUL_WIDTH / 8;              // 2
    localparam TILE_ELEMS   = ARRAY_DIM * ARRAY_DIM;      // 64
    localparam TILE_BYTES   = TILE_ELEMS * ELEM_BYTES;    // 128
    localparam DRAIN_CYCLES = 3 * ARRAY_DIM - 2;          // 22

    // main sequencer; S_IDLE must stay first so that a reset value of 0 lands there
    typedef enum logic [3:0] {
        S_IDLE,
        S_A_CMD, S_A_FILL, S_A_STS,
        S_B_CMD, S_B_FILL, S_B_STS,
        S_TILE_INIT,
        S_LOAD, S_LEAD, S_FEED, S_DRAIN,
        S_WAIT_WB, S_SNAP,
        S_WAIT_LAST, S_DONE
    } state_t;

    // output writeback sequencer
    typedef enum logic [1:0] { S_WB_IDLE, S_WB_CMD, S_WB_STREAM, S_WB_STS } wb_state_t;

    // the flops carry the raw encodings; the enum-typed copies are what the rest of
    // the module compares against, and are what shows up by name in waveforms
    state_t     state, next_state;
    wb_state_t  wb_state, next_wb_state;
    logic [3:0] state_bits;
    logic [1:0] wb_state_bits;

    // block extents in tiles
    logic [TILE_CNT_WIDTH-1:0] tiles_i, tiles_j, slices;

    // block fill
    logic [LB_ADDR_WIDTH-1:0] fill_row, fill_col;
    logic [LB_ADDR_WIDTH-1:0] fill_col_limit, fill_row_limit;
    logic                     fill_beat, fill_col_wrap, fill_done, fill_clear;
    logic [BYTES_WIDTH-1:0]   a_block_bytes, b_block_bytes;

    // tile and slice position
    logic [TILE_CNT_WIDTH-1:0] ti, tj, s;
    logic                      tj_wrap, last_tile, last_slice, tile_clear;

    // operand tile load
    logic [LOAD_WIDTH-1:0]     load_cnt, load_prev;
    logic [SB_ADDR_WIDTH-1:0]  load_r_rd, load_c_rd, load_r_wr, load_c_wr;
    logic                      load_clear, load_done;

    // slice computation
    logic [CYCLE_WIDTH-1:0] comp_cycle;
    logic                   comp_clear, comp_en, feed_done, drain_done;

    // output writeback
    logic [ELEM_IDX_WIDTH-1:0] wb_idx;
    logic                      wb_phase, wb_start, wb_busy, wb_clear;
    logic                      wb_cmd_accepted, wb_beat_accepted, wb_last_elem;
    logic [ADDR_WIDTH-1:0]     c_addr, next_c_addr;
    logic                      c_addr_en;

    // status
    logic mm2s_sts_accepted, s2mm_sts_accepted, sts_error, error_clear;

    ////////////////////////////////////////////////////////////////////////////////
    // FSM
    ////////////////////////////////////////////////////////////////////////////////

    flop_r #(.WIDTH(4), .RST(0)) fsm(.clk,
                                     .rst,
                                     .d(next_state),
                                     .q(state_bits));

    assign state = state_t'(state_bits);

    always_comb begin
        case (state)
            S_IDLE:      if (!start_blk_comp)      next_state = S_IDLE;
                         else if (load_block_en[0]) next_state = S_A_CMD;
                         else if (load_block_en[1]) next_state = S_B_CMD;
                         else                       next_state = S_TILE_INIT;

            S_A_CMD:     next_state = mm2s_cmd_ready ? S_A_FILL : S_A_CMD;
            S_A_FILL:    next_state = fill_done ? S_A_STS : S_A_FILL;
            S_A_STS:     if (!mm2s_sts_tvalid)      next_state = S_A_STS;
                         else if (load_block_en[1]) next_state = S_B_CMD;
                         else                       next_state = S_TILE_INIT;

            S_B_CMD:     next_state = mm2s_cmd_ready ? S_B_FILL : S_B_CMD;
            S_B_FILL:    next_state = fill_done ? S_B_STS : S_B_FILL;
            S_B_STS:     next_state = mm2s_sts_tvalid ? S_TILE_INIT : S_B_STS;

            S_TILE_INIT: next_state = S_LOAD;
            S_LOAD:      next_state = load_done ? S_LEAD : S_LOAD;
            S_LEAD:      next_state = S_FEED;
            S_FEED:      next_state = feed_done ? S_DRAIN : S_FEED;
            S_DRAIN:     if (!drain_done)  next_state = S_DRAIN;
                         else if (last_slice) next_state = S_WAIT_WB;
                         else                 next_state = S_LOAD;

            // the snapshot overwrites the buffer the writeback reads from, so it
            // has to wait for any writeback still in flight
            S_WAIT_WB:   next_state = wb_busy ? S_WAIT_WB : S_SNAP;
            S_SNAP:      next_state = last_tile ? S_WAIT_LAST : S_TILE_INIT;

            S_WAIT_LAST: next_state = wb_busy ? S_WAIT_LAST : S_DONE;
            S_DONE:      next_state = start_blk_comp ? S_DONE : S_IDLE;

            default:     next_state = S_IDLE;
        endcase
    end

    assign done_blk_comp = (state == S_DONE);

    assign tiles_i = TILE_CNT_WIDTH'(blk_m >> SB_ADDR_WIDTH);
    assign tiles_j = TILE_CNT_WIDTH'(blk_n >> SB_ADDR_WIDTH);
    assign slices  = TILE_CNT_WIDTH'(blk_k >> SB_ADDR_WIDTH);

    ////////////////////////////////////////////////////////////////////////////////
    // BLOCK LOADING
    ////////////////////////////////////////////////////////////////////////////////

    assign a_block_bytes = BYTES_WIDTH'(blk_m) * BYTES_WIDTH'(blk_k) * BYTES_WIDTH'(ELEM_BYTES);
    assign b_block_bytes = BYTES_WIDTH'(blk_k) * BYTES_WIDTH'(blk_n) * BYTES_WIDTH'(ELEM_BYTES);

    // one command fills a whole block, since the host stages it contiguously
    assign mm2s_cmd_valid = (state == S_A_CMD) | (state == S_B_CMD);
    assign mm2s_cmd_addr  = (state == S_A_CMD) ? a_block_addr  : b_block_addr;
    assign mm2s_cmd_bytes = (state == S_A_CMD) ? a_block_bytes : b_block_bytes;

    assign mm2s_tready = (state == S_A_FILL) | (state == S_B_FILL);
    assign fill_beat   = mm2s_tvalid & mm2s_tready;

    // the block occupies the leading rows and columns of its region, so the column
    // counter wraps at the block's width rather than at the buffer's; a limit of
    // LARGE_BUFFER_DIM truncates to zero here, which still compares correctly
    // because the wrap tests against 'limit - 1'
    assign fill_col_limit = (state == S_A_FILL) ? LB_ADDR_WIDTH'(blk_k) : LB_ADDR_WIDTH'(blk_n);
    assign fill_row_limit = (state == S_A_FILL) ? LB_ADDR_WIDTH'(blk_m) : LB_ADDR_WIDTH'(blk_k);

    assign fill_col_wrap = fill_beat & (fill_col == (fill_col_limit - 1'b1));
    assign fill_done     = fill_col_wrap & (fill_row == (fill_row_limit - 1'b1));
    assign fill_clear    = rst | (state == S_A_CMD) | (state == S_B_CMD);

    flop_enr #(.WIDTH(LB_ADDR_WIDTH), .RST(0)) fill_col_reg(.clk,
                                                            .rst(fill_clear),
                                                            .en(fill_beat),
                                                            .d(fill_col_wrap ? '0 : (fill_col + 1'b1)),
                                                            .q(fill_col));

    flop_enr #(.WIDTH(LB_ADDR_WIDTH), .RST(0)) fill_row_reg(.clk,
                                                            .rst(fill_clear),
                                                            .en(fill_col_wrap),
                                                            .d(fill_row + 1'b1),
                                                            .q(fill_row));

    assign large_bufs_write_en[0] = (state == S_A_FILL) & fill_beat;
    assign large_bufs_write_en[1] = (state == S_B_FILL) & fill_beat;

    assign large_bufs_write_addrs[0][0] = fill_row;
    assign large_bufs_write_addrs[0][1] = fill_col;
    assign large_bufs_write_addrs[1][0] = fill_row;
    assign large_bufs_write_addrs[1][1] = fill_col;

    ////////////////////////////////////////////////////////////////////////////////
    // TILE LIFECYCLE
    ////////////////////////////////////////////////////////////////////////////////

    assign tj_wrap    = (tj == (tiles_j - 1'b1));
    assign last_tile  = tj_wrap & (ti == (tiles_i - 1'b1));
    assign tile_clear = rst | (state == S_IDLE);

    flop_enr #(.WIDTH(TILE_CNT_WIDTH), .RST(0)) tj_reg(.clk,
                                                       .rst(tile_clear),
                                                       .en(state == S_SNAP),
                                                       .d(tj_wrap ? '0 : (tj + 1'b1)),
                                                       .q(tj));

    flop_enr #(.WIDTH(TILE_CNT_WIDTH), .RST(0)) ti_reg(.clk,
                                                       .rst(tile_clear),
                                                       .en((state == S_SNAP) & tj_wrap),
                                                       .d(ti + 1'b1),
                                                       .q(ti));

    // the PE accumulators are cleared here and nowhere else, so all of a tile's
    // slices accumulate into them
    assign zero_pe_array = (state == S_TILE_INIT);

    // takes the completed tile out of the PEs and into the buffer the writeback
    // reads from, freeing the array for the next tile
    assign small_bufs_out_write_en = (state == S_SNAP);

    ////////////////////////////////////////////////////////////////////////////////
    // SLICE LIFECYCLE
    ////////////////////////////////////////////////////////////////////////////////

    assign last_slice = (s == (slices - 1'b1));

    flop_enr #(.WIDTH(TILE_CNT_WIDTH), .RST(0)) s_reg(.clk,
                                                      .rst(rst | (state == S_TILE_INIT)),
                                                      .en(drain_done),
                                                      .d(s + 1'b1),
                                                      .q(s));

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND TILE LOAD
    ////////////////////////////////////////////////////////////////////////////////

    // the large buffer read is registered, so the write into the small buffers
    // trails the read by one cycle; the counter therefore runs one step past the
    // tile to retire the last element
    assign load_clear = rst | (state != S_LOAD);
    assign load_done  = (load_cnt == LOAD_WIDTH'(TILE_ELEMS));
    assign load_prev  = load_cnt - 1'b1;

    flop_enr #(.WIDTH(LOAD_WIDTH), .RST(0)) load_cnt_reg(.clk,
                                                         .rst(load_clear),
                                                         .en(state == S_LOAD),
                                                         .d(load_cnt + 1'b1),
                                                         .q(load_cnt));

    assign load_r_rd = load_cnt[ELEM_IDX_WIDTH-1:SB_ADDR_WIDTH];
    assign load_c_rd = load_cnt[SB_ADDR_WIDTH-1:0];
    assign load_r_wr = load_prev[ELEM_IDX_WIDTH-1:SB_ADDR_WIDTH];
    assign load_c_wr = load_prev[SB_ADDR_WIDTH-1:0];

    // A element (r,c) of slice s lives at row ti*ARRAY_DIM+r, column s*ARRAY_DIM+c;
    // B element (r,c) at row s*ARRAY_DIM+r, column tj*ARRAY_DIM+c
    assign large_bufs_read_addrs[0][0] = {ti[TILE_IDX_WIDTH-1:0], load_r_rd};
    assign large_bufs_read_addrs[0][1] = {s[TILE_IDX_WIDTH-1:0],  load_c_rd};
    assign large_bufs_read_addrs[1][0] = {s[TILE_IDX_WIDTH-1:0],  load_r_rd};
    assign large_bufs_read_addrs[1][1] = {tj[TILE_IDX_WIDTH-1:0], load_c_rd};

    assign small_bufs_opr_write_en = (state == S_LOAD) & (load_cnt != '0);

    // A is stored transposed so that a row read out of its buffer is a column of
    // the tile, which is what the horizontal PE stream consumes
    assign small_bufs_opr_write_addrs[0][0] = load_c_wr;
    assign small_bufs_opr_write_addrs[0][1] = load_r_wr;
    assign small_bufs_opr_write_addrs[1][0] = load_r_wr;
    assign small_bufs_opr_write_addrs[1][1] = load_c_wr;

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND FEEDING
    ////////////////////////////////////////////////////////////////////////////////

    // 'comp_cycle' is zero on the first cycle that a column sits at the skew
    // register inputs; PE (i,j) sees its operands for step k at cycle k+i+j, so the
    // last product of a slice lands in the accumulator at 3*ARRAY_DIM-2
    assign comp_clear = rst | ~((state == S_FEED) | (state == S_DRAIN));
    assign comp_en    = (state == S_FEED) | (state == S_DRAIN);

    flop_enr #(.WIDTH(CYCLE_WIDTH), .RST(0)) comp_cycle_reg(.clk,
                                                            .rst(comp_clear),
                                                            .en(comp_en),
                                                            .d(comp_cycle + 1'b1),
                                                            .q(comp_cycle));

    assign feed_done  = (state == S_FEED)  & (comp_cycle == CYCLE_WIDTH'(ARRAY_DIM - 1));
    assign drain_done = (state == S_DRAIN) & (comp_cycle == CYCLE_WIDTH'(DRAIN_CYCLES));

    // the small buffer read is registered too, so the index leads the data it
    // produces by one cycle
    assign small_bufs_opr_read_rowcol_idx = (state == S_LEAD) ? '0
                                                              : SB_ADDR_WIDTH'(comp_cycle + 1'b1);

    assign read_imm_operands = (state == S_FEED);

    // held clear outside the slice pass so that nothing stale can reach the array;
    // must stay clear through the drain so in-flight operands keep propagating
    assign zero_skew_regs = ~((state == S_FEED) | (state == S_DRAIN));

    ////////////////////////////////////////////////////////////////////////////////
    // OUTPUT WRITEBACK
    ////////////////////////////////////////////////////////////////////////////////

    // runs concurrently with the next tile's computation; one command per tile,
    // since a tile is contiguous in the tile-major output block
    flop_r #(.WIDTH(2), .RST(0)) wb_fsm(.clk,
                                        .rst,
                                        .d(next_wb_state),
                                        .q(wb_state_bits));

    assign wb_state = wb_state_t'(wb_state_bits);

    assign wb_start = (state == S_SNAP);
    assign wb_busy  = (wb_state != S_WB_IDLE);

    always_comb begin
        case (wb_state)
            S_WB_IDLE:   next_wb_state = wb_start ? S_WB_CMD : S_WB_IDLE;
            S_WB_CMD:    next_wb_state = wb_cmd_accepted ? S_WB_STREAM : S_WB_CMD;
            S_WB_STREAM: next_wb_state = (wb_beat_accepted & wb_last_elem) ? S_WB_STS : S_WB_STREAM;
            S_WB_STS:    next_wb_state = s2mm_sts_tvalid ? S_WB_IDLE : S_WB_STS;
            default:     next_wb_state = S_WB_IDLE;
        endcase
    end

    assign s2mm_cmd_valid  = (wb_state == S_WB_CMD);
    assign s2mm_cmd_addr   = c_addr;
    assign s2mm_cmd_bytes  = BYTES_WIDTH'(TILE_BYTES);
    assign wb_cmd_accepted = s2mm_cmd_valid & s2mm_cmd_ready;

    assign next_c_addr = (state == S_IDLE) ? c_block_addr : (c_addr + ADDR_WIDTH'(TILE_BYTES));
    assign c_addr_en   = ((state == S_IDLE) & start_blk_comp) | wb_cmd_accepted;

    flop_en #(.WIDTH(ADDR_WIDTH)) c_addr_reg(.clk,
                                             .en(c_addr_en),
                                             .d(next_c_addr),
                                             .q(c_addr));

    // the output buffer read is registered and re-issued every cycle, so an address
    // has to be held for a cycle before its data is valid; phase 0 settles the
    // address and phase 1 presents the beat
    assign wb_clear = rst | (wb_state != S_WB_STREAM);

    flop_enr #(.WIDTH(1), .RST(0)) wb_phase_reg(.clk,
                                                .rst(wb_clear),
                                                .en(wb_state == S_WB_STREAM),
                                                .d(wb_phase ? ~s2mm_tready : 1'b1),
                                                .q(wb_phase));

    flop_enr #(.WIDTH(ELEM_IDX_WIDTH), .RST(0)) wb_idx_reg(.clk,
                                                           .rst(wb_clear),
                                                           .en(wb_beat_accepted),
                                                           .d(wb_idx + 1'b1),
                                                           .q(wb_idx));

    assign s2mm_tvalid      = (wb_state == S_WB_STREAM) & wb_phase;
    assign wb_beat_accepted = s2mm_tvalid & s2mm_tready;
    assign wb_last_elem     = (wb_idx == ELEM_IDX_WIDTH'(TILE_ELEMS - 1));
    assign s2mm_tlast       = s2mm_tvalid & wb_last_elem;

    assign small_bufs_out_read_addr[0] = wb_idx[ELEM_IDX_WIDTH-1:SB_ADDR_WIDTH];
    assign small_bufs_out_read_addr[1] = wb_idx[SB_ADDR_WIDTH-1:0];

    ////////////////////////////////////////////////////////////////////////////////
    // STATUS
    ////////////////////////////////////////////////////////////////////////////////

    assign mm2s_sts_tready = (state == S_A_STS) | (state == S_B_STS);
    assign s2mm_sts_tready = (wb_state == S_WB_STS);

    assign mm2s_sts_accepted = mm2s_sts_tvalid & mm2s_sts_tready;
    assign s2mm_sts_accepted = s2mm_sts_tvalid & s2mm_sts_tready;

    // a healthy DataMover status word reports OKAY with none of the error bits set
    assign sts_error = (mm2s_sts_accepted & (mm2s_sts_tdata[7:4] != 4'b1000))
                     | (s2mm_sts_accepted & (s2mm_sts_tdata[7:4] != 4'b1000));

    assign error_clear = rst | ((state == S_IDLE) & start_blk_comp);

    flop_enr #(.WIDTH(1), .RST(0)) error_reg(.clk,
                                             .rst(error_clear),
                                             .en(sts_error),
                                             .d(1'b1),
                                             .q(error_blk_comp));

endmodule
