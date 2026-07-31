// large_bufs.sv -- Large buffers (overall operands)
//
// - Blocks intended to be tiled into tiles as the immediate operands for PE
//   array
// - Each block has an independent write enable so that one operand block can be
//   filled while the other stays resident
// - A block occupies the leading rows/columns of its region; the row stride is
//   always BUFFER_DIM, so every address is a plain concatenation of a row and a
//   column counter
// - Each region is split across NUM_BANKS banks selected by the low bits of the
//   row address, which keeps each inferred block RAM a few deep instead of one
//   32-deep cascade
// - Banking is invisible from the outside: every bank reads every cycle and the
//   delayed bank select picks one, so a read still costs exactly one cycle and the
//   contents are addressed exactly as if the region were undivided
// - BUFFER_DIM and NUM_BANKS must both be powers of two, and NUM_BANKS at least 2
//
// - A bank is a flat one-dimensional array, and has to stay that way: Vivado's
//   block RAM inference tops out at two dimensions (AR 53507), so declaring a bank
//   as [0:rows-1][0:cols-1] makes it a '3D RAM' that the inferencer refuses
//   (Synth 8-7186) and dissolves into flip-flops (Synth 8-11357) -- a megabit of
//   them per region, which is enough to run synthesis out of disk
// - One memory per always_ff block for the same reason; a single process driving
//   both banks is another pattern that falls back to registers
//
// - If the read ports are ever widened to hand a whole row/column to the array in
//   one cycle, operand A wants banking by row (as here) but operand B wants
//   banking by column instead

module large_bufs #(
    parameter BUFFER_DIM, // dimension of 1 of two regions (for each operand block)
    parameter ELEM_SIZE,
    parameter NUM_BANKS
) (
    input  logic                          clk,
    input  logic                          write_en    [0:1],
    input  logic [$clog2(BUFFER_DIM)-1:0] write_addrs [0:1][0:1],
    input  logic [ELEM_SIZE-1:0]          write_data  [0:1],
    input  logic [$clog2(BUFFER_DIM)-1:0] read_addrs  [0:1][0:1],
    output logic [ELEM_SIZE-1:0]          read_data   [0:1]
);

    localparam ADDR_WIDTH = $clog2(BUFFER_DIM);                    // 8
    localparam SEL_WIDTH  = $clog2(NUM_BANKS);                     // 2
    localparam BANK_ROWS  = BUFFER_DIM / NUM_BANKS;                // 64 rows per bank
    localparam BANK_DEPTH = BANK_ROWS * BUFFER_DIM;                // 16384 elements
    localparam BANK_WIDTH = $clog2(BANK_DEPTH);                    // 14

    logic [BANK_WIDTH-1:0] bank_write_addrs [0:1];
    logic [SEL_WIDTH-1:0]  write_bank_sels  [0:1];
    logic [BANK_WIDTH-1:0] bank_read_addrs  [0:1];
    logic [ELEM_SIZE-1:0]  bank_read_data   [0:1][0:NUM_BANKS-1];
    logic [SEL_WIDTH-1:0]  read_bank_sels   [0:1];

    ////////////////////////////////////////////////////////////////////////////////
    // BANK ADDRESSING
    ////////////////////////////////////////////////////////////////////////////////

    // the row's bank-local part concatenated with the column, which is the same
    // plain concatenation the region-wide address already was, minus the bank select
    // that the row's low bits carry
    generate
        for (genvar o = 0; o < 2; o++) begin : addrs
            assign bank_write_addrs[o] = {write_addrs[o][0][ADDR_WIDTH-1:SEL_WIDTH],
                                          write_addrs[o][1]};
            assign write_bank_sels[o]  = write_addrs[o][0][SEL_WIDTH-1:0];
            assign bank_read_addrs[o]  = {read_addrs[o][0][ADDR_WIDTH-1:SEL_WIDTH],
                                          read_addrs[o][1]};
        end
    endgenerate

    ////////////////////////////////////////////////////////////////////////////////
    // BANKS
    ////////////////////////////////////////////////////////////////////////////////

    generate
        // bank b holds the rows whose low address bits equal b
        for (genvar b = 0; b < NUM_BANKS; b++) begin : banks
            (* ram_style = "block" *) logic [ELEM_SIZE-1:0] block_A [0:BANK_DEPTH-1];
            (* ram_style = "block" *) logic [ELEM_SIZE-1:0] block_B [0:BANK_DEPTH-1];

            logic [ELEM_SIZE-1:0] read_A, read_B;

            always_ff @(posedge clk) begin
                if (write_en[0] & (write_bank_sels[0] == SEL_WIDTH'(b)))
                    block_A[bank_write_addrs[0]] <= write_data[0];

                read_A <= block_A[bank_read_addrs[0]];
            end

            always_ff @(posedge clk) begin
                if (write_en[1] & (write_bank_sels[1] == SEL_WIDTH'(b)))
                    block_B[bank_write_addrs[1]] <= write_data[1];

                read_B <= block_B[bank_read_addrs[1]];
            end

            assign bank_read_data[0][b] = read_A;
            assign bank_read_data[1][b] = read_B;
        end
    endgenerate

    ////////////////////////////////////////////////////////////////////////////////
    // OUTPUT SELECTION
    ////////////////////////////////////////////////////////////////////////////////

    // the banks register their reads, so the select is delayed by the same cycle for
    // the output mux to land on the bank the address actually asked for
    flop #(.WIDTH(SEL_WIDTH)) sel_A_reg(.clk,
                                        .d(read_addrs[0][0][SEL_WIDTH-1:0]),
                                        .q(read_bank_sels[0]));

    flop #(.WIDTH(SEL_WIDTH)) sel_B_reg(.clk,
                                        .d(read_addrs[1][0][SEL_WIDTH-1:0]),
                                        .q(read_bank_sels[1]));

    assign read_data[0] = bank_read_data[0][read_bank_sels[0]];
    assign read_data[1] = bank_read_data[1][read_bank_sels[1]];

endmodule