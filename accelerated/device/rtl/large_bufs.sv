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
//   row address. Vivado's elaborator rejects any single variable wider than
//   1,000,000 bits, and one undivided region of 256x256 16-bit elements is
//   1,048,576 bits; banking also keeps each inferred block RAM a few deep instead
//   of one 32-deep cascade
// - Banking is invisible from the outside: every bank reads every cycle and the
//   delayed bank select picks one, so a read still costs exactly one cycle and the
//   contents are addressed exactly as if the region were undivided
// - BUFFER_DIM and NUM_BANKS must both be powers of two, and NUM_BANKS at least 2
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

    localparam ADDR_WIDTH = $clog2(BUFFER_DIM);       // 8
    localparam SEL_WIDTH  = $clog2(NUM_BANKS);        // 3
    localparam BANK_DEPTH = BUFFER_DIM / NUM_BANKS;   // 32 rows per bank

    logic [ELEM_SIZE-1:0] bank_read_data [0:1][0:NUM_BANKS-1];
    logic [SEL_WIDTH-1:0] read_bank_sel  [0:1];

    ////////////////////////////////////////////////////////////////////////////////
    // BANKS
    ////////////////////////////////////////////////////////////////////////////////

    generate
        // bank b holds the rows whose low address bits equal b
        for (genvar b = 0; b < NUM_BANKS; b++) begin : banks
            (* ram_style = "block" *) logic [ELEM_SIZE-1:0] block_A [0:BANK_DEPTH-1][0:BUFFER_DIM-1];
            (* ram_style = "block" *) logic [ELEM_SIZE-1:0] block_B [0:BANK_DEPTH-1][0:BUFFER_DIM-1];

            logic [ELEM_SIZE-1:0] read_A, read_B;

            always_ff @(posedge clk) begin
                if (write_en[0] & (write_addrs[0][0][SEL_WIDTH-1:0] == SEL_WIDTH'(b)))
                    block_A[write_addrs[0][0][ADDR_WIDTH-1:SEL_WIDTH]][write_addrs[0][1]] <= write_data[0];

                if (write_en[1] & (write_addrs[1][0][SEL_WIDTH-1:0] == SEL_WIDTH'(b)))
                    block_B[write_addrs[1][0][ADDR_WIDTH-1:SEL_WIDTH]][write_addrs[1][1]] <= write_data[1];

                read_A <= block_A[read_addrs[0][0][ADDR_WIDTH-1:SEL_WIDTH]][read_addrs[0][1]];
                read_B <= block_B[read_addrs[1][0][ADDR_WIDTH-1:SEL_WIDTH]][read_addrs[1][1]];
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
                                        .q(read_bank_sel[0]));

    flop #(.WIDTH(SEL_WIDTH)) sel_B_reg(.clk,
                                        .d(read_addrs[1][0][SEL_WIDTH-1:0]),
                                        .q(read_bank_sel[1]));

    assign read_data[0] = bank_read_data[0][read_bank_sel[0]];
    assign read_data[1] = bank_read_data[1][read_bank_sel[1]];

endmodule
