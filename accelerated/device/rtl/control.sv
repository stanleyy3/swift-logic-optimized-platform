// control.sv -- Control

module control #(
    parameter ARRAY_DIM = 4
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         start,
    output logic                         done,
    output logic                         read_operands,
    output logic [$clog2(ARRAY_DIM)-1:0] read_operands_rowcol_num,
    output logic                         zero_data
);

    typedef enum logic { IDLE, ACTIVE } status_t;

    status_t status, next_status;

    // one matmul has latency of 3N-2 (where N is dimension of matmul)
    logic [$clog2(ARRAY_DIM)+1:0] comp_cycle, next_comp_cycle;
    logic                         zero_cycle;

    ////////////////////////////////////////////////////////////////////////////////
    // FSM
    ////////////////////////////////////////////////////////////////////////////////

    flop_r #(.WIDTH(1), .RST(0)) fsm(.clk,
                                     .rst,
                                     .d(next_status),
                                     .q(status));

    always_comb begin
        if (start) next_status = ACTIVE;
        else if (done) next_status = IDLE;
        else next_status = status;
    end

    ////////////////////////////////////////////////////////////////////////////////
    // COMPUTATION LIFECYCLE
    ////////////////////////////////////////////////////////////////////////////////

    // whether set cycle to 0
    assign zero_cycle = ((status == IDLE) | rst);

    flop_r #(.WIDTH($clog2(ARRAY_DIM)+2), .RST(0)) comp_cycle_reg(.clk,
                                                                  .rst(zero_cycle),
                                                                  .d(next_comp_cycle),
                                                                  .q(comp_cycle));

    assign next_comp_cycle = comp_cycle + 1;

    assign done = (comp_cycle == (3 * ARRAY_DIM - 2));

    ////////////////////////////////////////////////////////////////////////////////
    // OPERAND FEEDING
    ////////////////////////////////////////////////////////////////////////////////

    assign read_operands = ((status == ACTIVE) & (comp_cycle < ARRAY_DIM));
    assign read_operands_rowcol_num = comp_cycle[$clog2(ARRAY_DIM)-1:0];

    ////////////////////////////////////////////////////////////////////////////////

    assign zero_data = (start | rst);

endmodule
