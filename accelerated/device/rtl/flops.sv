// flops.sv -- Various flip-flops

module flop #(
    parameter WIDTH = 8
) ( 
    input  logic             clk,
    input  logic [WIDTH-1:0] d, 
    output logic [WIDTH-1:0] q
);

    always_ff @(posedge clk)
        q <= d;

endmodule

// flip flop with enable
module flop_en #(
    parameter WIDTH = 8
) (
    input  logic             clk, en,
    input  logic [WIDTH-1:0] d, 
    output logic [WIDTH-1:0] q
);

    always_ff @(posedge clk)
        if (en) q <= d;

endmodule

// flip flop with enable and reset
module flop_enr #(
    parameter WIDTH = 8, RST=0
) (
    input  logic             clk, rst, en,
    input  logic [WIDTH-1:0] d, 
    output logic [WIDTH-1:0] q
);

    always_ff @(posedge clk)
        if (rst)     q <= RST;
        else if (en) q <= d;

endmodule

// flip flop with reset
module flop_r #(
    parameter WIDTH = 8, RST=0
) ( 
    input  logic             clk, rst,
    input  logic [WIDTH-1:0] d, 
    output logic [WIDTH-1:0] q
);

    always_ff @(posedge clk)
        if (rst) q <= RST;
        else     q <= d;

endmodule
