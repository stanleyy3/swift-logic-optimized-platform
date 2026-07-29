// systolic_array_tb.sv -- Systolic array testbench

module systolic_array_tb;

    localparam ARRAY_DIM = 4;
    localparam MUL_WIDTH = 16;
    localparam ACC_WIDTH = 48;

    logic [MUL_WIDTH-1:0] operands_flat        [0:2*ARRAY_DIM*ARRAY_DIM-1];
    logic [ACC_WIDTH-1:0] acc_values_spec_flat [0:ARRAY_DIM*ARRAY_DIM-1];

    logic                 clk = 0;
    logic                 rst;
    logic                 load;
    logic [MUL_WIDTH-1:0] new_A_values [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic [MUL_WIDTH-1:0] new_B_values [0:ARRAY_DIM-1][0:ARRAY_DIM-1];
    logic                 start;
    logic                 done;
    logic [ACC_WIDTH-1:0] acc_values   [0:ARRAY_DIM-1][0:ARRAY_DIM-1];

    logic [ACC_WIDTH-1:0] acc_values_spec [0:ARRAY_DIM-1][0:ARRAY_DIM-1];

    systolic_array dut(.*);

    always #5 clk = ~clk;

    initial begin
        $readmemh("operands.txt", operands_flat);
        $readmemh("output_spec.txt", acc_values_spec_flat);

        // unflatten A operand
        for (int i = 0; i < ARRAY_DIM; i++) begin
            for (int j = 0; j < ARRAY_DIM; j++) begin
                new_A_values[i][j] = operands_flat[i*ARRAY_DIM+j];
            end
        end

        // unflatten B operand
        for (int i = 0; i < ARRAY_DIM; i++) begin
            for (int j = 0; j < ARRAY_DIM; j++) begin
                new_B_values[i][j] = operands_flat[(ARRAY_DIM+i)*ARRAY_DIM+j];
            end
        end

        // unflatten output
        for (int i = 0; i < ARRAY_DIM; i++) begin
            for (int j = 0; j < ARRAY_DIM; j++) begin
                acc_values_spec[i][j] = acc_values_spec_flat[i*ARRAY_DIM+j];
            end
        end

        load = 0;
        start = 0;

        rst = 1;

        @(posedge clk);  // reset control state
        #1;

        rst = 0;
        
        load = 1;
        start = 1;

        @(posedge clk);  // loads in operand matrices and starts computation
        #1;

        load = 0;
        start = 0;

        wait(done === 1);

        if (acc_values !== acc_values_spec) begin
            $display("Test case failed!\n");

            // actual output matrix values
            $display("Actual:");
            for (int i = 0; i < ARRAY_DIM; i++) begin
                for (int j = 0; j < ARRAY_DIM; j++) begin
                    $write("%d ", acc_values[i][j]);
                end
                $display("");
            end

            $display("");

            // expected output matrix values
            $display("Expected:");
            for (int i = 0; i < ARRAY_DIM; i++) begin
                for (int j = 0; j < ARRAY_DIM; j++) begin
                    $write("%d ", acc_values_spec[i][j]);
                end
                $display("");
            end

            $display("");

            $stop;
        end

        $display("Test case passed!");
        $stop;

    end

endmodule
