/**
 * config.h - Configuration for program
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_

// whether to randomize seed
#define RAND_SEED_RAND false

// dataset to train on (0 - MNIST, 1 - Fashion MNIST)
#define DATASET 0

// number of milliseconds per update of training status
#define TRAIN_UPDATE_FREQ 100

// matrix multiplication tile dimension (CPU path only)
#define TILE_DIM 64

// whether the matmuls in a training pass are offloaded to the FPGA
#ifndef USE_FPGA_MATMUL
#define USE_FPGA_MATMUL 0
#endif

// whether the FPGA path runs against the bit-exact software model of the device
// in device_model.c instead of real hardware, so the host can be exercised
// without a board. `make MODEL=1` sets this; the test target always does
#ifndef FPGA_MODEL
#define FPGA_MODEL 0
#endif

// Must match the parameters systolic_array.sv is elaborated with; the device
// has no way to report a mismatch. See the "Configuration for fp16" note at the
// top of systolic_array.sv.
#define FPGA_ARRAY_DIM        16
#define FPGA_LARGE_BUFFER_DIM 256

// Block dimension the host schedules with: no greater than
// FPGA_LARGE_BUFFER_DIM and a multiple of FPGA_ARRAY_DIM. control.sv derives
// its tile counts with a plain `>> $clog2(ARRAY_DIM)` and never validates its
// inputs, so a launch dimension that is not a multiple of FPGA_ARRAY_DIM is
// silently truncated and its leftover rows/columns are dropped from the result.
#define FPGA_BLOCK_DIM 256

// path to the compiled FPGA bitstream and the kernel name within it
#define FPGA_XCLBIN_PATH "matmul.xclbin"
#define FPGA_KERNEL_NAME "matmul_krnl"

// AXI4-Lite offset of the kernel's error register, which carries
// error_blk_comp. Must match the packaged kernel's register map; see
// accelerated/device/CONTROL_INTERFACE.md
#define FPGA_ERROR_REG_OFFSET 0x54

// terminal output color escape sequences
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RED    "\x1b[31m"
#define ANSI_COLOR_RESET  "\x1b[0m"

#endif
