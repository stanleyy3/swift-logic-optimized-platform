/**
 * fpga.h - Device lifecycle and block launches for the FPGA matmul accelerator
 *
 * The array computes one block matmul per launch: C := A @ B, where A is
 * `blk_m` x `blk_k` and B is `blk_k` x `blk_n`, each a multiple of
 * FPGA_ARRAY_DIM and no greater than FPGA_LARGE_BUFFER_DIM. The array tiles the
 * block itself and accumulates the block's whole K dimension internally, so a
 * caller only has to schedule blocks (see control.sv's header comment).
 *
 * Everything device-specific lives behind this interface, so callers hold no
 * XRT handles and the same code runs against the software model (FPGA_MODEL).
 */

#ifndef _FPGA_H_
#define _FPGA_H_

#include "quant.h"

// `load_block_en` bits: the array keeps an operand block resident in its large
// buffers across launches, so a run that reuses the previous A block only has
// to reload B. Reuse is only legal while `blk_m` and `blk_k` are unchanged.
#define FPGA_LOAD_A 0x1
#define FPGA_LOAD_B 0x2

/**
 * @brief Opens the device, loads the xclbin, opens the matmul kernel, and
 *        allocates the persistent operand/result block buffers
 *
 * Call once at program startup, before any call to tiled_mat_mat_mul_fpga
 */
void fpga_init(void);

/**
 * @brief Frees the block buffers and closes the kernel/device
 *
 * Call once at program shutdown
 */
void fpga_cleanup(void);

/**
 * @brief The staging buffer for the first operand's block
 *
 * FPGA_BLOCK_DIM x FPGA_BLOCK_DIM elements; a launch reads the leading
 * `blk_m` x `blk_k` of it, row-major with a row stride of `blk_k`
 *
 * @return Pointer to the buffer
 */
f16_t *fpga_block_a(void);

/**
 * @brief The staging buffer for the second operand's block
 *
 * Read as `blk_k` x `blk_n` row-major with a row stride of `blk_n`
 *
 * @return Pointer to the buffer
 */
f16_t *fpga_block_b(void);

/**
 * @brief The buffer a launch writes its result block into
 *
 * Laid out tile-major: tile `(ti,tj)` at element offset
 * `(ti*tiles_j + tj) * FPGA_ARRAY_DIM**2`, each tile row-major. Use
 * accum_dequantize_block() to read it.
 *
 * @return Pointer to the buffer
 */
const f16_t *fpga_block_c(void);

/**
 * @brief Runs one block matmul and waits for it to finish
 *
 * Pushes whichever operand blocks `load_block_en` selects, launches, waits,
 * checks the device's error flag, and brings the result back. On a device error
 * this reports and exits rather than letting a training run continue on corrupt
 * data.
 *
 * - all three dimensions must be multiples of FPGA_ARRAY_DIM and no greater
 *   than FPGA_BLOCK_DIM; control.sv does not validate them and silently drops
 *   the leftover rows/columns of anything ragged
 *
 * @param[in] blk_m         Rows in the first operand's block
 * @param[in] blk_k         Columns in the first operand's block
 * @param[in] blk_n         Columns in the second operand's block
 * @param[in] load_block_en Which operand blocks to push (FPGA_LOAD_A/_B)
 */
void fpga_launch_block(int blk_m, int blk_k, int blk_n, int load_block_en);

#endif
