/**
 * device_model.h - Bit-exact software model of the systolic array's block matmul
 *
 * Mirrors mac.sv, fix_to_float.sv and the block schedule in control.sv, so a
 * block computed here is identical to a block computed on the device, down to
 * the last bit. Two uses:
 *
 * - with FPGA_MODEL set, fpga_launch_block() runs this instead of touching XRT,
 *   so the host's blocking, padding and de-tiling can be tested without a board
 *
 * - as the oracle a test compares real hardware against, since the array's
 *   accumulator is fixed-point and a float32 reference would not match
 */

#ifndef _DEVICE_MODEL_H_
#define _DEVICE_MODEL_H_

#include "quant.h"

/**
 * @brief Computes one block matmul exactly as the device would
 *
 * Reproduces the contract in control.sv's header comment:
 *
 * - `blk_m`, `blk_k` and `blk_n` must all be multiples of `FPGA_ARRAY_DIM`
 *
 * - `a_block` is `blk_m` x `blk_k` row-major, `b_block` is `blk_k` x `blk_n`
 *   row-major
 *
 * - `c_block` comes out tile-major: tile `(ti,tj)` at element offset
 *   `(ti*tiles_j + tj) * FPGA_ARRAY_DIM**2`, each tile row-major
 *
 * - the accumulator is zeroed once per output tile, so a whole block's K is
 *   accumulated in fixed point before being converted to float16
 *
 * @param[in]  a_block First operand block
 * @param[in]  b_block Second operand block
 * @param[in]  blk_m   Rows in `a_block`
 * @param[in]  blk_k   Columns in `a_block` and rows in `b_block`
 * @param[in]  blk_n   Columns in `b_block`
 * @param[out] c_block Output block, tile-major
 */
void device_model_block_matmul(const f16_t *a_block, const f16_t *b_block,
                               int blk_m, int blk_k, int blk_n,
                               f16_t *c_block);

#endif
