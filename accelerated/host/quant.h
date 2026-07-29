/**
 * quant.h - Symmetric linear quantization between float32 and the systolic
 *           array's int16 operands / int48 accumulators
 *
 * A float matrix is represented as `X[i][j] ~= scale * X_q[i][j]`, where
 * `X_q` is int16. Scales are chosen per-row for the first operand and
 * per-column for the second, which is the coarsest granularity that still
 * factors out of a dot product:
 *
 *     C[i][j] =  sum_k A[i][k] * B[k][j]
 *             ~= sum_k (s_A[i] * A_q[i][k]) * (s_B[j] * B_q[k][j])
 *             =  s_A[i] * s_B[j] * sum_k (A_q[i][k] * B_q[k][j])
 *                                 \_________________________/
 *                                  exactly the integer value the
 *                                  array accumulates into int48
 *
 * Because the scales do not depend on `k`, the integer accumulators of
 * separate `k` tiles share a scale and can be summed directly, and the
 * accumulation itself carries no rounding error.
 */

#ifndef _QUANT_H_
#define _QUANT_H_

#include <stdint.h>

// largest magnitude an operand quantizes to (-32768 is left unused to keep
// the representable range symmetric)
#define QUANT_MAX 32767

/**
 * @brief Sign-extends a 48-bit accumulator read out of a 64-bit memory slot
 *
 * The array's accumulators are `ACC_WIDTH` (48) bits wide, so the upper bits
 * of the word the kernel writes back are not meaningful.
 *
 * @param[in] raw Raw 64-bit word containing the accumulator in bits 47:0
 * @return        The accumulator's signed value
 */
static inline int64_t sign_extend_48(uint64_t raw) {
    const uint64_t sign_bit = (uint64_t)1 << 47;
    const uint64_t mask = ((uint64_t)1 << 48) - 1;

    raw &= mask;

    return (int64_t)((raw ^ sign_bit) - sign_bit);
}

/**
 * @brief Quantizes `A` to int16 using one scale per row
 *
 * - `A_q` has the same dimensions and layout as `A`
 *
 * - use for the first operand of a matmul, whose row index survives into `C`
 *
 * @param[in]  A      Input matrix
 * @param[in]  A_m    Number of rows in `A`
 * @param[in]  A_n    Number of columns in `A`
 * @param[out] A_q    Quantized matrix
 * @param[out] scales Per-row scales (`A_m` elements)
 */
void quantize_rows(float *A,
                   int A_m, int A_n,
                   int16_t *A_q, float *scales);

/**
 * @brief Quantizes `B` to int16 using one scale per column
 *
 * - `B_q` has the same dimensions and layout as `B`
 *
 * - use for the second operand of a matmul, whose column index survives
 *   into `C`
 *
 * @param[in]  B      Input matrix
 * @param[in]  B_m    Number of rows in `B`
 * @param[in]  B_n    Number of columns in `B`
 * @param[out] B_q    Quantized matrix
 * @param[out] scales Per-column scales (`B_n` elements)
 */
void quantize_cols(float *B,
                   int B_m, int B_n,
                   int16_t *B_q, float *scales);

/**
 * @brief Converts a block of int48 accumulators back to float
 *
 * - `acc` and `C` may have different row strides, so this works equally on a
 *   whole matrix and on one tile addressed within a larger matrix
 *
 * - `row_scales` and `col_scales` are indexed tile-locally, so pass pointers
 *   offset to the block's first row and column
 *
 * @param[in]  acc        Accumulators to convert
 * @param[in]  acc_stride Number of elements per row of `acc`
 * @param[in]  m          Number of rows in the block
 * @param[in]  n          Number of columns in the block
 * @param[in]  row_scales Scales of the first operand's rows (`m` elements)
 * @param[in]  col_scales Scales of the second operand's columns (`n` elements)
 * @param[out] C          Output block
 * @param[in]  C_stride   Number of elements per row of `C`
 */
void dequantize_block(int64_t *acc, int acc_stride,
                      int m, int n,
                      float *row_scales, float *col_scales,
                      float *C, int C_stride);

#endif
