/**
 * quant.h - Conversion between float32 and the systolic array's float16
 *           operands / fixed-point accumulators
 *
 * A float matrix is represented as `X[i][j] ~= scale * X_h[i][j]`, where `X_h`
 * is float16 and `scale` is a power of two. Scales are chosen per-row for the
 * first operand and per-column for the second, which is the coarsest
 * granularity that still factors out of a dot product:
 *
 *     C[i][j] =  sum_k A[i][k] * B[k][j]
 *             ~= sum_k (s_A[i] * A_h[i][k]) * (s_B[j] * B_h[k][j])
 *             =  s_A[i] * s_B[j] * sum_k (A_h[i][k] * B_h[k][j])
 *                                 \_________________________/
 *                                  the fixed-point value the array
 *                                  accumulates, in units of 2**ACC_LSB
 *
 * Because the scales do not depend on `k`, the accumulators of separate `k`
 * tiles share a scale and can be summed directly as integers.
 *
 * float16 has the same 11 significant bits throughout its range, so unlike the
 * int16 scheme the scale buys no precision - it buys dynamic range. A float16
 * normal only spans 2**-14 to 65504, and the MAC flushes subnormals to zero
 * and misreads Inf/NaN as huge normals, so each row/column is first rescaled
 * so its maximum lands in [0.5, 1). Every operand is then <= 1, every product
 * is <= 1, and the accumulator cannot wrap. A power-of-two scale only shifts
 * exponents, so both the rescale and the dequantize multiply are exact.
 */

#ifndef _QUANT_H_
#define _QUANT_H_

#include <stdint.h>

// a raw IEEE-754 binary16 bit pattern (1 sign, 5 exponent, 10 fraction). The
// host only converts and ships these, so a bit pattern avoids depending on
// compiler `_Float16` support
typedef uint16_t f16_t;

// PE accumulator width and the binary weight of its LSB (must match ACC_WIDTH
// and ACC_LSB in systolic_array.sv); an accumulator value `v` represents
// `v * 2**ACC_LSB`. The device applies this weight itself on the way out, so
// nothing on the normal path needs these - they are here for the software
// model of the device in device_model.c
#define ACC_WIDTH 48
#define ACC_LSB   (-24)

// binary16 field geometry: unbiased exponents of the largest finite (65504)
// and smallest normal (2**-14) values, and the bit pattern of 65504
#define F16_FRAC_BITS 10
#define F16_BIAS      15
#define F16_MAX_EXP   15
#define F16_MIN_EXP   (-14)
#define F16_MAX_BITS  0x7BFFu

// binary32 field geometry, plus the fraction bits it has beyond binary16 and
// the weight of the most significant one dropped when rounding (half a
// binary16 ULP, so exactly the round-to-nearest-even tie point)
#define F32_FRAC_BITS        23
#define F32_BIAS             127
#define F32_EXCESS_FRAC_BITS (F32_FRAC_BITS - F16_FRAC_BITS)  // 13
#define F32_ROUND_HALF       (1u << (F32_EXCESS_FRAC_BITS - 1))

/**
 * @brief Converts a float32 to the float16 bit pattern the array consumes
 *
 * Rounds to nearest, ties to even. Magnitudes that do not reach the smallest
 * float16 normal become an exact zero (the MAC flushes subnormals anyway), and
 * those past the largest finite float16 - including Inf and NaN - saturate
 * rather than becoming Inf, which the MAC would decode as a huge normal.
 * Scaling should keep values away from both bounds; they are backstops.
 *
 * @param[in] x Value to convert
 * @return      Equivalent float16 bit pattern
 */
static inline f16_t f32_to_f16(float x) {
    // punning through a union is well-defined, unlike a pointer cast
    union { float f; uint32_t u; } bits = { .f = x };

    uint32_t exp_field = (bits.u >> F32_FRAC_BITS) & 0xFFu;
    uint32_t frac = bits.u & ((1u << F32_FRAC_BITS) - 1);

    // the sign is the only field that transfers unchanged
    f16_t sign_bit = (f16_t)((bits.u >> 31) << 15);

    // Inf/NaN saturate like any other out-of-range magnitude
    if (exp_field == 0xFFu) {
        return sign_bit | F16_MAX_BITS;
    }

    // a float32 zero or subnormal has exp_field == 0, landing at -127
    int exp = (int)exp_field - F32_BIAS;

    // round the 23-bit fraction down to 10 bits: round up when the dropped
    // bits exceed half a ULP, or tie and the retained fraction is odd
    uint32_t f16_frac = frac >> F32_EXCESS_FRAC_BITS;
    uint32_t dropped = frac & ((1u << F32_EXCESS_FRAC_BITS) - 1);

    if (dropped > F32_ROUND_HALF ||
        (dropped == F32_ROUND_HALF && (f16_frac & 1u))) {
        f16_frac++;

        // carrying out of the fraction means the mantissa rounded up to the
        // next power of two
        if (f16_frac == (1u << F16_FRAC_BITS)) {
            f16_frac = 0;
            exp++;
        }
    }

    // Both bounds are tested after rounding, since rounding can carry a value
    // across either one: just under 2**-14 up into the smallest normal, and
    // just under 65536 up past the largest finite value.
    if (exp < F16_MIN_EXP) {
        return sign_bit;
    }

    if (exp > F16_MAX_EXP) {
        return sign_bit | F16_MAX_BITS;
    }

    return (f16_t)(sign_bit
                   | ((uint32_t)(exp + F16_BIAS) << F16_FRAC_BITS)
                   | f16_frac);
}

/**
 * @brief Converts a float16 bit pattern back to float32
 *
 * Mirrors how the MAC decodes an operand rather than IEEE-754, so it can model
 * the array on the host: a zero exponent field reads as zero, and an all-ones
 * exponent field as a large normal. Never rounds, since every float16 normal
 * is exactly representable in float32.
 *
 * @param[in] h Value to convert
 * @return      Equivalent float32 value
 */
static inline float f16_to_f32(f16_t h) {
    uint32_t sign = (uint32_t)h >> 15;
    uint32_t exp_field = ((uint32_t)h >> F16_FRAC_BITS) & 0x1Fu;
    uint32_t frac = (uint32_t)h & ((1u << F16_FRAC_BITS) - 1);

    union { float f; uint32_t u; } bits;

    // zero and subnormals alike are zero to the MAC
    if (exp_field == 0) {
        bits.u = sign << 31;

        return bits.f;
    }

    // re-bias the exponent and left-align the fraction in float32's field
    bits.u = (sign << 31)
             | ((exp_field - F16_BIAS + F32_BIAS) << F32_FRAC_BITS)
             | (frac << F32_EXCESS_FRAC_BITS);

    return bits.f;
}

/**
 * @brief Converts `A` to float16 using one power-of-two scale per row
 *
 * - `A_h` has the same dimensions and layout as `A`
 *
 * - use for the first operand of a matmul, whose row index survives into `C`
 *
 * @param[in]  A      Input matrix
 * @param[in]  A_m    Number of rows in `A`
 * @param[in]  A_n    Number of columns in `A`
 * @param[out] A_h    Converted matrix
 * @param[out] scales Per-row scales (`A_m` elements)
 */
void quantize_rows(float *A,
                   int A_m, int A_n,
                   f16_t *A_h, float *scales);

/**
 * @brief Converts `B` to float16 using one power-of-two scale per column
 *
 * - `B_h` has the same dimensions and layout as `B`
 *
 * - use for the second operand of a matmul, whose column index survives
 *   into `C`
 *
 * @param[in]  B      Input matrix
 * @param[in]  B_m    Number of rows in `B`
 * @param[in]  B_n    Number of columns in `B`
 * @param[out] B_h    Converted matrix
 * @param[out] scales Per-column scales (`B_n` elements)
 */
void quantize_cols(float *B,
                   int B_m, int B_n,
                   f16_t *B_h, float *scales);

/**
 * @brief De-tiles one block of the array's float16 output, dequantizes it, and
 *        accumulates it into `C`
 *
 * The array writes a block back tile-major - tile `(ti,tj)` at element offset
 * `(ti*tiles_j + tj) * array_dim**2`, each tile row-major - so this scatters
 * into row-major `C` as it goes.
 *
 * Accumulates rather than assigns: a matmul whose K exceeds the block dimension
 * arrives as several block results, and since neither scale depends on `k` they
 * are summed here. Callers must zero `C` before the first block.
 *
 * - `m` and `n` are the block's *valid* extent, which may be less than the
 *   padded `blk_m`/`blk_n` the device was launched with; the padding rows and
 *   columns are simply not read
 *
 * - `row_scales` and `col_scales` are indexed block-locally, so pass pointers
 *   offset to the block's first row and column
 *
 * @param[in]  tiles      Block of float16 results as the device laid them out
 * @param[in]  blk_n      Padded column count the device was launched with,
 *                        which sets the tile-major stride
 * @param[in]  array_dim  Device tile dimension (`FPGA_ARRAY_DIM`)
 * @param[in]  m          Number of valid rows in the block
 * @param[in]  n          Number of valid columns in the block
 * @param[in]  row_scales Scales of the first operand's rows (`m` elements)
 * @param[in]  col_scales Scales of the second operand's columns (`n` elements)
 * @param[in,out] C       Output block, accumulated into
 * @param[in]  C_stride   Number of elements per row of `C`
 */
void accum_dequantize_block(const f16_t *tiles, int blk_n, int array_dim,
                            int m, int n,
                            const float *row_scales, const float *col_scales,
                            float *C, int C_stride);

#endif
