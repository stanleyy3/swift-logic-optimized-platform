/**
 * quant.c - Conversion between float32 and the systolic array's float16
 *           operands / fixed-point accumulators
 */

#include "quant.h"

#include <stdlib.h>
#include <math.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// bounds on the scale exponent, so that both 2**exp and 2**-exp stay normal
// float32s and the multiplies below are exact. Only reachable for rows whose
// maximum is itself near float32's limits, where the clamp just leaves the
// normalized values further from [0.5, 1) than usual
#define SCALE_EXP_MIN (-126)
#define SCALE_EXP_MAX 127

/**
 * @brief Picks the exponent of the power-of-two scale for a row or column
 *
 * @param[in] max_abs Largest magnitude in the row/column
 * @return            `exp` such that `max_abs / 2**exp` lies in [0.5, 1)
 */
static inline int scale_exp(float max_abs) {
    int exp;

    // frexpf splits max_abs into a mantissa in [0.5, 1) and this exponent; an
    // all-zero row reports exp == 0, which is a fine scale for zeros
    frexpf(max_abs, &exp);

    return MIN(MAX(exp, SCALE_EXP_MIN), SCALE_EXP_MAX);
}

void quantize_rows(float *A,
                   int A_m, int A_n,
                   f16_t *A_h, float *scales) {
    // loop across A's rows
    for (int i = 0; i < A_m; i++) {
        // find the row's largest magnitude
        float max_abs = 0.f;

        for (int j = 0; j < A_n; j++) {
            max_abs = MAX(max_abs, fabsf(A[i * A_n + j]));
        }

        int exp = scale_exp(max_abs);

        // 2**exp and 2**-exp are both exact, so scaling only shifts exponents
        scales[i] = ldexpf(1.f, exp);

        float inv_scale = ldexpf(1.f, -exp);

        for (int j = 0; j < A_n; j++) {
            A_h[i * A_n + j] = f32_to_f16(A[i * A_n + j] * inv_scale);
        }
    }
}

void quantize_cols(float *B,
                   int B_m, int B_n,
                   f16_t *B_h, float *scales) {
    // Both sweeps below run in row-major order, matching B's layout. Walking
    // B a column at a time would be the obvious way to reduce per column, but
    // strides of B_n across a matrix larger than cache costs ~3.5x here.
    float *col = calloc(B_n, sizeof(float));

    // accumulate every column's largest magnitude in one sweep
    for (int i = 0; i < B_m; i++) {
        for (int j = 0; j < B_n; j++) {
            col[j] = MAX(col[j], fabsf(B[i * B_n + j]));
        }
    }

    for (int j = 0; j < B_n; j++) {
        int exp = scale_exp(col[j]);

        scales[j] = ldexpf(1.f, exp);

        col[j] = ldexpf(1.f, -exp);  // reused below as the reciprocal
    }

    for (int i = 0; i < B_m; i++) {
        for (int j = 0; j < B_n; j++) {
            B_h[i * B_n + j] = f32_to_f16(B[i * B_n + j] * col[j]);
        }
    }

    free(col);
}

void accum_dequantize_block(const f16_t *tiles, int blk_n, int array_dim,
                            int m, int n,
                            const float *row_scales, const float *col_scales,
                            float *C, int C_stride) {
    int tiles_j = blk_n / array_dim;

    // loop across the block's valid elements
    for (int i = 0; i < m; i++) {
        // the row's position within its tile is fixed across the whole row, so
        // only the tile column and the offset within it move in the inner loop
        int ti = i / array_dim;
        int ri = i % array_dim;
        const f16_t *tile_row = tiles + ((size_t)ti * tiles_j) * array_dim * array_dim
                                + (size_t)ri * array_dim;

        for (int j = 0; j < n; j++) {
            int tj = j / array_dim;
            int cj = j % array_dim;

            float v = f16_to_f32(tile_row[(size_t)tj * array_dim * array_dim + cj]);

            // both scales are powers of two, so these multiplies are exact
            C[i * C_stride + j] += v * row_scales[i] * col_scales[j];
        }
    }
}
