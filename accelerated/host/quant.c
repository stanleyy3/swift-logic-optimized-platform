/**
 * quant.c - Symmetric linear quantization between float32 and the systolic
 *           array's int16 operands / int48 accumulators
 */

#include "quant.h"

#include <stdlib.h>
#include <math.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/**
 * @brief Rounds and clamps a scaled value into int16 range
 *
 * @param[in] scaled Value already divided by its scale
 * @return           Quantized value
 */
static inline int16_t quantize_elem(float scaled) {
    long q = lrintf(scaled);

    return (int16_t)MIN(MAX(q, -QUANT_MAX), QUANT_MAX);
}

void quantize_rows(float *A,
                   int A_m, int A_n,
                   int16_t *A_q, float *scales) {
    // loop across A's rows
    for (int i = 0; i < A_m; i++) {
        // find the row's largest magnitude
        float max_abs = 0.f;

        for (int j = 0; j < A_n; j++) {
            max_abs = MAX(max_abs, fabsf(A[i * A_n + j]));
        }

        // an all-zero row quantizes to zeros under any scale, so any nonzero
        // scale avoids a division by zero here and at dequantization
        float scale = (max_abs > 0.f) ? (max_abs / QUANT_MAX) : 1.f;
        float inv_scale = 1.f / scale;

        scales[i] = scale;

        for (int j = 0; j < A_n; j++) {
            A_q[i * A_n + j] = quantize_elem(A[i * A_n + j] * inv_scale);
        }
    }
}

void quantize_cols(float *B,
                   int B_m, int B_n,
                   int16_t *B_q, float *scales) {
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

    // an all-zero column quantizes to zeros under any scale, so any nonzero
    // scale avoids a division by zero here and at dequantization
    for (int j = 0; j < B_n; j++) {
        scales[j] = (col[j] > 0.f) ? (col[j] / QUANT_MAX) : 1.f;

        col[j] = 1.f / scales[j];  // reused below as the reciprocal
    }

    for (int i = 0; i < B_m; i++) {
        for (int j = 0; j < B_n; j++) {
            B_q[i * B_n + j] = quantize_elem(B[i * B_n + j] * col[j]);
        }
    }

    free(col);
}

void dequantize_block(int64_t *acc, int acc_stride,
                      int m, int n,
                      float *row_scales, float *col_scales,
                      float *C, int C_stride) {
    // loop across the block's elements
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * C_stride + j] = (float)acc[i * acc_stride + j]
                                  * row_scales[i] * col_scales[j];
        }
    }
}
