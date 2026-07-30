/**
 * mat_ops.c - Basic matrix and linear operations
 */

#include "mat_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "config.h"
#include "fpga.h"
#include "quant.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// rounds up to the next multiple of the array dimension; control.sv derives its
// tile counts by truncating division, so a launch dimension that is not a
// multiple of FPGA_ARRAY_DIM silently loses its leftover rows/columns
#define ROUND_UP_ARRAY_DIM(x) \
    ((((x) + FPGA_ARRAY_DIM - 1) / FPGA_ARRAY_DIM) * FPGA_ARRAY_DIM)

void naive_mat_mat_mul(float *A, float *B,
                       int A_m, int A_n, int B_n,
                       float *C) {
    // loop across C's elements
    for (int i = 0; i < A_m; i++) {
        for (int j = 0; j < B_n; j++) {
            float acc = 0.f;

            // loop across C_i,j's constituents
            for (int k = 0; k < A_n; k++) {
                acc += A[i * A_n + k] * B[k * B_n + j];
            }

            C[i * B_n + j] = acc;
        }
    }
}

/**
 * @brief Computes "tile" `C` := "tile" `A` @ "tile" `B`
 * 
 * - `A`, `B`, and `C` are "tile"s of larger matrices
 * 
 * - `C` cannot be one of `A` or `B`
 * 
 * - accumulates directly into "tile" C
 * 
 * - optimized for compiler vectorization
 * 
 * @param[in]  A        First input "tile" (whose address is within larger matrix)
 * @param[in]  B        Second input "tile" (whose address is within larger matrix)
 * @param[in]  tile_A_m Number of rows in "tile" `A`
 * @param[in]  tile_A_n Number of columns in "tile" `A` ("tile" `B` is inferred to have
 *                      the same number of rows)
 * @param[in]  tile_B_n Number of columns in "tile" `B`
 * @param[in]  A_n      Number of columns in larger matrix `A`
 * @param[in]  B_n      Number of columsn in larger matrix `B`
 * @param[out] C        Ouput "tile" (whose address is within larger matrix)
 */
static void mini_mat_mat_mul(float *restrict A, float *restrict B,
                             int tile_A_m, int tile_A_n, int tile_B_n,
                             int A_n, int B_n,
                             float *restrict C) {
    // loop across A's elements
    for (int i = 0; i < tile_A_m; i++) {
        for (int k = 0; k < tile_A_n; k++) {
            float A_i_k = A[i * A_n + k];

            // loop across B's columns
            for (int j = 0; j < tile_B_n; j++) {
                C[i * B_n + j] += A_i_k * B[k * B_n + j];
            }
        }
    }
}

void tiled_mat_mat_mul(float *restrict A, float *restrict B,
                       int A_m, int A_n, int B_n,
                       float *restrict C) {
    // zero-initialize C to be accumulated into
    memset(C, 0, A_m * B_n * sizeof(float));

    int num_tiles_i = (A_m + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_j = (B_n + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_k = (A_n + TILE_DIM - 1) / TILE_DIM;

    // loop across C's tiles (executed in parallel)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < num_tiles_i; i++) {
        for (int j = 0; j < num_tiles_j; j++) {
            // loop across C's tiled matmuls
            for (int k = 0; k < num_tiles_k; k++) {
                int tile_A_m = MIN(TILE_DIM, A_m - i * TILE_DIM);
                int tile_A_n = MIN(TILE_DIM, A_n - k * TILE_DIM);
                int tile_B_n = MIN(TILE_DIM, B_n - j * TILE_DIM);

                // compute tiled matmul and accumulate into tile C
                mini_mat_mat_mul(A + (i * TILE_DIM * A_n + k * TILE_DIM), B + (k * TILE_DIM * B_n + j * TILE_DIM),
                                 tile_A_m, tile_A_n, tile_B_n,
                                 A_n, B_n,
                                 C + (i * TILE_DIM * B_n + j * TILE_DIM));
            }
        }
    }
}

/**
 * @brief Stages one block of a quantized operand into a device buffer, padding
 *        the ragged edges with float16 zeros
 *
 * The array reads a block as `rows` x `cols` row-major with a row stride of
 * `cols`, and multiplies the whole padded block regardless of how much of it is
 * real. An all-zero halfword is float16 +0.0, which the MAC flushes to zero, so
 * padding contributes nothing to the result and the padded output rows and
 * columns are simply never read back.
 *
 * @param[out] dst       Device buffer to stage into
 * @param[in]  rows      Padded row count (the launch's dimension)
 * @param[in]  cols      Padded column count (also the row stride)
 * @param[in]  src       Quantized source matrix
 * @param[in]  src_row0  Index of the block's first row within `src`
 * @param[in]  src_col0  Index of the block's first column within `src`
 * @param[in]  src_cols  Number of columns per row of `src`
 * @param[in]  valid_r   Rows of `src` that fall inside the block
 * @param[in]  valid_c   Columns of `src` that fall inside the block
 */
static void stage_block(f16_t *dst, int rows, int cols,
                        const f16_t *src, int src_row0, int src_col0, int src_cols,
                        int valid_r, int valid_c) {
    for (int r = 0; r < valid_r; r++) {
        memcpy(dst + (size_t)r * cols,
               src + (size_t)(src_row0 + r) * src_cols + src_col0,
               (size_t)valid_c * sizeof(f16_t));

        // column tail of a real row
        memset(dst + (size_t)r * cols + valid_c, 0,
               (size_t)(cols - valid_c) * sizeof(f16_t));
    }

    // row tail: whole padding rows
    memset(dst + (size_t)valid_r * cols, 0,
           (size_t)(rows - valid_r) * cols * sizeof(f16_t));
}

void tiled_mat_mat_mul_fpga(float *A, float *B, int M, int K, int N, float *C) {
    // M = A_m, K = A_n, N = B_n

    // Convert both operands to float16 up front rather than per block. The
    // scales are per-row of A and per-column of B, so they do not vary along k:
    // every block of a given row/column shares a scale, which is what lets the
    // block results be summed below.
    f16_t *A_h = malloc((size_t)M * K * sizeof(f16_t));
    f16_t *B_h = malloc((size_t)K * N * sizeof(f16_t));
    float *A_scales = malloc(M * sizeof(float));
    float *B_scales = malloc(N * sizeof(float));

    if (!A_h || !B_h || !A_scales || !B_scales) {
        fprintf(stderr, "tiled_mat_mat_mul_fpga: out of memory quantizing a "
                        "%dx%d by %dx%d matmul\n", M, K, K, N);
        exit(1);
    }

    quantize_rows(A, M, K, A_h, A_scales);
    quantize_cols(B, K, N, B_h, B_scales);

    f16_t *a_block = fpga_block_a();
    f16_t *b_block = fpga_block_b();
    const f16_t *c_block = fpga_block_c();

    int num_blocks_i = (M + FPGA_BLOCK_DIM - 1) / FPGA_BLOCK_DIM;
    int num_blocks_j = (N + FPGA_BLOCK_DIM - 1) / FPGA_BLOCK_DIM;
    int num_blocks_k = (K + FPGA_BLOCK_DIM - 1) / FPGA_BLOCK_DIM;

    // C is accumulated into, since a K larger than the block dimension arrives
    // as several block results
    memset(C, 0, (size_t)M * N * sizeof(float));

    // Loop across C's blocks. The k loop sits outside the j loop so that the A
    // block, which is indexed by (i,k), stays put across a whole sweep of j -
    // the array keeps it resident in its large buffer and only reloads B. That
    // also hoists A's staging and its sync out of the inner loop.
    for (int i = 0; i < num_blocks_i; i++) {
        int valid_m = MIN(FPGA_BLOCK_DIM, M - i * FPGA_BLOCK_DIM);
        int blk_m = ROUND_UP_ARRAY_DIM(valid_m);

        for (int k = 0; k < num_blocks_k; k++) {
            int valid_k = MIN(FPGA_BLOCK_DIM, K - k * FPGA_BLOCK_DIM);
            int blk_k = ROUND_UP_ARRAY_DIM(valid_k);

            stage_block(a_block, blk_m, blk_k,
                        A_h, i * FPGA_BLOCK_DIM, k * FPGA_BLOCK_DIM, K,
                        valid_m, valid_k);

            for (int j = 0; j < num_blocks_j; j++) {
                int valid_n = MIN(FPGA_BLOCK_DIM, N - j * FPGA_BLOCK_DIM);
                int blk_n = ROUND_UP_ARRAY_DIM(valid_n);

                stage_block(b_block, blk_k, blk_n,
                            B_h, k * FPGA_BLOCK_DIM, j * FPGA_BLOCK_DIM, N,
                            valid_k, valid_n);

                // A only has to be pushed on the first launch of this sweep;
                // blk_m and blk_k are fixed across it, so the resident copy
                // stays valid
                int load_block_en = FPGA_LOAD_B | ((j == 0) ? FPGA_LOAD_A : 0);

                fpga_launch_block(blk_m, blk_k, blk_n, load_block_en);

                // the array accumulated this block's whole K internally; only
                // the sum across k blocks is left, and it happens here
                accum_dequantize_block(c_block, blk_n, FPGA_ARRAY_DIM,
                                       valid_m, valid_n,
                                       A_scales + i * FPGA_BLOCK_DIM,
                                       B_scales + j * FPGA_BLOCK_DIM,
                                       C + (size_t)(i * FPGA_BLOCK_DIM) * N
                                         + j * FPGA_BLOCK_DIM,
                                       N);
            }
        }
    }

    free(A_scales);
    free(B_scales);
    free(A_h);
    free(B_h);
}

void mat_mat_mul(float *A, float *B,
                 int A_m, int A_n, int B_n,
                 float *C) {
#if USE_FPGA_MATMUL
    tiled_mat_mat_mul_fpga(A, B, A_m, A_n, B_n, C);
#else
    tiled_mat_mat_mul(A, B, A_m, A_n, B_n, C);
#endif
}

void scal_mat_mul(float a, float *B,
                  int B_m, int B_n,
                  float *C) {
    // loop across C's elements
    for (int i = 0; i < B_m; i++){
        for (int j = 0; j < B_n; j++) {
            C[i * B_n + j] = a * B[i * B_n + j];
        }
    }
}

void mat_vec_add_broadcast(float *A, float *B,
                           int A_m, int A_n,
                           float *C) {
    // loop across C's elements
    for (int i = 0; i < A_m; i++) {
        for (int j = 0; j < A_n; j++) {
            C[i * A_n + j] = A[i * A_n + j] + B[i];
        }
    }
}

void mat_sub(float *A, float *B,
             int m, int n,
             float *C) {
    // loop across C's elements
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

void hadamard_product(float *A, float *B,
                      int m, int n,
                      float *C) {
    // loop over C's elements
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] * B[i * n + j];
        }
    }
}

void transpose(float *A,
               int A_m, int A_n,
               float *A_T) {
    // loop across A's elements
    for (int i = 0; i < A_m; i++) {
        for (int j = 0; j < A_n; j++) {
            A_T[j * A_m + i] = A[i * A_n + j];
        }
    }
}

void row_sum(float *A,
             int A_m, int A_n,
             float *B) {
    // loop across B's elements
    for (int i = 0; i < A_m; i++) {
        float acc = 0.f;

        for (int j = 0; j < A_n; j++) {
            acc += A[i * A_n + j];
        }

        B[i] = acc;
    }
}

void relu(float *Z,
          int Z_m, int Z_n,
          float *A) {
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            A[i * Z_n + j] = MAX(0, Z[i * Z_n + j]);
        }
    }
}

void relu_deriv(float *Z,
                int Z_m, int Z_n,
                float *Z_p) {
    // loop over Z_p's elements
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            Z_p[i * Z_n + j] = (float)(Z[i * Z_n + j] > 0);
        }
    }
}

void softmax(float *Z,
             int Z_m, int Z_n,
             float *A) {
    float *max_logit = malloc(Z_n * sizeof(float));
    float *norm_term = calloc(Z_n, sizeof(float));

    // find max of each sample's output logits
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            if (i == 0) {
                max_logit[j] = Z[j];
            } else {
                max_logit[j] = MAX(max_logit[j], Z[i * Z_n + j]);
            }
        }
    }

    // compute exponentiated elements and accumulate sum
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++ ) {
            float exp_Z_i = expf(Z[i * Z_n + j] - max_logit[j]);  // subtract per-sample max logit for numerical stability

            A[i * Z_n + j] = exp_Z_i;
            norm_term[j] += exp_Z_i;
        }
    }

    // normalize each exponentiated element
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            A[i * Z_n + j] = A[i * Z_n + j] / norm_term[j];
        }
    }

    free(max_logit);
    free(norm_term);
}

void one_hot(int *Y,
             int dim, int batch_size,
             float *one_hot_Y) {
    memset(one_hot_Y, 0, dim * batch_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        one_hot_Y[Y[i] * batch_size + i] = 1.f;
    }
}
