/**
 * mat_ops.c - Basic matrix and linear operations
 */

#include "mat_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <xrt.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "config.h"
#include "fpga.h"
#include "quant.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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

void tiled_mat_mat_mul_fpga(float *A, float *B, int M, int K, int N, float *C) {
    // M = A_m, K = A_n, N = B_n

    // Convert both operands to float16 up front rather than per tile. The
    // scales are per-row of A and per-column of B, so they do not vary along
    // k: every tile of a given row/column shares a scale, each tile of A is
    // reused across C's tile columns, and the fixed-point accumulators of
    // separate k tiles can be summed directly below.
    f16_t *A_h = malloc((size_t)M * K * sizeof(f16_t));
    f16_t *B_h = malloc((size_t)K * N * sizeof(f16_t));
    float *A_scales = malloc(M * sizeof(float));
    float *B_scales = malloc(N * sizeof(float));

    quantize_rows(A, M, K, A_h, A_scales);
    quantize_cols(B, K, N, B_h, B_scales);

    int num_tiles_i = (M + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_j = (N + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_k = (K + TILE_DIM - 1) / TILE_DIM;

    f16_t *a_map = xrtBOMap(fpga_bo_a);
    f16_t *b_map = xrtBOMap(fpga_bo_b);
    uint64_t *c_map = xrtBOMap(fpga_bo_c);

    if (!a_map || !b_map || !c_map) {
        fprintf(stderr, "tiled_mat_mat_mul_fpga: failed to map a tile buffer "
                        "(was fpga_init() called?)\n");
        exit(1);
    }

    // running fixed-point total for one tile of C, summed across k in units of
    // 2**ACC_LSB
    int64_t *acc_tile = malloc((size_t)TILE_DIM * TILE_DIM * sizeof(int64_t));

    // loop across C's tiles
    for (int i = 0; i < num_tiles_i; i++) {
        for (int j = 0; j < num_tiles_j; j++) {
            int tile_A_m = MIN(TILE_DIM, M - i * TILE_DIM);
            int tile_B_n = MIN(TILE_DIM, N - j * TILE_DIM);

            memset(acc_tile, 0, (size_t)TILE_DIM * TILE_DIM * sizeof(int64_t));

            // loop across C's tiled matmuls; only one tile of each operand is
            // ever resident on the device, never the whole of A or B
            for (int k = 0; k < num_tiles_k; k++) {
                int tile_A_n = MIN(TILE_DIM, K - k * TILE_DIM);

                // copy A(i,k) and B(k,j) tiles into the reused tile buffers
                for (int r = 0; r < tile_A_m; r++) {
                    memcpy(a_map + r * TILE_DIM,
                           A_h + (i * TILE_DIM + r) * K + k * TILE_DIM,
                           tile_A_n * sizeof(f16_t));
                }
                for (int r = 0; r < tile_A_n; r++) {
                    memcpy(b_map + r * TILE_DIM,
                           B_h + (k * TILE_DIM + r) * N + j * TILE_DIM,
                           tile_B_n * sizeof(f16_t));
                }

                // The array consumes a fixed-depth tile: ARRAY_DIM is a
                // compile-time parameter of systolic_array.sv, not a runtime
                // input, so it multiplies and accumulates the whole tile
                // regardless of tile_A_n. On a short k tile the remainder of
                // the buffers still holds the previous launch's operands,
                // which would otherwise be summed into the result, so zero
                // the padding. An all-zero halfword is float16 +0.0, so a
                // memset still pads with a value the MAC treats as zero. Both
                // memsets are no-ops unless k is ragged.
                for (int r = 0; r < tile_A_m; r++) {
                    memset(a_map + r * TILE_DIM + tile_A_n, 0,
                           (size_t)(TILE_DIM - tile_A_n) * sizeof(f16_t));
                }
                memset(b_map + (size_t)tile_A_n * TILE_DIM, 0,
                       (size_t)(TILE_DIM - tile_A_n) * TILE_DIM * sizeof(f16_t));

                // B is synced in full, since its zeroed padding rows extend
                // past tile_A_n and those zeros have to reach the device
                xrtBOSync(fpga_bo_a, XCL_BO_SYNC_BO_TO_DEVICE, (size_t)tile_A_m * TILE_DIM * sizeof(f16_t), 0);
                xrtBOSync(fpga_bo_b, XCL_BO_SYNC_BO_TO_DEVICE, (size_t)TILE_DIM * TILE_DIM * sizeof(f16_t), 0);

                xrtRunSetArg(fpga_matmul_run, 0, fpga_bo_a);
                xrtRunSetArg(fpga_matmul_run, 1, fpga_bo_b);
                xrtRunSetArg(fpga_matmul_run, 2, fpga_bo_c);
                xrtRunSetArg(fpga_matmul_run, 3, tile_A_m);
                xrtRunSetArg(fpga_matmul_run, 4, tile_A_n);
                xrtRunSetArg(fpga_matmul_run, 5, tile_B_n);
                xrtRunStart(fpga_matmul_run);
                xrtRunWait(fpga_matmul_run);

                xrtBOSync(fpga_bo_c, XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)tile_A_m * TILE_DIM * sizeof(uint64_t), 0);

                // The array zeroes its accumulators on every launch, so each
                // launch yields only this k tile's partial products. Summing
                // them here is exact: all k tiles share a scale, and every
                // accumulator uses the same fixed-point LSB weight, so these
                // are plain integer additions.
                for (int r = 0; r < tile_A_m; r++) {
                    for (int c = 0; c < tile_B_n; c++) {
                        acc_tile[r * TILE_DIM + c] += sign_extend_acc(c_map[r * TILE_DIM + c]);
                    }
                }
            }

            dequantize_block(acc_tile, TILE_DIM,
                             tile_A_m, tile_B_n,
                             A_scales + i * TILE_DIM, B_scales + j * TILE_DIM,
                             C + (i * TILE_DIM) * N + j * TILE_DIM, N);
        }
    }

    free(acc_tile);
    free(A_scales);
    free(B_scales);
    free(A_h);
    free(B_h);
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
