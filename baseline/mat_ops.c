/**
 * mat_ops.c - Basic matrix and linear operations
 */

#include "mat_ops.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "config.h"

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
