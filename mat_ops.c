/**
 * mat_ops.c - Basic matrix and linear operations
 */

#include "mat_ops.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void mat_mat_mul(float *A, float *B, int A_m, int A_n, int B_n, float *C) {
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

void scal_mat_mul(float a, float *B, int B_m, int B_n, float *C) {
    // loop across C's elements
    for (int i = 0; i < B_m; i++){
        for (int j = 0; j < B_n; j++) {
            C[i * B_n + j] = a * B[i * B_n + j];
        }
    }
}

void mat_add_broadcast(float *A, float *B, int A_m, int A_n, float *C) {
    // loop across C's elements
    for (int i = 0; i < A_m; i++) {
        for (int j = 0; j < A_n; j++) {
            C[i * A_n + j] = A[i * A_n + j] + B[i];
        }
    }
}

void mat_sub(float *A, float *B, int m, int n, float *C) {
    // loop across C's elements
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

void hadamard_product(float *A, float *B, int m, int n, float *C) {
    // loop over C's elements
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] * B[i * n + j];
        }
    }
}

void transpose(float *A, int A_m, int A_n, float *A_T) {
    // loop across A's elements
    for (int i = 0; i < A_m; i++) {
        for (int j = 0; j < A_n; j++) {
            A_T[j * A_m + i] = A[i * A_n + j];
        }
    }
}

void row_sum(float *A, int A_m, int A_n, float *B) {
    // loop across B's elements
    for (int i = 0; i < A_m; i++) {
        float acc = 0.f;

        for (int j = 0; j < A_n; j++) {
            acc += A[i * A_n + j];
        }

        B[i] = acc;
    }
}

void relu(float *Z, int Z_m, int Z_n, float *A) {
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            A[i * Z_n + j] = MAX(0, Z[i * Z_n + j]);
        }
    }
}

void relu_deriv(float *Z, int Z_m, int Z_n, float *Z_p) {
    // loop over Z_p's elements
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            Z_p[i * Z_n + j] = (float)(Z[i * Z_n + j] > 0);
        }
    }
}

void softmax(float *Z, int Z_m, int Z_n, float *A) {
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

void one_hot(int *Y, int dim, int batch_size, float *one_hot_Y) {
    memset(one_hot_Y, 0, dim * batch_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        one_hot_Y[Y[i] * batch_size + i] = 1.f;
    }
}
