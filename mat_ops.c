/**
 * mat_ops.c - Basic matrix and linear operations
 */

#include "mat_ops.h"

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
