/**
 * mat_ops.h - Basic matrix and linear operations
 */

#ifndef _MAT_OPS_H_
#define _MAT_OPS_H_

/**
 * @brief Computes `C` := `A` @ `B` (with naive loops)
 * 
 * - `B` is inferred to have `A_n` rows
 * 
 * - `C` cannot be one of `A` or `B`
 * 
 * @param[in]  A   First input matrix
 * @param[in]  B   Second input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A` (`B` is inferred to have
 *                 the same number of rows)
 * @param[in]  B_n Number of columns in `B`
 * @param[out] C   Output matrix
 */
void naive_mat_mat_mul(float *A, float *B,
                       int A_m, int A_n, int B_n,
                       float *C);

/**
 * @brief Computes `C` := `A` @ `B` (with tiling)
 * 
 * - `B` is inferred to have `A_n` rows
 * 
 * - `C` cannot be one of `A` or `B`
 * 
 * @param[in]  A   First input matrix
 * @param[in]  B   Second input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A` (`B` is inferred to have
 *                 the same number of rows)
 * @param[in]  B_n Number of columns in `B`
 * @param[out] C   Output matrix
 */
void tiled_mat_mat_mul(float *restrict A, float *restrict B,
                       int A_m, int A_n, int B_n,
                       float *restrict C);

/**
 * @brief Computes `C` := `A` @ `B` (tiled, offloaded to the FPGA systolic array)
 *
 * - `B` is inferred to have `A_n` rows
 *
 * - `C` cannot be one of `A` or `B`
 *
 * - requires `fpga_init()` to have been called first
 *
 * @param[in]  A First input matrix
 * @param[in]  B Second input matrix
 * @param[in]  M Number of rows in `A`
 * @param[in]  K Number of columns in `A` (`B` is inferred to have
 *               the same number of rows)
 * @param[in]  N Number of columns in `B`
 * @param[out] C Output matrix
 */
void tiled_mat_mat_mul_fpga(float *A, float *B,
                            int M, int K, int N,
                            float *C);

/**
 * @brief Computes `C` := `A` @ `B`, on whichever engine `USE_FPGA_MATMUL` selects
 *
 * The matmul a training pass calls. Keeping the choice in one place means the
 * CPU and FPGA paths can be diffed against each other by flipping one flag in
 * config.h rather than editing call sites.
 *
 * @param[in]  A   First input matrix
 * @param[in]  B   Second input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A` (`B` is inferred to have
 *                 the same number of rows)
 * @param[in]  B_n Number of columns in `B`
 * @param[out] C   Output matrix
 */
void mat_mat_mul(float *A, float *B,
                 int A_m, int A_n, int B_n,
                 float *C);

/**
 * @brief Computes `C` := `a` * `B`
 * 
 * - `a` is a scalar and `B` is a matrix
 * 
 * - `C` can be `B`
 * 
 * @param[in]  a   Input scalar
 * @param[in]  B   Input matrix
 * @param[in]  B_m Number of rows in `B`
 * @param[in]  B_n Number of columns in `B`
 * @param[out] C   Output matrix
 */
void scal_mat_mul(float a, float *B,
                  int B_m, int B_n,
                  float *C);

/**
 * @brief Computes `C` := `A` + `B` (broadcast)
 *
 * - `C` can be `A`
 * 
 * - `A` and `C` are matrices
 * 
 * - `B` is a column vector
 * 
 * @param[in]  A   Input matrix
 * @param[in]  B   Input vector
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A`
 * @param[out] C   Output vector
 */
void mat_vec_add_broadcast(float *A, float *B,
                           int A_m, int A_n,
                           float *C);

/**
 * @brief Computes `C` := `A` - `B`
 *
 * - `C` can be one of `A` or `B`
 * 
 * @param[in]  A First input vector
 * @param[in]  B Second input vector
 * @param[in]  m Number of elements in `A` and `B`
 * @param[out] C Output vector
 */
void mat_sub(float *A, float *B,
             int m, int n,
             float *C);

/**
 * @brief Computes `C` := hadamard product of `A` and `B`
 * 
 * - `C` can be one of `A` or `B`
 * 
 * @param[in]  A First input matrix
 * @param[in]  B Second input matrix
 * @param[in]  m Number of rows in `A` and `B`
 * @param[in]  n Number of columns in `A` and `B`
 * @param[out] C Output matrix
 */
void hadamard_product(float *A, float *B,
                      int m, int n,
                      float *C);

/**
 * @brief Computes `A_T` := `A`^T
 * 
 * - `A_T` cannot be `A`
 * 
 * @param[in]  A   Input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A`
 * @param[out] A_T Output matrix
 */
void transpose(float *A,
               int A_m, int A_n,
               float *A_T);

/**
 * @brief Computes `B` := row-sum of `A`
 * 
 * @param[in]  A   Input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A`
 * @param[out] B   Output matrix
 */
void row_sum(float *A,
             int A_m, int A_n,
             float *B);

/**
 * @brief Computes ReLU for each element of `Z`
 * 
 * @param[in]  Z   Weighted sums of layer
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] A   ReLU activations of layer
 */
void relu(float *Z,
          int Z_m, int Z_n,
          float *A);

/**
 * @brief Computes ReLU derivative for each element of `Z`
 * 
 * @param[in]  Z   Input matrix
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] Z_p Output matrix
 */
void relu_deriv(float *Z,
                int Z_m, int Z_n,
                float *Z_p);

/**
 * @brief Computes softmax for each element of `Z`
 * 
 * @param[in]  Z   Weighted sums of layer
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] A   Softmax activations of layer
 */
void softmax(float *Z,
             int Z_m, int Z_n,
             float *A);

/**
 * @brief Creates a batch one-hot encoding of a batch's labels
 * 
 * @param[in]  Y          Labels
 * @param[in]  dim        Dimension of output layer
 * @param[in]  batch_size Size of training batch
 * @param[out] one_hot_Y  One-hot encoding of `Y`
 */
void one_hot(int *Y,
             int dim, int batch_size,
             float *one_hot_Y);

#endif
