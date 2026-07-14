/**
 * mat_ops.h - Basic matrix and linear operations
 */

#ifndef MAT_OPS
#define MAT_OPS

/**
 * @brief Computes `C` := `A` @ `B`
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
void mat_mat_mul(float *A, float *B, int A_m, int A_n, int B_n, float *C);

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
void scal_mat_mul(float a, float *B, int B_m, int B_n, float *C);

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
void mat_add_broadcast(float *A, float *B, int A_m, int A_n, float *C);

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
void mat_sub(float *A, float *B, int m, int n, float *C);

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
void hadamard_product(float *A, float *B, int m, int n, float *C);

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
void transpose(float *A, int A_m, int A_n, float *A_T);

/**
 * @brief Computes `B` := row-sum of `A`
 * 
 * @param[in]  A   Input matrix
 * @param[in]  A_m Number of rows in `A`
 * @param[in]  A_n Number of columns in `A`
 * @param[out] B   Output matrix
 */
void row_sum(float *A, int A_m, int A_n, float *B);

#endif
