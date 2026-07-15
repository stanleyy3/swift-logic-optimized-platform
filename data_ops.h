/**
 * data_ops.h - Management of data
 */

#ifndef DATA_OPS
#define DATA_OPS

/**
 * @brief Loads the MNIST dataset into arrays
 * 
 * @param[out] train_input  Array for training image data
 * @param[out] train_labels Array for training labels
 * @param[out] test_input   Array for test image data
 * @param[out] test_labels  Array for test labels
 */
void load_mnist(float *train_input, int *train_labels,
                float *test_input, int *test_labels);

/**
 * @brief Shuffles elements of an array
 * 
 * Uses Fisher-Yates algorithm
 * 
 * @param[in, out] A Array to be shuffled
 * @param[in]      N Size of array
 */
void shuffle(int *A, int N);

#endif
