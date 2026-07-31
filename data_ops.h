/**
 * data_ops.h - Management of data
 * 
 * Loading and manipulation of data for the model
 */

#ifndef _DATA_OPS_H_
#define _DATA_OPS_H_

// dataset files needed for a run: training and test inputs and labels
#define NUM_DATA_FILES 4

typedef struct {
    float *input;
    int *labels;
} Dataset;

/**
 * @brief Checks that every dataset file can be opened
 *
 * - The paths are relative, so this is mostly a check that the program was
 * started from the project root
 *
 * - Worth calling before the UI takes over the screen, because the loader's own
 * failure messages would be lost along with the alternate screen buffer
 *
 * @return Path of the first unreadable file, or NULL if all of them are fine
 */
const char *missing_data_file(void);

/**
 * @brief Allocates memory for training and test sets and loads the MNIST
 * dataset into them
 * 
 * - Caller owns the allocated memory
 * 
 * @param[out] train_set      Training set
 * @param[out] test_set       Test set
 * @param[out] train_set_size Size of training set
 * @param[out] test_set_size  Size of test set
 */
void init_data_MNIST(Dataset *train_set, Dataset *test_set,
                     int *train_set_size, int *test_set_size);

/**
 * @brief Frees memory allocated for dataset
 * 
 * @param[in] dataset Dataset to be freed
 */
void free_data_set(Dataset *dataset);

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
