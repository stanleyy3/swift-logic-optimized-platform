/**
 * data_ops.c - Management of data
 * 
 * Loading and manipulation of data for the model
 */

#include "data_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "config.h"

static void check_read_error(size_t num_read, size_t count, FILE *stream) {
    if (num_read < count) {
        if (ferror(stream)) {
            perror("File read error");
            clearerr(stream);
        } else if (feof(stream)) {
            printf("End of file reached.\n");
        }

        exit(1);
    }
}

static uint32_t swap_endian32(uint32_t val) {
    return (val << 24)
           | ((val << 8) & 0x00FF0000)
           | ((val >> 8) & 0x0000FF00)
           | (val >> 24);
}

void init_data_MNIST(Dataset *train_set, Dataset *test_set,
                     int *train_set_size, int *test_set_size) {

#if DATASET == 0
    // MNIST dataset
    char *train_input_path = "../../data/mnist/train-images.idx3-ubyte";
    char *train_labels_path = "../../data/mnist/train-labels.idx1-ubyte";
    char *test_input_path = "../../data/mnist/test-images.idx3-ubyte";
    char *test_labels_path = "../../data/mnist/test-labels.idx1-ubyte";
#elif DATASET == 1
    // Fashion MNIST dataset
    char *train_input_path = "../../data/fashion_mnist/train-images.idx3-ubyte";
    char *train_labels_path = "../../data/fashion_mnist/train-labels.idx1-ubyte";
    char *test_input_path = "../../data/fashion_mnist/test-images.idx3-ubyte";
    char *test_labels_path = "../../data/fashion_mnist/test-labels.idx1-ubyte";
#endif

    unsigned char magic_bytes[4];
    int num_dims;
    uint32_t *dims;
    int total_elements;
    unsigned char *raw_bytes;

    size_t num_read;

    ////////////////////////////////////////////////////////////////////////////////
    // TRAINING INPUT
    ////////////////////////////////////////////////////////////////////////////////

    // note: training input is loaded sample-major

    FILE *train_input_file = fopen(train_input_path, "rb");
    if (!train_input_file) {
        printf("Error: Cannot open %s\n", train_input_path);
        exit(1);
    }

    // read magic numbers
    num_read = fread(magic_bytes, sizeof(unsigned char), 4, train_input_file);
    check_read_error(num_read, 4, train_input_file);
    num_dims = magic_bytes[3];
    if (!(num_dims > 0)) {
        printf("Not enough dimensions.\n");
        exit(1);
    }

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, train_input_file);
    check_read_error(num_read, num_dims, train_input_file);

    // calculate number of total elements
    int local_train_set_size = swap_endian32(dims[0]);
    int train_input_dim = 1;
    for (int i = 1; i < num_dims; i++) {
        train_input_dim *= (int)swap_endian32(dims[i]);
    }
    total_elements = local_train_set_size * train_input_dim;

    // allocate training input memory
    train_set->input = malloc(total_elements * sizeof(float));

    // set training set size
    *train_set_size = local_train_set_size;

    // read in data (sample-major)
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, train_input_file);
    check_read_error(num_read, total_elements, train_input_file);
    for (int i = 0; i < train_input_dim; i++) {
        for (int j = 0; j < local_train_set_size; j++) {
            train_set->input[j * train_input_dim + i] = (float)raw_bytes[j * train_input_dim + i] / 255.f;
        }
    }

    fclose(train_input_file);
    free(dims);
    free(raw_bytes);

    ////////////////////////////////////////////////////////////////////////////////
    // TRAINING LABELS
    ////////////////////////////////////////////////////////////////////////////////

    FILE *train_labels_file = fopen(train_labels_path, "rb");
    if (!train_labels_file) {
        printf("Error: Cannot open %s\n", train_labels_path);
        exit(1);
    }

    // read magic numbers
    num_read = fread(magic_bytes, sizeof(unsigned char), 4, train_labels_file);
    check_read_error(num_read, 4, train_labels_file);
    num_dims = magic_bytes[3];

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, train_labels_file);
    check_read_error(num_read, num_dims, train_labels_file);

    // calculate number of total elements
    total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        total_elements *= swap_endian32(dims[i]);
    }

    // allocate training label memory
    train_set->labels = malloc(total_elements * sizeof(int));

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, train_labels_file);
    check_read_error(num_read, total_elements, train_labels_file);
    for (int i = 0; i < total_elements; i++) {
        train_set->labels[i] = (int)raw_bytes[i];
    }

    fclose(train_labels_file);
    free(dims);
    free(raw_bytes);

    ////////////////////////////////////////////////////////////////////////////////
    // TEST INPUT
    ////////////////////////////////////////////////////////////////////////////////

    FILE *test_input_file = fopen(test_input_path, "rb");
    if (!test_input_file) {
        printf("Error: Cannot open %s\n", test_input_path);
        exit(1);
    }

    // read magic numbers
    num_read = fread(magic_bytes, sizeof(unsigned char), 4, test_input_file);
    check_read_error(num_read, 4, test_input_file);
    num_dims = magic_bytes[3];
    if (!(num_dims > 0)) {
        printf("Not enough dimensions.\n");
        exit(1);
    }

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, test_input_file);
    check_read_error(num_read, num_dims, test_input_file);

    // calculate number of total elements
    int local_test_set_size = swap_endian32(dims[0]);
    int test_input_dim = 1;
    for (int i = 1; i < num_dims; i++) {
        test_input_dim *= (int)swap_endian32(dims[i]);
    }
    total_elements = local_test_set_size * test_input_dim;

    // allocate test input memory
    test_set->input = malloc(total_elements * sizeof(float));

    // set test set size
    *test_set_size = local_test_set_size;

    // read in data (feature-major)
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, test_input_file);
    check_read_error(num_read, total_elements, test_input_file);
    for (int i = 0; i < test_input_dim; i++) {
        for (int j = 0; j < local_test_set_size; j++) {
            test_set->input[i * local_test_set_size + j] = (float)raw_bytes[j * test_input_dim + i] / 255.f;
        }
    }

    fclose(test_input_file);
    free(dims);
    free(raw_bytes);

    ////////////////////////////////////////////////////////////////////////////////
    // TEST LABELS
    ////////////////////////////////////////////////////////////////////////////////

    FILE *test_labels_file = fopen(test_labels_path, "rb");
    if (!test_labels_file) {
        printf("Error: Cannot open %s\n", test_labels_path);
        exit(1);
    }

    // read magic numbers
    num_read = fread(magic_bytes, sizeof(unsigned char), 4, test_labels_file);
    check_read_error(num_read, 4, test_labels_file);
    num_dims = magic_bytes[3];

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, test_labels_file);
    check_read_error(num_read, num_dims, test_labels_file);

    // calculate number of total elements
    total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        total_elements *= swap_endian32(dims[i]);
    }

    // allocate test label memory
    test_set->labels = malloc(total_elements * sizeof(int));

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, test_labels_file);
    check_read_error(num_read, total_elements, test_labels_file);
    for (int i = 0; i < total_elements; i++) {
        test_set->labels[i] = (int)raw_bytes[i];
    }

    fclose(test_labels_file);
    free(dims);
    free(raw_bytes);
    
}

void free_data_set(Dataset *dataset) {
    free(dataset->input);
    free(dataset->labels);

    free(dataset);
}

void shuffle(int *A, int N) {
    if (N > 1) {
        for (int i = N - 1; i > 0; i--) {
            // pick a random index from 0 to i
            int j = rand() % (i + 1);

            // swap A[i] with A[j]
            int temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}
