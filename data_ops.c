/**
 * data_ops.c - Management of data
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void check_read_error(size_t num_read, size_t count, FILE *stream) {
    if (num_read < count) {
        if (ferror(stream)) {
            perror("File read error");
            clearerr(stream);
        } else if (feof(stream)) {
            printf("End of file reached.\n");
        }
    }
}

uint32_t swap_endian32(uint32_t val) {
    return (val << 24)
           | ((val << 8) & 0x00FF0000)
           | ((val >> 8) & 0x0000FF00)
           | (val >> 24);
}

void load_mnist(float *train_input, int *train_labels,
                float *test_input, int *test_labels) {

    char *train_input_path = "data/mnist/train-images.idx3-ubyte";
    char *train_labels_path = "data/mnist/train-labels.idx1-ubyte";
    char *test_input_path = "data/mnist/test-images.idx3-ubyte";
    char *test_labels_path = "data/mnist/test-labels.idx1-ubyte";

    unsigned char magic_bytes[4];
    int num_dims;
    uint32_t *dims;
    size_t total_elements;
    unsigned char *raw_bytes;

    size_t num_read;

    ////////////////////////////////////////////////////////////////////////////////
    // TRAINING INPUT
    ////////////////////////////////////////////////////////////////////////////////

    FILE *train_input_file = fopen(train_input_path, "rb");
    if (!train_input_file) {
        printf("Error: Cannot open %s\n", train_input_path);
        exit(1);
    }

    // read magic numbers
    num_read = fread(magic_bytes, sizeof(unsigned char), 4, train_input_file);
    check_read_error(num_read, 4, train_input_file);
    num_dims = magic_bytes[3];

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, train_input_file);
    check_read_error(num_read, num_dims, train_input_file);

    // calculate number of total elements
    total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        total_elements *= swap_endian32(dims[i]);
    }

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, train_input_file);
    check_read_error(num_read, total_elements, train_input_file);
    for (size_t i = 0; i < total_elements; i++) {
        train_input[i] = (float)raw_bytes[i] / 255.f;
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

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, train_labels_file);
    check_read_error(num_read, total_elements, train_labels_file);
    for (size_t i = 0; i < total_elements; i++) {
        train_labels[i] = (int)raw_bytes[i];
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

    // get dimension sizes
    dims = malloc(num_dims * sizeof(uint32_t));
    num_read = fread(dims, sizeof(uint32_t), num_dims, test_input_file);
    check_read_error(num_read, num_dims, test_input_file);

    // calculate number of total elements
    total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        total_elements *= swap_endian32(dims[i]);
    }

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, test_input_file);
    check_read_error(num_dims, total_elements, test_input_file);
    for (size_t i = 0; i < total_elements; i++) {
        test_input[i] = (float)raw_bytes[i] / 255.f;
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

    // read in data
    raw_bytes = malloc(total_elements * sizeof(unsigned char));
    num_read = fread(raw_bytes, sizeof(unsigned char), total_elements, test_labels_file);
    check_read_error(num_read, total_elements, test_labels_file);
    for (size_t i = 0; i < total_elements; i++) {
        test_labels[i] = (int)raw_bytes[i];
    }

    fclose(test_labels_file);
    free(dims);
    free(raw_bytes);
    
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
