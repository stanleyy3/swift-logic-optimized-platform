/**
 * mlp.c - Basic MLP for an arbitrary workload
 */

#include "mlp.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "mat_ops.h"
#include "data_ops.h"

////////////////////////////////////////////////////////////////////////////////
// HYPERPARAMETERS
////////////////////////////////////////////////////////////////////////////////

/*
#define TOTAL_SET_SIZE 60000

#define TRAIN_SET_SPLIT 0.8f
// training set split + validation set split = 1.0
#define TRAIN_SET_SIZE (floorf(TRAIN_SET_SPLIT * TOTAL_SET_SIZE))
#define TEST_SET_SIZE (TOTAL_SET_SIZE - TRAIN_SET_SIZE)
*/

#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000

#define INPUT_DIM 784
#define LAYER_1_DIM 10
#define LAYER_2_DIM 10

#define NUM_EPOCHS 50
#define BATCH_SIZE 1000
#define EPOCH_ITERS (TRAIN_SET_SIZE / BATCH_SIZE)
#define NUM_ITERATIONS (EPOCH_ITERS * NUM_EPOCHS)
#define LEARNING_RATE 0.1f

#define TRAIN_UPDATE_FREQ 10

////////////////////////////////////////////////////////////////////////////////
// HELPER MACROS
////////////////////////////////////////////////////////////////////////////////

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

////////////////////////////////////////////////////////////////////////////////
// CORE LOGIC
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Initializes weights of a layer under He scheme
 * 
 * - Good for layers with ReLU activation
 * 
 * - Uses uniform sampling for simpler implementation
 * 
 * @param[in,out] W       Weights (dim. `fan_out` x `fan_in`)
 * @param[in]     fan_out Dimension of next layer
 * @param[in]     fan_in  Dimension of previous layer
 */
static void init_weights_he(float *W, int fan_out, int fan_in) {
    //srand(67u);

    // He initialization uniform sampling boundary
    float b = sqrtf(6.f / fan_in);

    float norm_factor = (1.f / RAND_MAX) * (2 * b);

    // sets each element of weight matrix
    for (int i = 0; i < fan_out; i++) {
        for (int j = 0; j < fan_in; j++) {
            W[i * fan_in + j] = rand() * norm_factor - b;  // constrains value to [-b, b]
        }
    }
}

/**
 * @brief Initializes weights of a layer under Xavier scheme
 * 
 * - Good for layers with softmax activation
 * 
 * - Uses uniform sampling for simpler implementation
 * 
 * @param[in, out] W       Weights (dim. `fan_out` x `fan_in`)
 * @param[in]      fan_out Dimension of next layer
 * @param[in]      fan_in  Dimension of previous layer
 */
static void init_weights_xavier(float *W, int fan_out, int fan_in) {
    //srand(67u);

    // Xavier initialization uniform sampling boundary
    float b = sqrtf(6.f / (fan_in + fan_out));

    float norm_factor = (1.f / RAND_MAX) * (2 * b);

    // sets each element of weight matrix
    for (int i = 0; i < fan_out; i++) {
        for (int j = 0; j < fan_in; j++) {
            W[i * fan_in + j] = rand() * norm_factor - b;  // constrains value to [-b, b]
        }
    }
}

/**
 * @brief Initializes biases of a layer
 * 
 * - Initializes biases to 0
 * 
 * - Zero-initialization works well for layers with ReLU or softmax activation
 * 
 * @param[in, out] B         Biases
 * @param[in]      layer_dim Dimension of layer
 */
static void init_biases(float *B, int layer_dim) {
    // zero-initialize all biases
    memset(B, 0, layer_dim * sizeof(float));
}

/**
 * @brief Computes ReLU for each element of `Z`
 * 
 * @param[in]  Z   Weighted sums of layer
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] A   ReLU activations of layer
 */
static void relu(float *Z, int Z_m, int Z_n, float *A) {
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            A[i * Z_n + j] = MAX(0, Z[i * Z_n + j]);
        }
    }
}

/**
 * @brief Computes softmax for each element of `Z`
 * 
 * @param[in]  Z   Weighted sums of layer
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] A   Softmax activations of layer
 */
static void softmax(float *Z, int Z_m, int Z_n, float *A) {
    float *norm_term = calloc(Z_n, sizeof(float));

    // compute exponentiated elements and accumulate sum
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++ ) {
            float exp_Z_i = expf(Z[i * Z_n + j]);

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

    free(norm_term);
}

/**
 * @brief Does a batched pass of forward propagation
 * 
 * - Note: For a m x n weight matrix, m the layer's dimension and n is the
 * incoming layer's dimension.
 * 
 * @param[in]  W1          Weights of first layer
 * @param[in]  B1          Biases of first layer
 * @param[in]  W2          Weights of second layer
 * @param[in]  B2          Biases of second layer
 * @param[in]  X           Input
 * @param[in]  input_dim   Dimension of input layer
 * @param[in]  layer_1_dim Dimension of layer 1
 * @param[in]  layer_2_dim Dimension of layer 2
 * @param[in]  batch_size  Size of training batch
 * @param[out] Z1          Weighted sums of first layer
 * @param[out] A1          Activations of first layer
 * @param[out] Z2          Weighted sums of second layer
 * @param[out] A2          Activations of second layer
 */
static void batched_forward_prop(float *W1, float *B1, float *W2, float *B2,
                                 float *X,
                                 int input_dim, int layer_1_dim, int layer_2_dim,
                                 int batch_size,
                                 float *Z1, float *A1, float *Z2, float *A2) {

    // propagate through layer 1
    mat_mat_mul(W1, X, layer_1_dim, input_dim, batch_size, Z1);
    mat_add_broadcast(Z1, B1, layer_1_dim, batch_size, Z1);
    relu(Z1, layer_1_dim, batch_size, A1);

    // propagate through layer 2
    mat_mat_mul(W2, A1, layer_2_dim, layer_1_dim, batch_size, Z2);
    mat_add_broadcast(Z2, B2, layer_2_dim, batch_size, Z2);
    softmax(Z2, layer_2_dim, batch_size, A2);

}

/**
 * @brief Computes ReLU derivative for each element of `Z`
 * 
 * @param[in]  Z   Input matrix
 * @param[in]  Z_m Dimension of layer
 * @param[in]  Z_n Batch size
 * @param[out] Z_p Output matrix
 */
static void relu_deriv(float *Z, int Z_m, int Z_n, float *Z_p) {
    // loop over Z_p's elements
    for (int i = 0; i < Z_m; i++) {
        for (int j = 0; j < Z_n; j++) {
            Z_p[i * Z_n + j] = (float)(Z[i * Z_n + j] > 0);
        }
    }
}

/**
 * @brief Creates a batch one-hot encoding of a batch's labels
 * 
 * @param[in]  Y          Labels
 * @param[in]  dim        Dimension of output layer
 * @param[in]  batch_size Size of training batch
 * @param[out] one_hot_Y  One-hot encoding of `Y`
 */
static void one_hot(int *Y, int dim, int batch_size, float *one_hot_Y) {
    memset(one_hot_Y, 0, dim * batch_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        one_hot_Y[Y[i] * batch_size + i] = 1.f;
    }
}

/**
 * @brief Does a batched pass of back propagation
 * 
 * @param[in]  Z1          Weighted sums of first layer
 * @param[in]  A1          Activations of first layer
 * @param[in]  Z2          Weighted sums of second layer
 * @param[in]  A2          Activations of second layer
 * @param[in]  W1          Weights of first layer
 * @param[in]  W2          Weights of second layer
 * @param[in]  X           Input
 * @param[in]  Y           Training point label (one-hot encoded)
 * @param[in]  input_dim   Dimension of input layer
 * @param[in]  layer_1_dim Dimension of first layer
 * @param[in]  layer_2_dim Dimension of second layer
 * @param[in]  batch_size  Size of training batch
 * @param[out] dW1         Gradient for weights of first layer
 * @param[out] dB1         Gradient for biases of first layer
 * @param[out] dW2         Gradient for weights of second layer
 * @param[out] dB2         Gradient for biases of second layer
 */
static void batched_back_prop(float *Z1, float *A1, float *Z2, float *A2,
                              float *W1, float *W2,
                              float *X, float *Y,
                              int input_dim, int layer_1_dim, int layer_2_dim,
                              int batch_size,
                              float *dW1, float *dB1, float *dW2, float *dB2) {

    float batch_size_inv = 1.f / batch_size;

    float *dZ1 = malloc(layer_1_dim * batch_size * sizeof(float));
    float *dZ2 = malloc(layer_2_dim * batch_size * sizeof(float));

    float *A1_T = malloc(layer_1_dim * batch_size * sizeof(float));
    float *W2_T = malloc(layer_2_dim * layer_1_dim * sizeof(float));
    float *X_T = malloc(input_dim * batch_size * sizeof(float));

    float *Z1_p = malloc(layer_1_dim * batch_size * sizeof(float));

    // layer 2 activation gradient
    mat_sub(A2, Y, layer_2_dim, batch_size, dZ2);
    
    // layer 2 weight gradient
    transpose(A1, layer_1_dim, batch_size, A1_T);
    mat_mat_mul(dZ2, A1_T, layer_2_dim, batch_size, layer_1_dim, dW2);
    scal_mat_mul(batch_size_inv, dW2, layer_2_dim, layer_1_dim, dW2);

    // layer 2 bias gradient
    row_sum(dZ2, layer_2_dim, batch_size, dB2);
    scal_mat_mul(batch_size_inv, dB2, layer_2_dim, 1, dB2);

    // layer 1 activation gradient
    transpose(W2, layer_2_dim, layer_1_dim, W2_T);
    mat_mat_mul(W2_T, dZ2, layer_1_dim, layer_2_dim, batch_size, dZ1);
    relu_deriv(Z1, layer_1_dim, batch_size, Z1_p);
    hadamard_product(dZ1, Z1_p, layer_1_dim, batch_size, dZ1);

    // layer 1 weight gradient
    transpose(X, input_dim, batch_size, X_T);
    mat_mat_mul(dZ1, X_T, layer_1_dim, batch_size, input_dim, dW1);
    scal_mat_mul(batch_size_inv, dW1, layer_1_dim, input_dim, dW1);

    // layer 1 bias gradient
    row_sum(dZ1, layer_1_dim, batch_size, dB1);
    scal_mat_mul(batch_size_inv, dB1, layer_1_dim, 1, dB1);

    // free

    free(dZ1);
    free(dZ2);

    free(A1_T);
    free(W2_T);
    free(X_T);

    free(Z1_p);

}

/**
 * @brief Updates parameters with gradient
 * 
 * @param[in, out]  W1          Weights of first layer
 * @param[in, out]  B1          Biases of first layer
 * @param[in, out]  W2          Weights of second layer
 * @param[in, out]  B2          Biases of second layer
 * @param[in]       dW1         Gradient for weights of first layer
 * @param[in]       dB1         Gradient for biases of first layer
 * @param[in]       dW2         Gradient for weights of second layer
 * @param[in]       dB2         Gradient for biases of second layer
 * @param[in]       alpha       Learning rate
 * @param[in]       input_dim   Dimension of input layer
 * @param[in]       layer_1_dim Dimension of first layer
 * @param[in]       layer_2_dim Dimension of second layer
 */
static void update_params(float *W1, float *B1, float *W2, float *B2,
                          float *dW1, float *dB1, float *dW2, float *dB2,
                          float alpha,
                          int input_dim, int layer_1_dim, int layer_2_dim) {
    
    // update W1
    scal_mat_mul(alpha, dW1, layer_1_dim, input_dim, dW1);
    mat_sub(W1, dW1, layer_1_dim, input_dim, W1);
    
    // update B1
    scal_mat_mul(alpha, dB1, layer_1_dim, 1, dB1);
    mat_sub(B1, dB1, layer_1_dim, 1, B1);

    // update W2
    scal_mat_mul(alpha, dW2, layer_2_dim, layer_1_dim, dW2);
    mat_sub(W2, dW2, layer_2_dim, layer_1_dim, W2);

    // update B2
    scal_mat_mul(alpha, dB2, layer_2_dim, 1, dB2);
    mat_sub(B2, dB2, layer_2_dim, 1, B2);

}

/**
 * @brief Computes accuracy of a group of data points (batch or set)
 * 
 * @param[in] A2   Activations of second layer
 * @param[in] Y    Labels of the data points
 * @param[in] A2_m Dimension of second layer
 * @param[in] A2_n Size of group
 * 
 * @return Accuracy of model on group
 */
static float get_accuracy(float *A2, int *Y, int A2_m, int A2_n) {
    float *A2_T = malloc(A2_m * A2_n * sizeof(float));

    transpose(A2, A2_m, A2_n, A2_T);

    float max_prob;
    int max_label;

    int num_correct = 0;

    // for each training example
    for (int i = 0; i < A2_n; i++) {
        max_prob = A2_T[i * A2_m];
        max_label = 0;

        // for each output category
        for (int j = 1; j < A2_m; j++) {
            bool new_max = A2_T[i * A2_m + j] > max_prob;
            max_prob = new_max ? A2_T[i * A2_m + j] : max_prob;
            max_label = new_max ? j : max_label;
        }

        if (max_label == Y[i])
            num_correct++;
    }

    free(A2_T);

    return (float)num_correct / A2_n;
}

/**
 * @brief Computes test accuracy on a test set
 * 
 * - Runs forward propagation for entire test set, then computes accuracy in
 * the same way as training accuracy
 * 
 * @param[in]  W1            Weights of first layer
 * @param[in]  B1            Biases of first layer
 * @param[in]  W2            Weights of second layer
 * @param[in]  B2            Biases of second layer
 * @param[in]  X             Input
 * @param[in]  Y             Labels
 * @param[in]  input_dim     Dimension of input layer
 * @param[in]  layer_1_dim   Dimension of layer 1
 * @param[in]  layer_2_dim   Dimension of layer 2
 * @param[in]  test_set_size Size of test set
 * @param[out] Z1            Weighted sums of first layer
 * @param[out] A1            Activations of first layer
 * @param[out] Z2            Weighted sums of second layer
 * @param[out] A2            Activations of second layer
 * 
 * @return Accuracy of model on test set
 */
static float get_test_accuracy(float *W1, float *B1, float *W2, float *B2,
                               float *X, int *Y,
                               int input_dim, int layer_1_dim, int layer_2_dim,
                               int test_set_size,
                               float *Z1, float *A1, float *Z2, float *A2) {

    batched_forward_prop(W1, B1, W2, B2,
                         X,
                         input_dim, layer_1_dim, layer_2_dim,
                         test_set_size,
                         Z1, A1, Z2, A2);

    return get_accuracy(A2, Y, layer_2_dim, test_set_size);

}

////////////////////////////////////////////////////////////////////////////////
// TRAINING / GRADIENT DESCENT
////////////////////////////////////////////////////////////////////////////////

void train() {
    float *train_input;
    int *train_labels;
    float *test_input;
    float *test_input_T;
    int *test_labels;

    float *W1;  // layer 1 dim x input dim
    float *B1;  // layer 1 dim x 1
    float *W2;  // layer 2 dim x layer 1 dim
    float *B2;  // layer 2 dim x 1

    float *Z1;  // layer 1 dim x batch size
    float *A1;  // layer 1 dim x batch size
    float *Z2;  // layer 2 dim x batch size
    float *A2;  // layer 2 dim x batch size

    float *Z1_test;  // layer 1 dim x test set size
    float *A1_test;  // layer 1 dim x test set size
    float *Z2_test;  // layer 2 dim x test set size
    float *A2_test;  // layer 2 dim x test set size

    int *train_epoch_indices;  // 1 x training set size
    float *X;                  // input dim x batch size
    int *Y;                    // 1 x batch size
    float *one_hot_Y;          // layer 2 dim x batch size
    
    float *dW1;  // layer 1 dim x input dim
    float *dB1;  // layer 1 dim x 1
    float *dW2;  // layer 2 dim x layer 1 dim
    float *dB2;  // layer 2 dim x 1

    train_input = malloc(TRAIN_SET_SIZE * INPUT_DIM * sizeof(float));
    train_labels = malloc(TRAIN_SET_SIZE * sizeof(int));
    test_input = malloc(TEST_SET_SIZE * INPUT_DIM * sizeof(float));
    test_input_T = malloc(INPUT_DIM * TEST_SET_SIZE * sizeof(float));
    test_labels = malloc(TEST_SET_SIZE *sizeof(int));

    W1 = malloc(LAYER_1_DIM * INPUT_DIM * sizeof(float));
    B1 = malloc(LAYER_1_DIM * sizeof(float));
    W2 = malloc(LAYER_2_DIM * LAYER_1_DIM * sizeof(float));
    B2 = malloc(LAYER_2_DIM * sizeof(float));

    Z1 = malloc(LAYER_1_DIM * BATCH_SIZE * sizeof(float));
    A1 = malloc(LAYER_1_DIM * BATCH_SIZE * sizeof(float));
    Z2 = malloc(LAYER_2_DIM * BATCH_SIZE * sizeof(float));
    A2 = malloc(LAYER_2_DIM * BATCH_SIZE * sizeof(float));

    Z1_test = malloc(LAYER_1_DIM * TEST_SET_SIZE * sizeof(float));
    A1_test = malloc(LAYER_1_DIM * TEST_SET_SIZE * sizeof(float));
    Z2_test = malloc(LAYER_2_DIM * TEST_SET_SIZE * sizeof(float));
    A2_test = malloc(LAYER_2_DIM * TEST_SET_SIZE * sizeof(float));


    train_epoch_indices = malloc(TRAIN_SET_SIZE * sizeof(int));
    X = malloc(INPUT_DIM * BATCH_SIZE * sizeof(float));
    Y = malloc(BATCH_SIZE * sizeof(int));
    one_hot_Y = malloc(LAYER_2_DIM * BATCH_SIZE * sizeof(float));

    dW1 = malloc(LAYER_1_DIM * INPUT_DIM * sizeof(float));
    dB1 = malloc(LAYER_1_DIM * sizeof(float));
    dW2 = malloc(LAYER_2_DIM * LAYER_1_DIM * sizeof(float));
    dB2 = malloc(LAYER_2_DIM * sizeof(float));
    
    // load in data into training and validation sets
    load_mnist(train_input, train_labels,
               test_input, test_labels);
    transpose(test_input, TEST_SET_SIZE, INPUT_DIM, test_input_T);

    // initialize training batch indices
    for (int i = 0; i < TRAIN_SET_SIZE; i++) {
        train_epoch_indices[i] = i;
    }

    // initialize parameters
    init_weights_he(W1, LAYER_1_DIM, INPUT_DIM);
    init_biases(B1, LAYER_1_DIM);
    init_weights_xavier(W2, LAYER_2_DIM, LAYER_1_DIM);
    init_biases(B2, LAYER_2_DIM);

    int batch_start_idx = 0;
    float train_accuracy;
    float test_accuracy;

    // gradient descent (batched)
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // per epoch
        if (i % EPOCH_ITERS == 0 ) {
            // shuffle training indices
            shuffle(train_epoch_indices, TRAIN_SET_SIZE);

            // reset position for `train_epoch_indices` reads
            batch_start_idx = 0;
        }

        // load batch training points
        for (int j = 0; j < BATCH_SIZE; j++) {
            int train_point_idx = train_epoch_indices[batch_start_idx + j];

            // load training image data
            for (int k = 0; k < INPUT_DIM; k++) {
                X[k * BATCH_SIZE + j] = train_input[train_point_idx * INPUT_DIM + k];
            }

            // load training labels
            Y[j] = train_labels[train_point_idx];
        }

        // forward prop
        batched_forward_prop(W1, B1, W2, B2,
                             X,
                             INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM,
                             BATCH_SIZE,
                             Z1, A1, Z2, A2);

        // back prop
        one_hot(Y, LAYER_2_DIM, BATCH_SIZE, one_hot_Y);
        batched_back_prop(Z1, A1, Z2, A2,
                          W1, W2,
                          X, one_hot_Y,
                          INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM,
                          BATCH_SIZE,
                          dW1, dB1, dW2, dB2);

        // update params
        update_params(W1, B1, W2, B2,
                      dW1, dB1, dW2, dB2,
                      LEARNING_RATE,
                      INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM);

        // accuracy update
        if (i % TRAIN_UPDATE_FREQ == 0) {
            printf("Epoch: %d\n", i / EPOCH_ITERS);
            printf("Iteration: %d\n", i);
            
            // calculate and print training accuracy
            train_accuracy = get_accuracy(A2, Y, LAYER_2_DIM, BATCH_SIZE);
            printf("Training accuracy: %.4f\n", train_accuracy);

            // test accuracy update
            if (i % EPOCH_ITERS == 0) {
                // calculate and print test accuracy
                test_accuracy = get_test_accuracy(W1, B1, W2, B2,
                                                  test_input_T, test_labels,
                                                  INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM,
                                                  TEST_SET_SIZE,
                                                  Z1_test, A1_test, Z2_test, A2_test);
            }
            printf("Most recent test accuracy: %.4f\n", test_accuracy);

            printf("\n");
        }

        batch_start_idx += BATCH_SIZE;
    }

    test_accuracy = get_test_accuracy(W1, B1, W2, B2,
                                       test_input_T, test_labels,
                                       INPUT_DIM, LAYER_1_DIM, LAYER_2_DIM,
                                       TEST_SET_SIZE,
                                       Z1_test, A1_test, Z2_test, A2_test);
    printf("Final test accuracy: %.4f\n", test_accuracy);

    // free memory

    free(train_input);
    free(train_labels);
    free(test_input);
    free(test_input_T);
    free(test_labels);

    free(W1);
    free(B1);
    free(W2);
    free(B2);

    free(Z1);
    free(A1);
    free(Z2);
    free(A2);

    free(Z1_test);
    free(A1_test);
    free(Z2_test);
    free(A2_test);

    free(train_epoch_indices);
    free(X);
    free(Y);
    free(one_hot_Y);

    free(dW1);
    free(dB1);
    free(dW2);
    free(dB2);
}
