/**
 * train.c - Manages an MLP training run
 * 
 * Main training logic
 */

#include "train.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "mat_ops.h"
#include "data_ops.h"
#include "model.h"
#include "config.h"

typedef enum accuracy_type {
    TRAIN,
    TEST
} Accuracy_type;

/**
 * @brief Initializes hidden layer weights under He scheme
 * 
 * - For layers with ReLU activation
 * 
 * - Uses uniform sampling for simpler implementation
 * 
 * @param[in, out] model Model whose hidden layers' weights are to be initialized
 */
static void init_weights_he(Model *model) {
    for (int L = 0; L < model->n_h_l; L++) {
        int fan_in = (L == 0) ? model->i_d : model->h_l_d[L-1];
        int fan_out = model->h_l_d[L];

        // He initialization uniform sampling boundary
        float b = sqrtf(6.f / fan_in);

        float norm_factor = (float)(1.f / (double)RAND_MAX) * (2 * b);

        // sets each element of weight matrix
        for (int i = 0; i < fan_out; i++) {
            for (int j = 0; j < fan_in; j++) {
                model->Ws[L][i * fan_in + j] = rand() * norm_factor - b;  // constrains value to [-b, b]
            }
        }
    }
}

/**
 * @brief Initializes output layer weights under Xavier scheme
 * 
 * - For layers with softmax activation
 * 
 * - Uses uniform sampling for simpler implementation
 * 
 * @param[in, out] model Model whose output layer weights are to be initialized
 */
static void init_weights_xavier(Model *model) {
    int fan_in = (model->n_h_l == 0) ? model->i_d : model->h_l_d[model->n_h_l-1];
    int fan_out = model->o_d;

    // Xavier initialization uniform sampling boundary
    float b = sqrtf(6.f / (fan_in + fan_out));

    float norm_factor = (float)(1.f / (double)RAND_MAX) * (2 * b);

    // sets each element of weight matrix
    for (int i = 0; i < fan_out; i++) {
        for (int j = 0; j < fan_in; j++) {
            model->Ws[model->n_h_l][i * fan_in + j] = rand() * norm_factor - b;  // constrains value to [-b, b]
        }
    }
}

/**
 * @brief Initializes all layers' biases
 * 
 * - Initializes biases to 0
 * 
 * - Zero-initialization works well for layers with ReLU or softmax activation
 * 
 * @param[in, out] model Model whose biases are to be initialized
 */
static void init_biases(Model *model) {
    int num_total_layers = model->n_h_l + 1;

    for (int L = 0; L < num_total_layers; L++) {
        int fan_out = (L == num_total_layers - 1) ? model->o_d : model->h_l_d[L];

        // zero-initialize all biases
        memset(model->Bs[L], 0, fan_out * sizeof(float));
    }
}

/**
 * @brief Does a mini-batched pass of forward propagation
 * 
 * @param[in, out] model      Model on which to run a pass of forward propagation on
 * @param[in]      batch_X    Batch of training inputs
 * @param[in]      batch_size Size of batch
 */
static void batch_forward_prop(Model *model, float *batch_X, int batch_size) {
    int num_total_layers = model->n_h_l + 1;

    // for each hidden layer
    for (int L = 0; L < num_total_layers; L++) {
        int fan_in = (L == 0) ? model->i_d : model->h_l_d[L-1];
        int fan_out = (L == num_total_layers - 1) ? model->o_d : model->h_l_d[L];

        float *layer_inputs = (L == 0) ? batch_X : model->As[L-1];

        // compute pre-activations for layer
        mat_mat_mul(model->Ws[L], layer_inputs,
                    fan_out, fan_in, batch_size,
                    model->Zs[L]);
        mat_vec_add_broadcast(model->Zs[L], model->Bs[L],
                              fan_out,
                              batch_size,
                              model->Zs[L]);

        // compute activations for layer
        if (L != num_total_layers - 1) {
            // for hidden layer
            relu(model->Zs[L],
                 fan_out, batch_size,
                 model->As[L]);
        } else {
            // for output layer
            softmax(model->Zs[model->n_h_l],
                    fan_out, batch_size,
                    model->As[model->n_h_l]);
        }
    }
}

/**
 * @brief Does a mini-batched pass of back propagation
 * 
 * @param[in, out] model           Model on which to run a pass of back propagation on
 * @param[in]      batch_X         Batch of training inputs
 * @param[in]      one_hot_batch_Y Batch of one hot training labels
 * @param[in]      batch_size      Size of batch
 */
static void batch_back_prop(Model *model, float *batch_X, float *one_hot_batch_Y, int batch_size) {
    float batch_size_inv = 1.f / batch_size;

    // compute output layer pre-activations gradient
    mat_sub(model->As[model->n_h_l], one_hot_batch_Y,
            model->o_d, batch_size,
            model->dZs[model->n_h_l]);

    // for each layer (including output layer)
    for (int L = model->n_h_l; L >=0; L--) {
        int fan_in = (L == 0) ? model->i_d : model->h_l_d[L-1];
        int fan_out = (L == model->n_h_l) ? model->o_d : model->h_l_d[L];

        float *layer_inputs = (L == 0) ? batch_X : model->As[L-1];

        // compute weights gradient for layer

        float *layer_inputs_T = malloc(fan_in * batch_size * sizeof(float));

        transpose(layer_inputs,
                  fan_in, batch_size,
                  layer_inputs_T);
        mat_mat_mul(model->dZs[L], layer_inputs_T,
                    fan_out, batch_size, fan_in,
                    model->dWs[L]);
        scal_mat_mul(batch_size_inv, model->dWs[L],
                     fan_out, fan_in,
                     model->dWs[L]);

        free(layer_inputs_T);

        // compute bias gradient for layer
        row_sum(model->dZs[L],
                fan_out, batch_size,
                model->dBs[L]);
        scal_mat_mul(batch_size_inv, model->dBs[L],
                     fan_out, 1,
                     model->dBs[L]);

        // compute pre-activations gradient for previous layer (hidden layers)
        if (L > 0) {
            float *W_T = malloc(fan_out * fan_in * sizeof(float));
            float *Z_p = malloc(fan_in * batch_size * sizeof(float));

            transpose(model->Ws[L],
                      fan_out, fan_in,
                      W_T);
            mat_mat_mul(W_T, model->dZs[L],
                        fan_in, fan_out, batch_size,
                        model->dZs[L-1]);
            relu_deriv(model->Zs[L-1],
                       fan_in, batch_size,
                       Z_p);
            hadamard_product(model->dZs[L-1], Z_p,
                             fan_in, batch_size,
                             model->dZs[L-1]);

            free(W_T);
            free(Z_p);
        }
    }
}

/**
 * @brief Updates parameters with gradient
 * 
 * @param[in, out] model         Model whose parameters are to be update
 * @param[in]      learning_rate Learning rate
 */
static void update_params(Model *model, float learning_rate) {
    int num_total_layers = model->n_h_l + 1;

    // for each layer
    for (int L = 0; L < num_total_layers; L++) {
        int fan_in = (L == 0) ? model->i_d : model->h_l_d[L-1];
        int fan_out = (L == model->n_h_l) ? model->o_d : model->h_l_d[L];

        // update weights of layer
        scal_mat_mul(learning_rate, model->dWs[L],
                     fan_out, fan_in,
                     model->dWs[L]);
        mat_sub(model->Ws[L], model->dWs[L],
                fan_out, fan_in,
                model->Ws[L]);

        // update biases of layer
        scal_mat_mul(learning_rate, model->dBs[L],
                     fan_out, 1,
                     model->dBs[L]);
        mat_sub(model->Bs[L], model->dBs[L],
                fan_out, 1,
                model->Bs[L]);
    }
}

/**
 * @brief Computes accuracy of a group of data points (batch or set)
 * 
 * @param[in] model      Model whose accuracy is to be evaluated
 * @param[in] batch_Y    Labels of the data points
 * @param[in] group_size Size of the group
 * @param[in] type       Type of accuracy computation (training or test)
 * 
 * @return Accuracy of model on group
 */
static float get_accuracy(Model *model, int *batch_Y, int group_size, Accuracy_type type) {
    float *predictions = (type == TRAIN) ? model->As[model->n_h_l] : model->As_test[model->n_h_l];

    float max_prob;
    int max_label;

    int num_correct = 0;

    // for each data point
    for (int i = 0; i < group_size; i++) {
        max_prob = predictions[i];
        max_label = 0;

        // for each output category
        for (int j = 1; j < model->o_d; j++) {
            bool is_new_max = predictions[j * group_size + i] > max_prob;
            max_prob = is_new_max ? predictions[j * group_size + i] : max_prob;
            max_label = is_new_max ? j : max_label;
        }

        if (max_label == batch_Y[i])
            num_correct++;
    }

    return (float)num_correct / group_size;
}

/**
 * @brief Does a full-batched pass of forward propagation on the test set
 * 
 * @param[in, out] model      Model on which to run a pass of forward propagation on
 * @param[in]      X          Training set inputs
 */
static void test_forward_prop(Model *model, float *X) {
    int num_total_layers = model->n_h_l + 1;

    // for each hidden layer
    for (int L = 0; L < num_total_layers; L++) {
        int fan_in = (L == 0) ? model->i_d : model->h_l_d[L-1];
        int fan_out = (L == num_total_layers - 1) ? model->o_d : model->h_l_d[L];

        float *layer_inputs = (L == 0) ? X : model->As_test[L-1];

        // compute pre-activations for layer
        mat_mat_mul(model->Ws[L], layer_inputs,
                    fan_out, fan_in, model->te_s_s,
                    model->Zs_test[L]);
        mat_vec_add_broadcast(model->Zs_test[L], model->Bs[L],
                              fan_out, model->te_s_s,
                              model->Zs_test[L]);

        // compute activations for layer
        if (L != num_total_layers - 1) {
            // for hidden layer
            relu(model->Zs_test[L],
                 fan_out, model->te_s_s,
                 model->As_test[L]);
        } else {
            // for output layer
            softmax(model->Zs_test[model->n_h_l],
                    fan_out, model->te_s_s,
                    model->As_test[model->n_h_l]);
        }
    }
}

/**
 * @brief Computes test accuracy on a test set
 * 
 * - Runs forward propagation for entire test set, then computes accuracy in
 * the same way as training accuracy
 * 
 * @param[in] model Model whose test accuracy is to be evaluated
 * 
 * @return Accuracy of model on test set
 */
static float get_test_accuracy(Model *model) {

    // run predictions on test inputs
    test_forward_prop(model, model->test_data->input);

    // calculate test loss on test predictions
    return get_accuracy(model, model->test_data->labels, model->te_s_s, TEST);

}

/**
 * @brief Clears terminal and prints header for model
 * 
 * @param[in] model         Model whose header is to be printed
 * @param[in] num_epochs    Number of epochs
 * @param[in] batch_size    Size of batch
 * @param[in] learning_rate Learning rate
 */
static void print_model_header(Model *model,
                               int num_epochs, int batch_size, float learning_rate) {
    // \e[1;1H moves the cursor to row 1, column 1
    // \e[2J clears the entire screen
    printf("\e[1;1H\e[2J");

    printf("--------------------------------------------------------------\n");
    printf("| "
               ANSI_COLOR_RED "S"
               ANSI_COLOR_YELLOW "wift "
               ANSI_COLOR_RED "L"
               ANSI_COLOR_YELLOW "ogic "
               ANSI_COLOR_RED "O"
               ANSI_COLOR_YELLOW"ptimized "
               ANSI_COLOR_RED "P"
               ANSI_COLOR_YELLOW"latform"
               ANSI_COLOR_RESET "                             |\n");
    printf("--------------------------------------------------------------\n");
    printf("\n");

#if DATASET == 0
    printf("Dataset: MNIST\n");
#elif DATASET == 1
    printf("Dataset: Fashion MNIST\n");
#endif
    printf("Architecture: %d -> ", model->i_d);
    for (int L = 0; L < model->n_h_l; L++) printf("%d -> ", model->h_l_d[L]);
    printf("%d\n", model->o_d);
    printf("\n");

    printf("Number of epochs: %d\n", num_epochs);
    printf("Batch size: %d\n", batch_size);
    printf("Learning rate: %.2f\n", learning_rate);
    printf("\n");

    printf("--------------------------------------------------------------\n");
    printf("\n");
}

// note: training set inputs are loaded sample-major, so it is transposed when writing into batch array
//       test set inputs are loaded feature-major, so it is used for test accuracy without transposition
void train_MNIST(int num_hidden_layers, int *hidden_layer_dims,
                 int num_epochs, int batch_size, float learning_rate,
                 bool rand_seed_rand) {

    if (rand_seed_rand)
        srand(time(NULL));
    else
        srand(42);

    Model MLP;
    // note: training input is loaded sample-major
    new_MLP_MNIST(num_hidden_layers, hidden_layer_dims, batch_size, &MLP);

    int epoch_iters = MLP.tr_s_s / batch_size;
    int num_iterations = epoch_iters * num_epochs;


    float *batch_X = malloc(MLP.i_d * batch_size * sizeof(float));
    int *batch_Y = malloc(batch_size * sizeof(int));
    float *one_hot_batch_Y = malloc(MLP.o_d * batch_size * sizeof(float));

    // initialize parameters
    init_weights_he(&MLP);
    init_weights_xavier(&MLP);
    init_biases(&MLP);

    int batch_start_idx = 0;
    float train_accuracy;
    float test_accuracy = 0.f;

    // begin training run timing
    struct timespec start;
    timespec_get(&start, TIME_UTC);
    struct timespec end;
    struct timespec last_update_time;
    timespec_get(&last_update_time, TIME_UTC);
    double total_elapsed;

    // gradient descent (batched)
    // note: leaves out training samples at the end of an epoch that don't fit into a batch
    for (int i = 0; i < num_iterations; i++) {
        // per epoch
        if (i % epoch_iters == 0 ) {
            // shuffle training indices
            shuffle(MLP.train_epoch_indices, MLP.tr_s_s);

            // reset position for `train_epoch_indices` reads
            batch_start_idx = 0;
        }

        // load batch training points
        for (int j = 0; j < batch_size; j++) {
            int train_point_idx = MLP.train_epoch_indices[batch_start_idx + j];

            // load training image data
            for (int k = 0; k < MLP.i_d; k++) {
                batch_X[k * batch_size + j] = MLP.train_data->input[train_point_idx * MLP.i_d + k];
            }

            // load training labels
            batch_Y[j] = MLP.train_data->labels[train_point_idx];
        }

        // forward prop
        batch_forward_prop(&MLP, batch_X, batch_size);

        // back prop
        one_hot(batch_Y,
                MLP.o_d, batch_size,
                one_hot_batch_Y);
        batch_back_prop(&MLP, batch_X, one_hot_batch_Y, batch_size);

        // update params
        update_params(&MLP, learning_rate);

        // test accuracy update
        if (i % epoch_iters == 0) {
            // calculate test accuracy
            test_accuracy = get_test_accuracy(&MLP);
        }

        // calculate elapsed time since last update (milliseconds)
        timespec_get(&end, TIME_UTC);
        int update_elapsed = (end.tv_sec * 1e3 + end.tv_nsec / 1e6)
                              - (last_update_time.tv_sec * 1e3 + last_update_time.tv_nsec / 1e6);

        // accuracy update
        if (update_elapsed >= TRAIN_UPDATE_FREQ) {
            // calculate training accuracy
            // note: evaluates on predictions from previous iteration before latest parameter update
            train_accuracy = get_accuracy(&MLP, batch_Y, batch_size, TRAIN);

            print_model_header(&MLP,
                               num_epochs, batch_size, learning_rate);

            printf("Epoch: %d\n", i / epoch_iters);
            printf("Iteration: %d\n", i);
            printf("Training accuracy: %.4f\n", train_accuracy);
            printf("Most recent test accuracy: %.4f\n", test_accuracy);
            printf("\n");

            printf("--------------------------------------------------------------\n");
            printf("\n");

            // calculate total elapsed time since start of training run
            total_elapsed = (float)(end.tv_sec - start.tv_sec) + (float)(end.tv_nsec - start.tv_nsec) / 1e9;

            printf("Elapsed time: %.1f seconds\n", total_elapsed);
            printf("Training run progress: %2.f%%\n", (float)i / num_iterations * 100);

            timespec_get(&last_update_time, TIME_UTC);
        }

        batch_start_idx += batch_size;
    }

    // calculate and print final test accuracy
    test_accuracy = get_test_accuracy(&MLP);
    print_model_header(&MLP,
                       num_epochs, batch_size, learning_rate);
    printf("Final test accuracy: %.4f\n", test_accuracy);
    printf("\n");

    // end training run timing
    total_elapsed = (float)(end.tv_sec - start.tv_sec) + (float)(end.tv_nsec - start.tv_nsec) / 1e9;

    // calculate and print elapsed execution time for training run
    printf("Total elapsed time for training run: %.2f seconds\n", total_elapsed);
    printf("\n");

    // free memory

    free(batch_X);
    free(batch_Y);
    free(one_hot_batch_Y);

    free_MLP_MNIST(&MLP);

}
