/**
 * model.c - Manages an MLP model's high-level data
 */

#include "model.h"

#include <stdlib.h>

void new_MLP_MNIST(int num_hidden_layers, int *hidden_layer_dims, int batch_size, Model *model) {

    int num_total_layers = num_hidden_layers + 1;  // includes output layer

    model->train_data = malloc(sizeof(Dataset));
    model->test_data = malloc(sizeof(Dataset));

    init_data_MNIST(model->train_data, model->test_data,
                    &(model->tr_s_s), &(model->te_s_s));

    // set hyperparameters (non-exhaustive)

    model->i_d = 784;
    model->o_d = 10;
    model->n_h_l = num_hidden_layers;
    model->h_l_d = hidden_layer_dims;

    // allocate memory for top-level arrays for model
    model->Ws = malloc(num_total_layers * sizeof(float *));
    model->Bs = malloc(num_total_layers * sizeof(float *));
    model->Zs = malloc(num_total_layers * sizeof(float *));
    model->As = malloc(num_total_layers * sizeof(float *));
    model->Zs_test = malloc(num_total_layers * sizeof(float *));
    model->As_test = malloc(num_total_layers * sizeof(float *));
    model->dWs = malloc(num_total_layers * sizeof(float *));
    model->dBs = malloc(num_total_layers * sizeof(float *));
    model->dZs = malloc(num_total_layers * sizeof(float *));

    int output_layer_fan_in = (model->n_h_l == 0) ? model->i_d : model->h_l_d[model->n_h_l-1];

    // allocate memory for weights for all layers
    for (int i = 0; i < model->n_h_l; i++) {
        int fan_in = (i == 0) ? model->i_d : model->h_l_d[i-1];

        model->Ws[i] = malloc(model->h_l_d[i] * fan_in * sizeof(float));  // allocate hidden weight matrices
    }
    model->Ws[model->n_h_l] = malloc(model->o_d
                                     * output_layer_fan_in
                                     * sizeof(float));                    // allocate last weight matrix

    // allocate memory for biases for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->Bs[i] = malloc(model->h_l_d[i] * sizeof(float));    // allocate hidden bias vectors
    model->Bs[model->n_h_l] = malloc(model->o_d * sizeof(float));  // allocate last bias vector

    // allocate memory for training pre-activations for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->Zs[i] = malloc(model->h_l_d[i] * batch_size * sizeof(float));    // allocate hidden training pre-activation matrices
    model->Zs[model->n_h_l] = malloc(model->o_d * batch_size * sizeof(float));  // allocate last training pre-activation matrix

    // allocate memory for training activations for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->As[i] = malloc(model->h_l_d[i] * batch_size * sizeof(float));    // allocate hidden training activation matrices
    model->As[model->n_h_l] = malloc(model->o_d * batch_size * sizeof(float));  // allocate last training activation matrix

    // allocate memory for test pre-activations for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->Zs_test[i] = malloc(model->h_l_d[i]
                                   * model->te_s_s
                                   * sizeof(float));             // allocate hidden test pre-activation matrices
    model->Zs_test[model->n_h_l] = malloc(model->o_d
                                          * model->te_s_s
                                          * sizeof(float));      // allocate last test pre-activation matrix

    // allocate memory for test activations for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->As_test[i] = malloc(model->h_l_d[i]
                                   * model->te_s_s
                                   * sizeof(float));             // allocate hidden test activation matrices
    model->As_test[model->n_h_l] = malloc(model->o_d
                                          * model->te_s_s
                                          * sizeof(float));      // allocate last test activation matrix

    // allocate memory for weight gradients for all layers
    for (int i = 0; i < model->n_h_l; i++) {
        int fan_in = (i == 0) ? model->i_d : model->h_l_d[i-1];

        model->dWs[i] = malloc(model->h_l_d[i] * fan_in * sizeof(float));  // allocate hidden weight gradient matrices
    }
    model->dWs[model->n_h_l] = malloc(model->o_d
                                      * output_layer_fan_in
                                      * sizeof(float));                    // allocate last weight gradient matrix

    // allocate memory for bias gradients for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->dBs[i] = malloc(model->h_l_d[i] * sizeof(float));    // allocate hidden bias gradient vectors
    model->dBs[model->n_h_l] = malloc(model->o_d * sizeof(float));  // allocate last bias gradient vector

    // allocate memory for pre-activation gradients for all layers
    for (int i = 0; i < model->n_h_l; i++)
        model->dZs[i] = malloc(model->h_l_d[i] * batch_size * sizeof(float));    // allocate hidden pre-activation gradient matrices
    model->dZs[model->n_h_l] = malloc(model->o_d * batch_size * sizeof(float));  // allocate last pre-activation gradient matrix

    // allocate memory for training epoch indices and initialize
    model->train_epoch_indices = malloc(model->tr_s_s * sizeof(int));
    for (int i = 0; i < model->tr_s_s; i++) {
        model->train_epoch_indices[i] = i;
    }

}

void free_MLP_MNIST(Model *model) {
    int num_total_layers = model->n_h_l + 1;  // includes output layer

    free_data_set(model->train_data);
    free_data_set(model->test_data);

    // free memory for weights for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->Ws[i]);

    // free memory for biases for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->Bs[i]);

    // free memory for training pre-activations for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->Zs[i]);

    // free memory for training activations for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->As[i]);

    // free memory for test pre-activations for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->Zs_test[i]);

    // free memory for test activations for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->As_test[i]);

    // free memory for weight gradients for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->dWs[i]);

    // free memory for bias gradients for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->dBs[i]);

    // free memory for pre-activation gradients for all layers
    for (int i = 0; i < num_total_layers; i++)
        free(model->dZs[i]);

    // free memory for top-level arrays for model
    free(model->Ws);
    free(model->Bs);
    free(model->Zs);
    free(model->As);
    free(model->Zs_test);
    free(model->As_test);
    free(model->dWs);
    free(model->dBs);
    free(model->dZs);

    // free memory for training epoch indices
    free(model->train_epoch_indices);
}
