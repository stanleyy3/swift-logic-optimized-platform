/**
 * model.h - Manages an MLP model's high-level data
 */

#ifndef _MODEL_H_
#define _MODEL_H_

#include "data_ops.h"

typedef struct {
    Dataset *train_data;  // each data point is a column
    Dataset *test_data;   // each data point is a column
    int tr_s_s;           // training set size
    int te_s_s;           // test set size

    int i_d;     // (dimension of input)
    int o_d;     // (dimension of output)
    int n_h_l;   // (number of hidden layers)
    int *h_l_d;  // (hidden layer dimensions)

    float **Ws;  // (number of hidden layers + 1), layer N dim x layer N-1 dim
    float **Bs;  // (number of hidden layers + 1), layer N dim x 1

    float **Zs;       // (number of hidden layers + 1), layer N dim x batch size
    float **As;       // (number of hidden layers + 1), layer N dim x batch size
    float **Zs_test;  // (number of hidden layers + 1), layer N dim x test set size
    float **As_test;  // (number of hidden layers + 1), layer N dim x test set size

    float **dWs;  // (number of hidden layers + 1), layer N dim x layer N-1 dim
    float **dBs;  // (number of hidden layers + 1), layer N dim x 1
    
    float **dZs;  // (number of hidden layers + 1), layer N dim x batch size

    int *train_epoch_indices;  // 1 x training set size
} Model;

/**
 * @brief Initializes and allocates memory for an MLP for MNIST
 * 
 * - Note: training input is loaded sample-major
 * 
 * @param[in]  num_hidden_layers Number of hidden layers
 * @param[in]  hidden_layer_dims Dimensions of hidden layers
 * @param[in]  batch_size        Size of batch
 * @param[out] model             MLP to train
 */
void new_MLP_MNIST(int num_hidden_layers, int *hidden_layer_dims, int batch_size, Model *model);

/**
 * @brief Frees memory allocated for model
 * 
 * @param[in] model Model to be freed
 */
void free_MLP_MNIST(Model *model);

#endif
