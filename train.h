/**
 * train.h - Manages an MLP training run
 * 
 * Main training logic
 */

#include <stdbool.h>

#ifndef _TRAIN_H_
#define _TRAIN_H_

/**
 * @brief Trains a basic MLP for MNIST end-to-end
 * 
 * @param[in] num_hidden_layers Number of hidden layers
 * @param[in] hidden_layer_dims Dimensions of hidden layers
 * @param[in] num_epochs        Number of epochs
 * @param[in] batch_size        Size of batch
 * @param[in] learning_rate     Learning rate
 * @param[in] seed_rand         Whether to seed the RNG for parameter initialization
 */
void train_MNIST(int num_hidden_layers, int *hidden_layer_dims,
                 int num_epochs, int batch_size, float learning_rate,
                 bool seed_rand);

#endif
