/**
 * train.h - Manages an MLP training run
 * 
 * Main training logic
 */

#ifndef _TRAIN_H_
#define _TRAIN_H_

#include <stdbool.h>

/**
 * @brief Trains a basic MLP for MNIST end-to-end
 * 
 * - Gives training and test loss updates
 * 
 * @param[in] num_hidden_layers Number of hidden layers
 * @param[in] hidden_layer_dims Dimensions of hidden layers
 * @param[in] num_epochs        Number of epochs
 * @param[in] batch_size        Size of batch
 * @param[in] learning_rate     Learning rate
 * @param[in] rand_seed_rand    Whether to randomize the seed for RNG
 */
void train_MNIST(int num_hidden_layers, int *hidden_layer_dims,
                 int num_epochs, int batch_size, float learning_rate,
                 bool rand_seed_rand);

#endif
