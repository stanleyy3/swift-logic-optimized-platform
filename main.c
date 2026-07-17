/**
 * main.c - Interface to train an MLP
 * 
 * Interfacing script for the user
 */

#include <string.h>
#include <stdlib.h>

#include "train.h"

int main(int argc, char *argv[]) {
    // hyperparameters

    int num_hidden_layers = 1;
    int *hidden_layer_dims = malloc(num_hidden_layers * sizeof(int));
    memcpy(hidden_layer_dims, (int[]){10}, num_hidden_layers * sizeof(int));

    int num_epochs = 50;
    int batch_size = 1000;  // constrain to evenly divide training set size (60000 for MNIST)
    float learning_rate = 0.1f;

    bool rand_seed_rand = false;

    // launch a training run
    train_MNIST(num_hidden_layers, hidden_layer_dims,
                num_epochs, batch_size, learning_rate,
                rand_seed_rand);

    free(hidden_layer_dims);

    return 0;
}
