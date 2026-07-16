/**
 * mlp.c - Script to train an MLP
 */

#include "mlp.h"

//#include <math.h>

int main(int argc, char *argv[]) {
    // hyperparameters

    /*
    int total_set_size = 60000;

    float train_set_split = 0.8f;
    // training set split + test set split = 1.0
    int train_set_size = floorf(train_set_split * total_set_size);
    int test_set_size = total_set_size - train_set_size;
    */

    int train_set_size = 60000;
    int test_set_size = 10000;

    int input_dim = 784;
    int layer_1_dim = 10;
    int layer_2_dim = 10;

    int num_epochs = 50;
    int batch_size = 1000;
    float learning_rate = 0.1f;

    int train_update_freq = 10;

    train(train_set_size, test_set_size,
          input_dim, layer_1_dim, layer_2_dim,
          num_epochs, batch_size, learning_rate,
          train_update_freq);

    return 0;
}
