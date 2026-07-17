/**
 * main.c - Interface to launch MLP training runs
 * 
 * Interfacing script for the user
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "train.h"

#define RAND_SEED_RAND false

/**
 * @brief Gets hyperparameters from user through terminal
 * 
 * @param[out] num_hidden_layers Number of hidden layers
 * @param[out] hidden_layer_dims Dimensions of hidden layers
 * @param[out] num_epochs        Number of epochs
 * @param[out] batch_size        Size of batch
 * @param[out] learning_rate     Learning rate
 */
static void get_hyperparams(int *num_hidden_layers, int **hidden_layer_dims,
                            int *num_epochs, int *batch_size,
                            float *learning_rate) {
    int ret;

    // get number of hidden layers
    do {
        printf("Number of hidden layers (0-3): ");

        ret = scanf("%d", num_hidden_layers);
        if (ret == EOF) exit(0);     // Ctrl-D pressed
        if (ret != 1) scanf("%*s");  // flush an invalid token
    } while (ret != 1 || *num_hidden_layers < 0 || *num_hidden_layers > 3);
    printf("\n");

    // allocate memory for hidden layer dimensions array
    *hidden_layer_dims = malloc(*num_hidden_layers * sizeof(int));

    // get hidden layer dimensions
    for (int i = 0; i < *num_hidden_layers; i++) {
        do {
            printf("Hidden layer %d dimension (16, 32, 64, 128, 256, 512): ", i + 1);

            ret = scanf("%d", &(*hidden_layer_dims)[i]);
            if (ret == EOF) exit(0);     // Ctrl-D pressed
            if (ret != 1) scanf("%*s");  // flush an invalid token
        } while (ret != 1
                 || !((*hidden_layer_dims)[i] == 16
                      || (*hidden_layer_dims)[i] == 32
                      || (*hidden_layer_dims)[i] == 64
                      || (*hidden_layer_dims)[i] == 128
                      || (*hidden_layer_dims)[i] == 256
                      || (*hidden_layer_dims)[i] == 512));
    }
    if (*num_hidden_layers > 0) printf("\n");

    // get number of epochs
    do {
        printf("Number of epochs (1-40): ");

        ret = scanf("%d", num_epochs);
        if (ret == EOF) exit(0);     // Ctrl-D pressed
        if (ret != 1) scanf("%*s");  // flush an invalid token
    } while (ret != 1 || *num_epochs < 1 || *num_epochs > 40);
    printf("\n");

    // get batch size
    do {
        printf("Batch size (100-1000): ");

        ret = scanf("%d", batch_size);
        if (ret == EOF) exit(0);     // Ctrl-D pressed
        if (ret != 1) scanf("%*s");  // flush an invalid token
    } while (ret != 1 || *batch_size < 100 || *batch_size > 1000);
    printf("\n");

    // get learning rate
    do {
        printf("Learning rate (0.01, 0.05, 0.10): ");

        ret = scanf("%f", learning_rate);
        if (ret == EOF) exit(0);     // Ctrl-D pressed
        if (ret != 1) scanf("%*s");  // flush an invalid token
    } while (ret != 1
             || !(*learning_rate == 0.01f
                  || *learning_rate == 0.05f
                  || *learning_rate == 0.1f));
    printf("\n");
}

int main(int argc, char *argv[]) {
    // hyperparameters

    int num_hidden_layers;
    int *hidden_layer_dims;

    int num_epochs;
    int batch_size;
    float learning_rate;

    int user_continue = 1;

    while (user_continue) {
        // \e[1;1H moves the cursor to row 1, column 1
        // \e[2J clears the entire screen
        printf("\e[1;1H\e[2J");

        printf("--------------------------------------------------------------\n");
        printf("| MLP Training                                               |\n");
        printf("| accelerated by Swift Logic Optimized Platform (SLOP)       |\n");
        printf("--------------------------------------------------------------\n");
        printf("\n");

        // get hyperparameters
        get_hyperparams(&num_hidden_layers, &hidden_layer_dims,
                        &num_epochs, &batch_size,
                        &learning_rate);

        printf("Launching training run...\n");

        time_t start = time(NULL);

        // launch a training run
        train_MNIST(num_hidden_layers, hidden_layer_dims,
                    num_epochs, batch_size, learning_rate,
                    RAND_SEED_RAND);

        time_t end = time(NULL);

        // calculate and print elapsed execution time for training run
        printf("Elapsed time: %.0f seconds\n", difftime(end, start));
        printf("\n");

        free(hidden_layer_dims);

        // get whether to continue
        int ret;
        do {
            printf("Would you like to launch another training run (1 (for yes), 0 (for no)): ");

            ret = scanf("%d", &user_continue);
            if (ret == EOF) exit(0);     // Ctrl-D pressed
            if (ret != 1) scanf("%*s");  // flush an invalid token
        } while (ret != 1 || user_continue < 0 || user_continue > 1);
        printf("\n");

    }

    return 0;
}
