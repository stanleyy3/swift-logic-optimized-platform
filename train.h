/**
 * train.h - Manages an MLP training run
 *
 * Main training logic
 */

#ifndef _TRAIN_H_
#define _TRAIN_H_

#include <stdbool.h>

// most hidden layers a run may have
#define MAX_HIDDEN_LAYERS 3

// everything that describes one training run
// note: `hidden_layer_dims` is an embedded array, so a configuration can be
//       copied by assignment and needs no allocation
typedef struct {
    int num_hidden_layers;
    int hidden_layer_dims[MAX_HIDDEN_LAYERS];

    int num_epochs;
    int batch_size;
    float learning_rate;
} TrainConfig;

/**
 * @brief Fills in a known-good starting configuration
 *
 * @param[out] cfg Configuration to fill
 */
void train_default_config(TrainConfig *cfg);

/**
 * @brief Trains a basic MLP for MNIST end-to-end
 *
 * - Reports progress, accuracy history and completion through metrics.h; writes
 * nothing to the terminal, so it can run on a worker thread while the UI draws
 *
 * - Checks for a requested stop and for a pause once per iteration
 *
 * @param[in] cfg            Configuration for the run
 * @param[in] rand_seed_rand Whether to randomize the seed for RNG
 */
void train_MNIST(const TrainConfig *cfg, bool rand_seed_rand);

#endif
