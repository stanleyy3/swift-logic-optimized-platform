/**
 * mlp.h - Basic MLP for an arbitrary workload
 */

/**
 * @brief Trains a basic MLP for MNIST end-to-end
 * 
 * @param[in] train_set_size    Size of training set
 * @param[in] test_set_size     Size of test set
 * @param[in] input_dim         Dimension of input layer
 * @param[in] layer_1_dim       Dimension of layer 1
 * @param[in] layer_2_dim       Dimension of layer 2
 * @param[in] num_epochs        Number of epochs
 * @param[in] batch_size        Size of batch
 * @param[in] learning_rate     Learning rate
 * @param[in] train_update_freq Training update frequency
 */
void train(int train_set_size, int test_set_size,
           int input_dim, int layer_1_dim, int layer_2_dim,
           int num_epochs, int batch_size, float learning_rate,
           int train_update_freq);
