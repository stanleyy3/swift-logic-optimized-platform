/**
 * config.h - Configuration for program
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_

// whether to randomize seed
#define RAND_SEED_RAND false

// dataset to train on (0 - MNIST, 1 - Fashion MNIST)
#define DATASET 0

// number of milliseconds per update of training status
#define TRAIN_UPDATE_FREQ 100

// matrix multiplication tile dimension
#define TILE_DIM 64

// terminal output color escape sequences
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RED    "\x1b[31m"
#define ANSI_COLOR_RESET  "\x1b[0m"

#endif
