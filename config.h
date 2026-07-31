/**
 * config.h - Configuration for program
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_

// whether to randomize seed
#define RAND_SEED_RAND false

// dataset to train on (0 - MNIST)
// note: this branch ships only MNIST, and data_ops.c holds those paths directly
#define DATASET 0

// accuracy points published per epoch
// note: this is the sampling rate of the training curve, not the redraw rate;
//       the history buffer decimates itself, so a long run costs no more than a
//       short one to store or to draw
#define TRAIN_POINTS_PER_EPOCH 200

// dashboard frames per second
// note: the UI redraws at this rate no matter how long a single training
//       iteration takes, and only the cells that changed are written - which is
//       what matters here, where the matrix multiply is untiled and
//       single-threaded and one iteration can take a long time
#define TUI_FPS 20

// smallest terminal the UI will draw a screen in
#define TUI_MIN_COLS 60
#define TUI_MIN_ROWS 18

// force ASCII curves instead of braille
// note: braille is used automatically when the locale looks like UTF-8
#define TUI_ASCII_FALLBACK 0

#endif
