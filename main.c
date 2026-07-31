/**
 * main.c - Interface to launch MLP training runs
 *
 * Interfacing script for the user
 */

#include <stdio.h>

#include "data_ops.h"
#include "train.h"
#include "tui/tui.h"

int main(void) {
    // the dataset paths are relative, and the loader's own error messages would
    // be swallowed by the alternate screen buffer, so check before taking over
    const char *missing = missing_data_file();
    if (missing) {
        fprintf(stderr, "Error: cannot open %s\n", missing);
        fprintf(stderr, "Run this from the project root, where the data directory is.\n");
        return 1;
    }

    if (!tui_init()) {
        fprintf(stderr, "Error: this program needs an interactive terminal.\n");
        return 1;
    }

    TrainConfig cfg;
    train_default_config(&cfg);

    // the form carries the previous run's settings over, so a second run is
    // usually a one-key change away
    while (tui_run_form(&cfg)) {
        if (tui_run_training(&cfg) == TUI_QUIT) break;
    }

    tui_shutdown();

    return 0;
}
