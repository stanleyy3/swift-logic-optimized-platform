/**
 * dash.c - Live training dashboard
 *
 * The training run happens on a worker thread; this file runs the UI on the main
 * thread, which owns the terminal outright. That split is what keeps the display
 * responsive under extreme hyperparameters: a single iteration of a
 * 784->512->512->10 model at batch size 1000 takes on the order of 100ms, and
 * the epoch-boundary test pass over 10,000 samples takes about a second, but the
 * UI keeps drawing at its own fixed rate throughout either of them.
 *
 * The only things crossing between the threads are the scalars and bounded
 * history in metrics.h - never the model's matrices, so the UI never reads
 * memory that the OpenMP regions are writing.
 */

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "../config.h"
#include "../metrics.h"
#include "../train.h"
#include "dash.h"
#include "plot.h"
#include "term.h"
#include "tui.h"

// milliseconds between dashboard frames
#define FRAME_MS (1000 / TUI_FPS)

// left margin of the body
#define BODY_COL 2

// gap between the stat line's fields
#define STAT_GAP 3

// the worker's own copy of the configuration, so it cannot outlive its owner
static TrainConfig worker_cfg;

static void *worker_main(void *arg) {
    (void)arg;

    train_MNIST(&worker_cfg, RAND_SEED_RAND);

    return NULL;
}

/**
 * @brief Number of decimal digits needed to print a non-negative number
 *
 * - Used to keep the stat line's field widths constant, so the numbers do not
 * jitter sideways as they grow
 */
static int digits(int v) {
    int n = 1;

    while (v >= 10) {
        v /= 10;
        n++;
    }

    return n;
}

////////////////////////////////////////////////////////////////////////////////
// DRAWING
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Draws the epoch, iteration, elapsed time and estimated time remaining
 */
static void draw_progress_stats(int row, const TrainSnapshot *snap, const TrainConfig *cfg) {
    char elapsed[32];
    char eta[32];

    tui_format_duration(elapsed, sizeof(elapsed), snap->elapsed_s);

    int iter_width = digits(snap->total_iters > 0 ? snap->total_iters : 1);

    int col = BODY_COL;

    col = term_printf(row, col, T_DEFAULT, T_DIM, "Epoch ");
    col = term_printf(row, col, T_DEFAULT, T_BOLD, "%*d/%d",
                      digits(cfg->num_epochs), snap->epoch + 1, cfg->num_epochs);
    col += STAT_GAP;

    col = term_printf(row, col, T_DEFAULT, T_DIM, "Iteration ");
    if (snap->total_iters > 0) {
        col = term_printf(row, col, T_DEFAULT, T_BOLD, "%*d/%d",
                          iter_width, snap->iter, snap->total_iters);
    } else {
        // the dataset is still loading, so the run's length is not known yet
        col = term_printf(row, col, T_DEFAULT, T_BOLD, "--");
    }
    col += STAT_GAP;

    col = term_printf(row, col, T_DEFAULT, T_DIM, "Elapsed ");
    col = term_printf(row, col, T_DEFAULT, T_BOLD, "%-8s", elapsed);
    col += STAT_GAP;

    // an estimate needs at least one iteration's worth of evidence
    if (snap->iter > 0 && snap->total_iters > snap->iter && snap->state != M_DONE) {
        double remaining = snap->elapsed_s
                           * (double)(snap->total_iters - snap->iter) / (double)snap->iter;
        tui_format_duration(eta, sizeof(eta), remaining);
    } else {
        snprintf(eta, sizeof(eta), "--");
    }

    col = term_printf(row, col, T_DEFAULT, T_DIM, "Remaining ");
    term_printf(row, col, T_DEFAULT, T_BOLD, "%-8s", eta);
}

/**
 * @brief Draws the two accuracies, coloured to match their curves
 *
 * - This doubles as the graph's legend, which is why it carries the same swatch
 * glyph the curves are drawn in; keeping the legend out of the plot region means
 * it can never cover the curves it describes
 */
static void draw_accuracy_stats(int row, const TrainSnapshot *snap) {
    uint32_t swatch = plot_swatch();

    int col = BODY_COL;

    term_put(row, col, swatch, PLOT_TRAIN_COLOR, T_NORMAL);
    col = term_printf(row, col + 2, PLOT_TRAIN_COLOR, T_DIM, "Train accuracy ");
    col = term_printf(row, col, PLOT_TRAIN_COLOR, T_BOLD, "%.4f", snap->train_acc);
    col += STAT_GAP;

    term_put(row, col, swatch, PLOT_TEST_COLOR, T_NORMAL);
    col = term_printf(row, col + 2, PLOT_TEST_COLOR, T_DIM, "Test accuracy ");
    term_printf(row, col, PLOT_TEST_COLOR, T_BOLD, "%.4f", snap->test_acc);
}

/**
 * @brief Draws a bracketed progress bar with a percentage
 *
 * @param[in] row   Row to draw on
 * @param[in] width Total width available, including the brackets and percentage
 * @param[in] frac  Fraction complete, in [0, 1]
 */
static void draw_progress_bar(int row, int width, float frac) {
    // "[" + bar + "] " + "100%"
    int inner = width - 7;
    if (inner < 4) return;

    if (frac < 0.f) frac = 0.f;
    if (frac > 1.f) frac = 1.f;

    int filled = (int)(frac * (float)inner + 0.5f);

    int col = BODY_COL;
    term_put(row, col++, '[', T_DEFAULT, T_DIM);

    term_fill(row, col, filled, '#', T_GREEN, T_NORMAL);
    term_fill(row, col + filled, inner - filled, '-', T_DEFAULT, T_DIM);
    col += inner;

    term_put(row, col++, ']', T_DEFAULT, T_DIM);
    term_printf(row, col + 1, T_DEFAULT, T_BOLD, "%3.0f%%", frac * 100.f);
}

/**
 * @brief Picks the status message for the run's current state
 *
 * @param[out] out   Buffer for the message
 * @param[in]  cap   Size of the buffer
 * @param[in]  snap  Current snapshot
 * @param[out] color Colour to draw the message in
 */
static void status_message(char *out, size_t cap, const TrainSnapshot *snap, uint8_t *color) {
    char elapsed[32];

    switch (snap->state) {
        case M_PAUSED:
            *color = T_YELLOW;
            snprintf(out, cap, "Paused - ^P resumes");
            break;

        case M_STOPPING:
            *color = T_YELLOW;
            snprintf(out, cap, "Stopping at the end of this iteration");
            break;

        case M_DONE:
            tui_format_duration(elapsed, sizeof(elapsed), snap->elapsed_s);

            if (snap->stopped_early) {
                *color = T_YELLOW;
                snprintf(out, cap, "Stopped after %d iterations - test accuracy %.4f in %s",
                         snap->iter, snap->test_acc, elapsed);
            } else {
                *color = T_GREEN;
                snprintf(out, cap, "Done - final test accuracy %.4f in %s",
                         snap->test_acc, elapsed);
            }
            break;

        default:
            *color = T_DEFAULT;

            // totals are only known once the dataset has been read, and reading
            // it takes long enough to be worth saying so
            if (snap->total_iters == 0) {
                snprintf(out, cap, "Loading %s", tui_dataset_name());
            } else if (snap->iter >= snap->total_iters) {
                // the closing pass over the whole test set is a second or so on a
                // large model, and it would otherwise look like a stall at 100%
                snprintf(out, cap, "Evaluating final test accuracy");
            } else {
                snprintf(out, cap, "Training run in progress");
            }
            break;
    }
}

static void draw_shortcut_bar(const TrainSnapshot *snap) {
    static const Tui_shortcut running[] = {
        { "^X", "Stop" },
        { "^P", "Pause" },
        { "^R", "Restart" },
        { "^C", "Quit" },
        { "^L", "Redraw" }
    };
    static const Tui_shortcut paused[] = {
        { "^X", "Stop" },
        { "^P", "Resume" },
        { "^R", "Restart" },
        { "^C", "Quit" },
        { "^L", "Redraw" }
    };
    static const Tui_shortcut finished[] = {
        { "^X", "Exit" },
        { "Enter", "New run" },
        { "^L", "Redraw" }
    };

    if (snap->state == M_DONE) {
        tui_draw_shortcuts(finished, (int)(sizeof(finished) / sizeof(finished[0])));
    } else if (snap->state == M_PAUSED) {
        tui_draw_shortcuts(paused, (int)(sizeof(paused) / sizeof(paused[0])));
    } else {
        tui_draw_shortcuts(running, (int)(sizeof(running) / sizeof(running[0])));
    }
}

static void draw_dash(const TrainConfig *cfg, const TrainSnapshot *snap,
                      const MetricsHistory *hist) {
    char context[128];
    char status[256];
    uint8_t status_color;

    // title bar: dataset and architecture, the way the old header printed it
    int n = snprintf(context, sizeof(context), "%s  784", tui_dataset_name());
    for (int i = 0; i < cfg->num_hidden_layers && n > 0 && n < (int)sizeof(context); i++) {
        n += snprintf(context + n, sizeof(context) - (size_t)n, " -> %d",
                      cfg->hidden_layer_dims[i]);
    }
    if (n > 0 && n < (int)sizeof(context)) {
        snprintf(context + n, sizeof(context) - (size_t)n, " -> 10   batch %d   lr %.3f",
                 cfg->batch_size, cfg->learning_rate);
    }

    term_clear();
    tui_draw_title(context);

    if (tui_too_small()) {
        tui_draw_too_small();
        term_flush();
        return;
    }

    Tui_layout layout = tui_layout();
    int row = layout.body_row;

    draw_progress_stats(row, snap, cfg);
    draw_accuracy_stats(row + 1, snap);

    float frac = (snap->total_iters > 0)
                     ? (float)snap->iter / (float)snap->total_iters
                     : 0.f;
    if (snap->state == M_DONE && !snap->stopped_early) frac = 1.f;

    draw_progress_bar(row + 2, layout.cols - 2 * BODY_COL, frac);

    // the graph takes whatever the stats leave behind
    Plot_rect rect;
    rect.row = row + DASH_HEADER_ROWS;
    rect.col = BODY_COL;
    rect.width = layout.cols - 2 * BODY_COL;
    rect.height = layout.body_rows - DASH_HEADER_ROWS;

    plot_accuracy(rect, hist, (float)cfg->num_epochs);

    status_message(status, sizeof(status), snap, &status_color);
    tui_draw_status(status, status_color);

    draw_shortcut_bar(snap);

    term_flush();
}

////////////////////////////////////////////////////////////////////////////////
// SCREEN LOOP
////////////////////////////////////////////////////////////////////////////////

Tui_action tui_run_training(const TrainConfig *cfg) {
    // history is 16KB, so keep it out of the stack and re-copy it only when the
    // worker has actually added a point
    static MetricsHistory hist;

    worker_cfg = *cfg;

    // the UI owns the run's lifecycle: clearing before the worker starts means a
    // stop requested while the dataset is still loading cannot be lost
    metrics_reset();
    memset(&hist, 0, sizeof(hist));

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0) {
        metrics_finish(true, 0.f);

        term_clear();
        tui_draw_title(NULL);
        tui_draw_status("could not start the training thread", T_RED);
        term_flush();
        term_read_key(-1);

        return TUI_QUIT;
    }

    unsigned drawn_generation = ~0u;

    bool leave_when_done = false;
    Tui_action pending = TUI_QUIT;
    Tui_action action = TUI_QUIT;

    double next_frame = term_now_ms() + FRAME_MS;

    for (;;) {
        term_check_resize();

        TrainSnapshot snap;
        metrics_snapshot(&snap);

        if (snap.generation != drawn_generation) {
            metrics_copy_history(&hist);
            drawn_generation = snap.generation;
        }

        draw_dash(cfg, &snap, &hist);

        if (snap.state == M_DONE) {
            if (leave_when_done) {
                action = pending;
                break;
            }

            // the screen is static now, so wait indefinitely instead of spinning
            // note: a resize interrupts the wait and returns K_NONE, which redraws
            int key = term_read_key(-1);

            if (key == K_CTRL('L')) {
                term_force_repaint();
            } else if (key == K_CTRL('R') || key == K_ENTER) {
                action = TUI_AGAIN;
                break;
            } else if (key == K_CTRL('X') || key == K_CTRL('C') || key == K_ESC) {
                action = TUI_QUIT;
                break;
            }

            continue;
        }

        // spend the rest of the frame waiting for input; a keypress redraws
        // immediately rather than at the next tick
        for (;;) {
            int wait_ms = (int)(next_frame - term_now_ms());
            if (wait_ms <= 0) break;

            int key = term_read_key(wait_ms);
            if (key == K_NONE) continue;

            switch (key) {
                case K_CTRL('X'):
                    // stop, but stay on the dashboard: the run still computes a
                    // final test accuracy and losing it to a keystroke would be
                    // a poor trade
                    metrics_request_stop();
                    break;

                case K_CTRL('C'):
                    metrics_request_stop();
                    leave_when_done = true;
                    pending = TUI_QUIT;
                    break;

                case K_CTRL('R'):
                    metrics_request_stop();
                    leave_when_done = true;
                    pending = TUI_AGAIN;
                    break;

                case K_CTRL('P'):
                    metrics_toggle_pause();
                    break;

                case K_CTRL('L'):
                    term_force_repaint();
                    break;

                default:
                    break;
            }

            break;  // redraw now that something changed
        }

        next_frame = term_now_ms() + FRAME_MS;
    }

    pthread_join(worker, NULL);

    return action;
}
