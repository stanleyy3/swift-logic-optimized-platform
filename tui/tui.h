/**
 * tui.h - Terminal user interface for launching and watching training runs
 *
 * This is the only TUI header `main.c` needs. Everything that reads from or
 * writes to the terminal lives under `tui/`; the training code publishes
 * numbers through `metrics.h` and never learns that a terminal exists.
 *
 * The screen is laid out like nano:
 *
 *     row 0            inverse-video title bar
 *     rows 1..n        body (the hyperparameter form, or the dashboard)
 *     row n+1          status/message line
 *     rows n+2, n+3    two-row shortcut bar, inverse-video keys
 */

#ifndef _TUI_TUI_H_
#define _TUI_TUI_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../train.h"

// what the user asked to do after a training run finished
typedef enum {
    TUI_QUIT,  // leave the program
    TUI_AGAIN  // go back to the form for another run
} Tui_action;

// one entry in the shortcut bar
typedef struct {
    const char *key;    // e.g. "^X"
    const char *label;  // e.g. "Exit"
} Tui_shortcut;

// where the chrome leaves room for a screen's contents
typedef struct {
    int cols;      // terminal width
    int body_row;  // first row of the body
    int body_rows; // number of body rows
} Tui_layout;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM-LEVEL
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Takes over the terminal and prepares the UI
 *
 * @return Whether the UI came up (fails when stdin/stdout is not a terminal)
 */
bool tui_init(void);

/**
 * @brief Hands the terminal back
 */
void tui_shutdown(void);

/**
 * @brief Runs the hyperparameter form until the user starts a run or quits
 *
 * @param[in, out] cfg Configuration to edit; carried over between runs
 *
 * @return Whether a run should start (false means the user quit)
 */
bool tui_run_form(TrainConfig *cfg);

/**
 * @brief Runs a training run on a worker thread and draws it live
 *
 * - Returns once the run has finished and the user has dismissed the result
 *
 * @param[in] cfg Configuration for the run
 *
 * @return Whether to return to the form or leave the program
 */
Tui_action tui_run_training(const TrainConfig *cfg);

////////////////////////////////////////////////////////////////////////////////
// CHROME, SHARED BY THE SCREENS
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Computes the body region left over between the bars
 */
Tui_layout tui_layout(void);

/**
 * @brief Whether the terminal is too small to draw a screen in
 */
bool tui_too_small(void);

/**
 * @brief Draws a "terminal too small" notice in place of a screen's body
 */
void tui_draw_too_small(void);

/**
 * @brief Draws the inverse-video title bar
 *
 * @param[in] context Text shown next to the program name (may be NULL)
 */
void tui_draw_title(const char *context);

/**
 * @brief Draws the status/message line
 *
 * @param[in] msg  Message to show, bracketed and centred like nano's (may be NULL)
 * @param[in] fg   Colour for the message
 */
void tui_draw_status(const char *msg, uint8_t fg);

/**
 * @brief Draws the two-row shortcut bar
 *
 * @param[in] items Shortcuts to show
 * @param[in] count Number of shortcuts
 */
void tui_draw_shortcuts(const Tui_shortcut *items, int count);

/**
 * @brief Formats a duration the way the dashboard shows it
 *
 * @param[out] buf     Destination buffer
 * @param[in]  cap     Size of the destination buffer
 * @param[in]  seconds Duration to format
 */
void tui_format_duration(char *buf, size_t cap, double seconds);

/**
 * @brief Name of the dataset the program was built against
 *
 * - Keeps the DATASET conditional in one place instead of in every screen
 */
const char *tui_dataset_name(void);

#endif
