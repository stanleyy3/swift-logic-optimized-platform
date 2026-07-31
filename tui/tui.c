/**
 * tui.c - Shared chrome and lifecycle for the terminal user interface
 */

#include "tui.h"

#include <stdio.h>
#include <string.h>

#include "../config.h"
#include "plot.h"
#include "term.h"

// rows the chrome takes: title bar, status line and two shortcut rows
#define CHROME_ROWS 4

// column the title bar's context text starts no earlier than
#define TITLE_CONTEXT_COL 14

bool tui_init(void) {
    if (!term_init()) return false;

    plot_init();

    return true;
}

void tui_shutdown(void) {
    term_shutdown();
}

Tui_layout tui_layout(void) {
    Tui_layout layout;

    layout.cols = term_cols();
    layout.body_row = 1;
    layout.body_rows = term_rows() - CHROME_ROWS;

    if (layout.body_rows < 0) layout.body_rows = 0;

    return layout;
}

bool tui_too_small(void) {
    return term_cols() < TUI_MIN_COLS || term_rows() < TUI_MIN_ROWS;
}

void tui_draw_too_small(void) {
    char line[128];

    int rows = term_rows();
    int cols = term_cols();

    snprintf(line, sizeof(line), "terminal is %dx%d, need at least %dx%d",
             cols, rows, TUI_MIN_COLS, TUI_MIN_ROWS);

    int len = (int)strlen(line);
    int col = (cols - len) / 2;
    if (col < 0) col = 0;

    term_puts(rows / 2, col, line, T_YELLOW, T_NORMAL);
}

void tui_draw_title(const char *context) {
    int cols = term_cols();

    term_fill(0, 0, cols, ' ', T_DEFAULT, T_INVERSE);
    term_puts(0, 2, "SLOP 1.0", T_DEFAULT, T_INVERSE | T_BOLD);

    if (!context) return;

    int len = (int)strlen(context);
    int col = (cols - len) / 2;
    if (col < TITLE_CONTEXT_COL) col = TITLE_CONTEXT_COL;

    term_puts(0, col, context, T_DEFAULT, T_INVERSE);
}

void tui_draw_status(const char *msg, uint8_t fg) {
    if (!msg || !msg[0]) return;

    char framed[256];
    snprintf(framed, sizeof(framed), "[ %s ]", msg);

    int cols = term_cols();
    int len = (int)strlen(framed);
    int col = (cols - len) / 2;
    if (col < 0) col = 0;

    term_puts(term_rows() - 3, col, framed, fg, T_INVERSE);
}

void tui_draw_shortcuts(const Tui_shortcut *items, int count) {
    if (count <= 0) return;

    int cols = term_cols();
    int base_row = term_rows() - 2;

    // nano lays its shortcuts out column-major, two rows at a time
    int num_cols = (count + 1) / 2;
    int cell_w = cols / num_cols;

    for (int i = 0; i < count; i++) {
        int row = base_row + (i % 2);
        int col = (i / 2) * cell_w + 1;

        int c = term_puts(row, col, items[i].key, T_DEFAULT, T_INVERSE);
        term_puts(row, c + 1, items[i].label, T_DEFAULT, T_NORMAL);
    }
}

const char *tui_dataset_name(void) {
#if DATASET == 0
    return "MNIST";
#elif DATASET == 1
    return "Fashion MNIST";
#else
    return "unknown";
#endif
}

void tui_format_duration(char *buf, size_t cap, double seconds) {
    if (seconds < 0.0) seconds = 0.0;

    if (seconds < 60.0) {
        snprintf(buf, cap, "%.1fs", seconds);
    } else if (seconds < 3600.0) {
        int mins = (int)(seconds / 60.0);
        snprintf(buf, cap, "%dm %02ds", mins, (int)(seconds - mins * 60.0));
    } else {
        int hours = (int)(seconds / 3600.0);
        int mins = (int)((seconds - hours * 3600.0) / 60.0);
        snprintf(buf, cap, "%dh %02dm", hours, mins);
    }
}
