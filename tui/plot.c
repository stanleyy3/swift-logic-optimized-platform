/**
 * plot.c - Braille curve plotting for the training dashboard
 */

#include "plot.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../config.h"
#include "term.h"

// width of the y-axis label gutter, including the axis column itself
// note: "1.00 |" is six cells
#define Y_GUTTER 6

// rows below the canvas: the x axis line and its labels
#define X_AXIS_ROWS 2

// base of the braille block; the low 8 bits of a glyph are its dot bitmask
#define BRAILLE_BASE 0x2800u
#define BRAILLE_FULL 0x28ffu

// series colours
#define TRAIN_COLOR PLOT_TRAIN_COLOR
#define TEST_COLOR PLOT_TEST_COLOR

// braille dot bits, indexed by [x within cell][y within cell]
//     1 4
//     2 5
//     3 6
//     7 8
static const uint8_t DOT[2][4] = { { 0x01, 0x02, 0x04, 0x40 },
                                   { 0x08, 0x10, 0x20, 0x80 } };

static bool ascii_mode = false;

// dot canvas, kept across frames so that plotting does not malloc at frame rate
static uint8_t *dots = NULL;   // one dot bitmask per cell
static uint8_t *colors = NULL; // one colour per cell
static int canvas_w = 0;       // canvas width in cells
static int canvas_h = 0;       // canvas height in cells
static size_t canvas_cap = 0;  // cells currently allocated

// scratch for resampling a series down to one value per dot column
static float *col_sum = NULL;
static int *col_n = NULL;
static size_t col_cap = 0;

void plot_init(void) {
#if TUI_ASCII_FALLBACK
    ascii_mode = true;
#else
    // braille needs a UTF-8 locale and a font with the glyphs; if the locale
    // does not claim UTF-8, assume the glyphs will not render
    const char *vars[3];
    vars[0] = getenv("LC_ALL");
    vars[1] = getenv("LC_CTYPE");
    vars[2] = getenv("LANG");

    ascii_mode = true;
    for (int i = 0; i < 3; i++) {
        if (!vars[i] || !vars[i][0]) continue;

        if (strstr(vars[i], "UTF-8") || strstr(vars[i], "utf8")
            || strstr(vars[i], "UTF8") || strstr(vars[i], "utf-8")) {
            ascii_mode = false;
        }
        break;  // the first variable that is set is the one that decides
    }
#endif
}

bool plot_is_ascii(void) {
    return ascii_mode;
}

uint32_t plot_swatch(void) {
    return ascii_mode ? (uint32_t)'#' : BRAILLE_FULL;
}

////////////////////////////////////////////////////////////////////////////////
// DOT CANVAS
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Sizes and clears the dot canvas
 *
 * @param[in] w Canvas width in cells
 * @param[in] h Canvas height in cells
 *
 * @return Whether the canvas is usable
 */
static bool canvas_reset(int w, int h) {
    if (w <= 0 || h <= 0) return false;

    size_t need = (size_t)w * (size_t)h;

    if (need > canvas_cap) {
        uint8_t *new_dots = realloc(dots, need);
        if (!new_dots) return false;
        dots = new_dots;

        uint8_t *new_colors = realloc(colors, need);
        if (!new_colors) return false;
        colors = new_colors;

        canvas_cap = need;
    }

    size_t num_cols = (size_t)w * 2;
    if (num_cols > col_cap) {
        float *new_sum = realloc(col_sum, num_cols * sizeof(float));
        if (!new_sum) return false;
        col_sum = new_sum;

        int *new_n = realloc(col_n, num_cols * sizeof(int));
        if (!new_n) return false;
        col_n = new_n;

        col_cap = num_cols;
    }

    canvas_w = w;
    canvas_h = h;

    memset(dots, 0, need);
    memset(colors, T_DEFAULT, need);

    return true;
}

/**
 * @brief Sets one dot on the canvas
 *
 * - Out-of-range dots are dropped, which is what clips a curve to the panel
 *
 * @param[in] x     Dot column, 0 at the left edge
 * @param[in] y     Dot row, 0 at the top edge
 * @param[in] color Colour to attribute the containing cell to
 */
static void canvas_dot(int x, int y, uint8_t color) {
    if (x < 0 || y < 0) return;
    if (x >= canvas_w * 2 || y >= canvas_h * 4) return;

    size_t idx = (size_t)(y / 4) * (size_t)canvas_w + (size_t)(x / 2);

    dots[idx] |= DOT[x & 1][y & 3];
    colors[idx] = color;
}

/**
 * @brief Draws a straight line of dots (Bresenham)
 */
static void canvas_line(int x0, int y0, int x1, int y1, uint8_t color) {
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        canvas_dot(x0, y0, color);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

/**
 * @brief Draws a small cross of dots, marking a real measurement
 */
static void canvas_marker(int x, int y, uint8_t color) {
    canvas_dot(x, y, color);
    canvas_dot(x - 1, y, color);
    canvas_dot(x + 1, y, color);
    canvas_dot(x, y - 1, color);
    canvas_dot(x, y + 1, color);
}

/**
 * @brief Picks an ASCII stand-in for a cell's dot pattern
 *
 * - Collapses the cell's two dot columns onto its four dot rows and picks a
 * character that sits at roughly the right height
 */
static char ascii_glyph(uint8_t mask) {
    int top = -1;
    int bottom = -1;

    for (int y = 0; y < 4; y++) {
        if (mask & (DOT[0][y] | DOT[1][y])) {
            if (top < 0) top = y;
            bottom = y;
        }
    }

    if (top < 0) return ' ';
    if (bottom - top >= 2) return '|';
    if (bottom - top == 1) return ':';

    static const char levels[4] = { '\'', '-', '.', '_' };
    return levels[top];
}

/**
 * @brief Writes the canvas into the terminal back buffer
 *
 * @param[in] row Row of the canvas' top-left cell
 * @param[in] col Column of the canvas' top-left cell
 */
static void canvas_blit(int row, int col) {
    for (int r = 0; r < canvas_h; r++) {
        for (int c = 0; c < canvas_w; c++) {
            size_t idx = (size_t)r * (size_t)canvas_w + (size_t)c;

            uint8_t mask = dots[idx];
            if (mask == 0) continue;

            uint32_t ch = ascii_mode ? (uint32_t)ascii_glyph(mask)
                                     : (BRAILLE_BASE | mask);

            term_put(row + r, col + c, ch, colors[idx], T_NORMAL);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// ACCURACY GRAPH
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Maps an accuracy in [0, 1] to a dot row
 */
static int acc_to_dot_y(float acc) {
    int max_y = canvas_h * 4 - 1;

    if (acc < 0.f) acc = 0.f;
    if (acc > 1.f) acc = 1.f;

    return (int)((1.f - acc) * (float)max_y + 0.5f);
}

/**
 * @brief Maps an x in [0, x_max] to a dot column
 */
static int x_to_dot_x(float x, float x_max) {
    int max_x = canvas_w * 2 - 1;

    if (x_max <= 0.f) return 0;
    if (x < 0.f) x = 0.f;
    if (x > x_max) x = x_max;

    return (int)(x / x_max * (float)max_x + 0.5f);
}

/**
 * @brief Draws the y-axis labels and the axis column
 */
static void draw_y_axis(Plot_rect rect, int canvas_rows) {
    static const float labels[] = { 1.f, 0.75f, 0.5f, 0.25f };

    // the axis column runs the height of the canvas
    for (int r = 0; r < canvas_rows; r++) {
        term_put(rect.row + r, rect.col + Y_GUTTER - 1, '|', T_DEFAULT, T_DIM);
    }

    int num_labels = (canvas_rows >= 8) ? 4 : 2;  // drop the quarter marks when short

    for (int i = 0; i < num_labels; i++) {
        float v = labels[(num_labels == 4) ? i : i * 2];

        int r = (int)((1.f - v) * (float)canvas_rows);
        if (r >= canvas_rows) r = canvas_rows - 1;

        term_printf(rect.row + r, rect.col, T_DEFAULT, T_DIM, "%4.2f", v);
    }
}

/**
 * @brief Draws the x-axis line, its ticks and its labels
 */
static void draw_x_axis(Plot_rect rect, int canvas_rows, float x_max) {
    int axis_row = rect.row + canvas_rows;
    int label_row = axis_row + 1;
    int axis_col = rect.col + Y_GUTTER - 1;

    // 0.00 sits on the axis line rather than inside the canvas
    term_printf(axis_row, rect.col, T_DEFAULT, T_DIM, "%4.2f", 0.f);

    // the corner doubles as the tick for x = 0
    term_put(axis_row, axis_col, '+', T_DEFAULT, T_DIM);
    term_fill(axis_row, axis_col + 1, canvas_w, '-', T_DEFAULT, T_DIM);

    // name the axis in the gutter under the y labels, where nothing can collide
    const char *unit = "epoch";
    term_puts(label_row, axis_col - (int)strlen(unit), unit, T_DEFAULT, T_DIM);

    if (x_max <= 0.f) return;

    // the corner is the tick for x = 0, so it only needs its label
    term_puts(label_row, axis_col + 1, "0", T_DEFAULT, T_DIM);

    // the right edge says how long the whole run is, so it is labelled first and
    // never dropped; intermediate labels give way to it
    char last[16];
    int last_len = snprintf(last, sizeof(last), "%d", (int)x_max);
    int last_col = rect.col + rect.width - last_len;

    term_put(axis_row, axis_col + 1 + x_to_dot_x(x_max, x_max) / 2, '+', T_DEFAULT, T_DIM);
    term_puts(label_row, last_col, last, T_DEFAULT, T_DIM);

    // widen the tick spacing until the labels cannot collide
    int stride = 1;
    while (x_max / (float)stride > (float)canvas_w / 8.f) stride *= 2;

    for (int v = stride; (float)v < x_max; v += stride) {
        int col = axis_col + 1 + x_to_dot_x((float)v, x_max) / 2;

        term_put(axis_row, col, '+', T_DEFAULT, T_DIM);

        // centre the label under its tick
        char buf[16];
        int n = snprintf(buf, sizeof(buf), "%d", v);
        int label_col = col - n / 2;

        // leave the tick mark but drop the label if it would run into either of
        // the two labels that always get drawn
        if (label_col <= axis_col + 2) continue;
        if (label_col + n >= last_col - 1) continue;

        term_puts(label_row, label_col, buf, T_DEFAULT, T_DIM);
    }
}

/**
 * @brief Draws the dense train-accuracy series as a polyline
 *
 * - A run publishes many more points than the canvas has dot columns, so the
 * points are first resampled to one mean per column. Drawing them all instead
 * would paint each column's entire spread and the trend would disappear into a
 * solid band.
 */
static void draw_train_series(const MetricsHistory *hist, float x_max) {
    int num_cols = canvas_w * 2;

    for (int c = 0; c < num_cols; c++) {
        col_sum[c] = 0.f;
        col_n[c] = 0;
    }

    for (int i = 0; i < hist->train_len; i++) {
        int c = x_to_dot_x(hist->train[i].x, x_max);
        if (c < 0 || c >= num_cols) continue;

        col_sum[c] += hist->train[i].mean;
        col_n[c]++;
    }

    int prev_x = -1;
    int prev_y = 0;

    for (int c = 0; c < num_cols; c++) {
        if (col_n[c] == 0) continue;

        int y = acc_to_dot_y(col_sum[c] / (float)col_n[c]);

        if (prev_x < 0) {
            canvas_dot(c, y, TRAIN_COLOR);
        } else {
            canvas_line(prev_x, prev_y, c, y, TRAIN_COLOR);
        }

        prev_x = c;
        prev_y = y;
    }
}

/**
 * @brief Draws the sparse test-accuracy series
 *
 * - Segments connect the measurements and a marker sits on each one, so the
 * lower sample rate reads as deliberate rather than as a coarse approximation
 */
static void draw_test_series(const MetricsHistory *hist, float x_max) {
    int prev_x = 0;
    int prev_y = 0;

    for (int i = 0; i < hist->test_len; i++) {
        int x = x_to_dot_x(hist->test[i].x, x_max);
        int y = acc_to_dot_y(hist->test[i].acc);

        if (i > 0) canvas_line(prev_x, prev_y, x, y, TEST_COLOR);

        prev_x = x;
        prev_y = y;
    }

    // markers go on last so that a segment never paints over one
    for (int i = 0; i < hist->test_len; i++) {
        canvas_marker(x_to_dot_x(hist->test[i].x, x_max),
                      acc_to_dot_y(hist->test[i].acc),
                      TEST_COLOR);
    }
}

void plot_accuracy(Plot_rect rect, const MetricsHistory *hist, float x_max) {
    if (rect.width < PLOT_MIN_WIDTH || rect.height < PLOT_MIN_HEIGHT) return;

    int canvas_cols = rect.width - Y_GUTTER;
    int canvas_rows = rect.height - X_AXIS_ROWS;

    if (!canvas_reset(canvas_cols, canvas_rows)) return;

    // train first, so the sparser and more meaningful test series wins any cell
    // the two of them contend for
    draw_train_series(hist, x_max);
    draw_test_series(hist, x_max);

    canvas_blit(rect.row, rect.col + Y_GUTTER);

    draw_y_axis(rect, canvas_rows);
    draw_x_axis(rect, canvas_rows, x_max);
}
