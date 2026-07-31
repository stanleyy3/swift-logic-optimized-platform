/**
 * plot.h - Braille curve plotting for the training dashboard
 *
 * Each terminal cell holds a 2x4 grid of braille dots (U+2800 - U+28FF), so a
 * plot region of W x H cells is a 2W x 4H dot canvas - roughly eight times the
 * resolution of plotting with ASCII characters.
 *
 * Two series share one panel. A braille cell cannot show which series a dot
 * belongs to, so colour carries series identity and the legend names both.
 *
 * Plot cost is O(canvas cells + history points), and the history the caller
 * hands over is bounded by `metrics.h`, so plotting a 3,000,000-iteration run
 * costs the same as plotting a short one.
 */

#ifndef _TUI_PLOT_H_
#define _TUI_PLOT_H_

#include <stdbool.h>
#include <stdint.h>

#include "../metrics.h"
#include "term.h"

// smallest plot region that still has room for a canvas, an axis and labels
#define PLOT_MIN_WIDTH 24
#define PLOT_MIN_HEIGHT 5

// series colours, shared with whoever draws the legend
#define PLOT_TRAIN_COLOR T_CYAN
#define PLOT_TEST_COLOR T_YELLOW

// a rectangle of terminal cells, including the axes and their labels
typedef struct {
    int row;     // topmost row (0-indexed)
    int col;     // leftmost column (0-indexed)
    int width;   // width in cells
    int height;  // height in cells
} Plot_rect;

/**
 * @brief Decides whether braille glyphs are usable, once per program run
 *
 * - Falls back to ASCII when the locale does not look like UTF-8, or when
 * TUI_ASCII_FALLBACK is set in config.h
 */
void plot_init(void);

/**
 * @brief Whether plotting is running in ASCII fallback mode
 */
bool plot_is_ascii(void);

/**
 * @brief Codepoint of a filled glyph, for use as a legend swatch
 *
 * - The legend lives outside the plot region so that it cannot cover the curves,
 * but it has to be drawn in the same glyph the curves are drawn in
 */
uint32_t plot_swatch(void);

/**
 * @brief Draws the accuracy graph into the back buffer
 *
 * - The y axis is fixed to [0, 1] and the x axis to [0, x_max] for the whole
 * run, so the curve grows into a stable frame instead of rescaling every frame
 *
 * @param[in] rect  Region of the screen to draw into
 * @param[in] hist  History to plot
 * @param[in] x_max Right edge of the x axis, in epochs
 */
void plot_accuracy(Plot_rect rect, const MetricsHistory *hist, float x_max);

#endif
