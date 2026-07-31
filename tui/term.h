/**
 * term.h - Low-level terminal control for the TUI
 *
 * Owns every byte written to or read from the terminal:
 *
 * - raw mode (termios), alternate screen, cursor hiding, autowrap suppression
 * - a double-buffered cell grid that is flushed as a minimal damage diff, so a
 *   frame that changes nothing writes nothing and there is never a full-screen
 *   clear (which is what makes the old `\e[2J`-per-frame display flicker)
 * - terminal size tracking via TIOCGWINSZ + SIGWINCH
 * - `poll`-based key reading, which doubles as the frame clock
 *
 * No other module may write to stdout or read from stdin.
 */

#ifndef _TUI_TERM_H_
#define _TUI_TERM_H_

#include <stdbool.h>
#include <stdint.h>

// cell foreground colors (basic ANSI palette)
typedef enum {
    T_DEFAULT = 0,
    T_RED,
    T_GREEN,
    T_YELLOW,
    T_BLUE,
    T_MAGENTA,
    T_CYAN,
    T_WHITE
} Term_color;

// cell attribute bitmask
#define T_NORMAL  0x00u
#define T_BOLD    0x01u
#define T_INVERSE 0x02u
#define T_DIM     0x04u

// special keys returned by `term_read_key`
// note: plain characters are returned as their byte value, so control keys are
//       simply K_CTRL('x') == 0x18
typedef enum {
    K_NONE = 0,      // no key available before the timeout expired
    K_ENTER = '\r',
    K_ESC = 0x1b,
    K_BACKSPACE = 0x7f,
    K_UP = 0x100,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_HOME,
    K_END,
    K_PGUP,
    K_PGDN,
    K_DELETE
} Term_key;

#define K_CTRL(c) ((c) & 0x1f)

/**
 * @brief Puts the terminal into raw full-screen mode
 *
 * - Installs signal handlers and an atexit hook that restore the terminal, so
 * neither a signal nor an abnormal exit can leave the user's shell wrecked
 *
 * @return Whether initialization succeeded (fails if stdin/stdout is not a tty)
 */
bool term_init(void);

/**
 * @brief Restores the terminal to the state it had before `term_init`
 *
 * - Idempotent, so calling it after the atexit hook has already run is safe
 */
void term_shutdown(void);

/**
 * @brief Number of columns in the terminal
 */
int term_cols(void);

/**
 * @brief Number of rows in the terminal
 */
int term_rows(void);

/**
 * @brief Reports whether the terminal was resized since the last call
 *
 * - Consumes the pending SIGWINCH, re-queries the size, reallocates the cell
 * grids and forces a full repaint of the next frame
 *
 * @return Whether a resize was handled
 */
bool term_check_resize(void);

/**
 * @brief Clears the back buffer to blank cells
 *
 * - Call at the top of every frame; this does not write to the terminal
 */
void term_clear(void);

/**
 * @brief Writes one Unicode codepoint into the back buffer
 *
 * - Out-of-bounds positions are silently clipped
 *
 * @param[in] row  Row to write to (0-indexed)
 * @param[in] col  Column to write to (0-indexed)
 * @param[in] ch   Unicode codepoint to write
 * @param[in] fg   Foreground color
 * @param[in] attr Attribute bitmask
 */
void term_put(int row, int col, uint32_t ch, uint8_t fg, uint8_t attr);

/**
 * @brief Writes a UTF-8 string into the back buffer
 *
 * @param[in] row  Row to write to (0-indexed)
 * @param[in] col  Column at which the string starts (0-indexed)
 * @param[in] s    UTF-8 string to write
 * @param[in] fg   Foreground color
 * @param[in] attr Attribute bitmask
 *
 * @return Column just past the last one written
 */
int term_puts(int row, int col, const char *s, uint8_t fg, uint8_t attr);

/**
 * @brief Writes a printf-formatted string into the back buffer
 *
 * @param[in] row  Row to write to (0-indexed)
 * @param[in] col  Column at which the string starts (0-indexed)
 * @param[in] fg   Foreground color
 * @param[in] attr Attribute bitmask
 * @param[in] fmt  Format string
 *
 * @return Column just past the last one written
 */
int term_printf(int row, int col, uint8_t fg, uint8_t attr, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

/**
 * @brief Fills a span of a row with a single codepoint
 *
 * - Used for the inverse-video title and shortcut bars
 *
 * @param[in] row   Row to fill (0-indexed)
 * @param[in] col   First column to fill (0-indexed)
 * @param[in] width Number of columns to fill
 * @param[in] ch    Unicode codepoint to fill with
 * @param[in] fg    Foreground color
 * @param[in] attr  Attribute bitmask
 */
void term_fill(int row, int col, int width, uint32_t ch, uint8_t fg, uint8_t attr);

/**
 * @brief Flushes the back buffer to the terminal as a minimal damage diff
 *
 * - Emits only the cells that differ from what is on screen, as one `write`
 */
void term_flush(void);

/**
 * @brief Forces the next flush to redraw every cell
 *
 * - For `^L` and for recovering from output written by something else
 */
void term_force_repaint(void);

/**
 * @brief Waits up to `timeout_ms` for a keypress
 *
 * - This is the frame clock: pass the time remaining in the current frame and
 * the UI stays responsive without busy-waiting or sleep jitter
 *
 * @param[in] timeout_ms Milliseconds to wait (0 polls, negative blocks)
 *
 * @return The key pressed, or K_NONE if the timeout expired first
 */
int term_read_key(int timeout_ms);

/**
 * @brief Monotonic millisecond clock for frame pacing
 */
double term_now_ms(void);

#endif
