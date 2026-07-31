/**
 * term.c - Low-level terminal control for the TUI
 */

#include "term.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

// milliseconds to wait for the remaining bytes of an escape sequence before
// concluding that the user pressed a bare ESC
#define ESC_SEQ_WAIT_MS 25

// identical cells tolerated inside a damage run; writing a couple of unchanged
// cells is cheaper than emitting another cursor-position escape sequence
#define GAP_TOL 4

// fallback size if TIOCGWINSZ fails (e.g. under a pty with no size set)
#define FALLBACK_COLS 80
#define FALLBACK_ROWS 24

// escape sequences for entering and leaving full-screen mode
// \e[?1049h  switch to the alternate screen buffer
// \e[?25l    hide the cursor
// \e[?7l     disable autowrap, so writing the bottom-right cell cannot scroll
static const char ENTER_SEQ[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J";
static const char LEAVE_SEQ[] = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l";

typedef struct {
    uint32_t ch;    // Unicode codepoint (0 is treated as a blank)
    uint8_t fg;     // Term_color
    uint8_t attr;   // attribute bitmask
} Cell;

static struct termios orig_termios;
static bool termios_saved = false;
static bool active = false;

static int rows = 0;
static int cols = 0;

static Cell *front = NULL;  // what the terminal is currently showing
static Cell *back = NULL;   // what the next frame should show

// set from the SIGWINCH handler, consumed by `term_check_resize`
static volatile sig_atomic_t winch_pending = 0;

// output staging buffer, so a whole frame leaves as one write()
static char *out_buf = NULL;
static size_t out_len = 0;
static size_t out_cap = 0;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT BUFFER
////////////////////////////////////////////////////////////////////////////////

static void ob_reserve(size_t extra) {
    if (out_len + extra <= out_cap) return;

    size_t new_cap = (out_cap == 0) ? 8192 : out_cap;
    while (new_cap < out_len + extra) new_cap *= 2;

    char *grown = realloc(out_buf, new_cap);
    if (!grown) return;  // drop output rather than die mid-frame

    out_buf = grown;
    out_cap = new_cap;
}

static void ob_append(const char *s, size_t n) {
    ob_reserve(n);
    if (out_len + n > out_cap) return;

    memcpy(out_buf + out_len, s, n);
    out_len += n;
}

static void ob_str(const char *s) {
    ob_append(s, strlen(s));
}

static void ob_printf(const char *fmt, ...) {
    char tmp[64];

    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if (n > 0) ob_append(tmp, (size_t)n < sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

/**
 * @brief Appends a codepoint to the output buffer as UTF-8
 */
static void ob_utf8(uint32_t cp) {
    if (cp == 0) cp = ' ';

    if (cp < 0x80) {
        char c = (char)cp;
        ob_append(&c, 1);
    } else if (cp < 0x800) {
        char s[2] = { (char)(0xc0 | (cp >> 6)), (char)(0x80 | (cp & 0x3f)) };
        ob_append(s, 2);
    } else if (cp < 0x10000) {
        char s[3] = { (char)(0xe0 | (cp >> 12)),
                      (char)(0x80 | ((cp >> 6) & 0x3f)),
                      (char)(0x80 | (cp & 0x3f)) };
        ob_append(s, 3);
    } else {
        char s[4] = { (char)(0xf0 | (cp >> 18)),
                      (char)(0x80 | ((cp >> 12) & 0x3f)),
                      (char)(0x80 | ((cp >> 6) & 0x3f)),
                      (char)(0x80 | (cp & 0x3f)) };
        ob_append(s, 4);
    }
}

////////////////////////////////////////////////////////////////////////////////
// TERMINAL SETUP AND TEARDOWN
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Restores the terminal using only async-signal-safe calls
 */
static void restore_raw(void) {
    if (!active) return;
    active = false;

    ssize_t ignored = write(STDOUT_FILENO, LEAVE_SEQ, sizeof(LEAVE_SEQ) - 1);
    (void)ignored;

    if (termios_saved) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

static void on_winch(int sig) {
    (void)sig;
    winch_pending = 1;
}

/**
 * @brief Restores the terminal, then lets the signal kill us as it normally would
 */
static void on_fatal(int sig) {
    restore_raw();

    signal(sig, SIG_DFL);
    raise(sig);
}

static void query_size(void) {
    struct winsize ws;

    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        cols = ws.ws_col;
        rows = ws.ws_row;
    } else {
        cols = FALLBACK_COLS;
        rows = FALLBACK_ROWS;
    }
}

/**
 * @brief Marks every cell of the front buffer as unknown
 *
 * - Filling with 0xff gives each cell a codepoint and a style that no real cell
 * can hold, so the next flush repaints all of them
 *
 * - Note that zeroing would not do: `cell_eq` reads a zero codepoint as a space,
 * so a zeroed cell compares equal to a blank one and blank cells would silently
 * keep whatever the terminal happened to be showing
 */
static void invalidate_front(void) {
    if (!front) return;

    memset(front, 0xff, (size_t)rows * (size_t)cols * sizeof(Cell));
}

/**
 * @brief Allocates the cell grids for the current terminal size
 *
 * @return Whether allocation succeeded
 */
static bool alloc_grids(void) {
    size_t n = (size_t)rows * (size_t)cols;

    Cell *new_front = realloc(front, n * sizeof(Cell));
    if (!new_front) return false;
    front = new_front;

    Cell *new_back = realloc(back, n * sizeof(Cell));
    if (!new_back) return false;
    back = new_back;

    memset(back, 0, n * sizeof(Cell));
    invalidate_front();

    return true;
}

bool term_init(void) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    if (tcgetattr(STDIN_FILENO, &orig_termios) != 0) return false;
    termios_saved = true;

    struct termios raw = orig_termios;

    // input: no CR/NL translation, no flow control, no parity checks
    raw.c_iflag &= ~(unsigned)(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    // output: no post-processing, so `\n` is not turned into CRLF
    raw.c_oflag &= ~(unsigned)OPOST;
    // local: no echo, no line buffering, no signal generation (^C becomes a key)
    raw.c_lflag &= ~(unsigned)(ECHO | ICANON | ISIG | IEXTEN);
    raw.c_cflag |= CS8;
    // reads return immediately; `poll` does the waiting
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) {
        termios_saved = false;
        return false;
    }

    query_size();
    if (!alloc_grids()) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        return false;
    }

    struct sigaction sa;

    // note: no SA_RESTART, so a resize interrupts the poll in `term_read_key`
    //       and the UI reacts to it within the same frame
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_winch;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGWINCH, &sa, NULL);

    // restore the terminal on the signals that would otherwise leave the user's
    // shell without echo and without a cursor
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_fatal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

    // a write to a dead terminal should fail, not kill us mid-frame
    signal(SIGPIPE, SIG_IGN);

    active = true;
    atexit(term_shutdown);

    ssize_t ignored = write(STDOUT_FILENO, ENTER_SEQ, sizeof(ENTER_SEQ) - 1);
    (void)ignored;

    return true;
}

void term_shutdown(void) {
    restore_raw();

    free(front);
    free(back);
    free(out_buf);

    front = NULL;
    back = NULL;
    out_buf = NULL;
    out_cap = 0;
    out_len = 0;
}

int term_cols(void) {
    return cols;
}

int term_rows(void) {
    return rows;
}

bool term_check_resize(void) {
    if (!winch_pending) return false;
    winch_pending = 0;

    int old_rows = rows;
    int old_cols = cols;

    query_size();

    if (rows == old_rows && cols == old_cols) {
        // size did not actually change, but something resized us, so repaint
        term_force_repaint();
        return true;
    }

    if (!alloc_grids()) {
        // could not grow the grids, so stay at the old size rather than write
        // outside them
        rows = old_rows;
        cols = old_cols;
        return false;
    }

    // the terminal keeps showing whatever it reflowed the old screen into, and
    // anything the new layout leaves blank would otherwise survive the resize
    term_force_repaint();

    return true;
}

void term_force_repaint(void) {
    if (!front) return;

    invalidate_front();
    ob_str("\x1b[2J");
}

////////////////////////////////////////////////////////////////////////////////
// BACK BUFFER WRITES
////////////////////////////////////////////////////////////////////////////////

void term_clear(void) {
    if (!back) return;

    size_t n = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < n; i++) {
        back[i].ch = ' ';
        back[i].fg = T_DEFAULT;
        back[i].attr = T_NORMAL;
    }
}

void term_put(int row, int col, uint32_t ch, uint8_t fg, uint8_t attr) {
    if (!back) return;
    if (row < 0 || row >= rows || col < 0 || col >= cols) return;

    Cell *cell = &back[(size_t)row * (size_t)cols + (size_t)col];
    cell->ch = ch;
    cell->fg = fg;
    cell->attr = attr;
}

/**
 * @brief Decodes one UTF-8 sequence
 *
 * @param[in]  s        String to decode from
 * @param[out] consumed Number of bytes consumed
 *
 * @return The decoded codepoint (invalid bytes decode to '?')
 */
static uint32_t utf8_decode(const char *s, int *consumed) {
    unsigned char c = (unsigned char)s[0];

    int len;
    uint32_t cp;

    if (c < 0x80) {
        *consumed = 1;
        return c;
    } else if ((c & 0xe0) == 0xc0) {
        len = 2;
        cp = c & 0x1fu;
    } else if ((c & 0xf0) == 0xe0) {
        len = 3;
        cp = c & 0x0fu;
    } else if ((c & 0xf8) == 0xf0) {
        len = 4;
        cp = c & 0x07u;
    } else {
        *consumed = 1;
        return '?';
    }

    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xc0) != 0x80) {
            // truncated sequence
            *consumed = i;
            return '?';
        }
        cp = (cp << 6) | ((unsigned char)s[i] & 0x3fu);
    }

    *consumed = len;
    return cp;
}

int term_puts(int row, int col, const char *s, uint8_t fg, uint8_t attr) {
    while (*s && col < cols) {
        int consumed = 1;
        uint32_t cp = utf8_decode(s, &consumed);

        term_put(row, col, cp, fg, attr);

        s += consumed;
        col++;
    }

    return col;
}

int term_printf(int row, int col, uint8_t fg, uint8_t attr, const char *fmt, ...) {
    char tmp[512];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    return term_puts(row, col, tmp, fg, attr);
}

void term_fill(int row, int col, int width, uint32_t ch, uint8_t fg, uint8_t attr) {
    for (int i = 0; i < width; i++) {
        term_put(row, col + i, ch, fg, attr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// DAMAGE DIFF FLUSH
////////////////////////////////////////////////////////////////////////////////

static bool cell_eq(const Cell *a, const Cell *b) {
    uint32_t ach = (a->ch == 0) ? ' ' : a->ch;
    uint32_t bch = (b->ch == 0) ? ' ' : b->ch;

    return ach == bch && a->fg == b->fg && a->attr == b->attr;
}

/**
 * @brief Emits the SGR sequence for a cell's style if it differs from the current one
 *
 * @param[in, out] cur_fg   Foreground color currently set on the terminal
 * @param[in, out] cur_attr Attributes currently set on the terminal
 * @param[in]      fg       Foreground color wanted
 * @param[in]      attr     Attributes wanted
 */
static void emit_style(int *cur_fg, int *cur_attr, uint8_t fg, uint8_t attr) {
    if (*cur_fg == (int)fg && *cur_attr == (int)attr) return;

    // reset first, then set, so we never have to track which attributes need
    // individually turning off
    ob_str("\x1b[0");
    if (attr & T_BOLD) ob_str(";1");
    if (attr & T_DIM) ob_str(";2");
    if (attr & T_INVERSE) ob_str(";7");
    if (fg != T_DEFAULT) ob_printf(";%d", 30 + (fg - 1));
    ob_str("m");

    *cur_fg = fg;
    *cur_attr = attr;
}

void term_flush(void) {
    if (!active || !front || !back) return;

    int cur_fg = -1;
    int cur_attr = -1;

    for (int r = 0; r < rows; r++) {
        Cell *frow = &front[(size_t)r * (size_t)cols];
        Cell *brow = &back[(size_t)r * (size_t)cols];

        int c = 0;
        while (c < cols) {
            // skip to the next changed cell
            while (c < cols && cell_eq(&frow[c], &brow[c])) c++;
            if (c >= cols) break;

            // extend the run past short stretches of unchanged cells
            int start = c;
            int last_dirty = c;
            for (int i = c; i < cols; i++) {
                if (!cell_eq(&frow[i], &brow[i])) {
                    last_dirty = i;
                } else if (i - last_dirty > GAP_TOL) {
                    break;
                }
            }

            ob_printf("\x1b[%d;%dH", r + 1, start + 1);

            for (int i = start; i <= last_dirty; i++) {
                emit_style(&cur_fg, &cur_attr, brow[i].fg, brow[i].attr);
                ob_utf8(brow[i].ch);
                frow[i] = brow[i];
            }

            c = last_dirty + 1;
        }
    }

    if (out_len == 0) return;

    // leave the terminal in a neutral style between frames
    ob_str("\x1b[0m");

    size_t written = 0;
    while (written < out_len) {
        ssize_t n = write(STDOUT_FILENO, out_buf + written, out_len - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        written += (size_t)n;
    }

    out_len = 0;
}

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Reads a single byte, waiting at most `ms` milliseconds
 *
 * @param[in] ms Milliseconds to wait (0 polls, negative blocks)
 *
 * @return The byte read, or -1 on timeout, error or signal interruption
 */
static int read_byte(int ms) {
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    pfd.revents = 0;

    int n = poll(&pfd, 1, ms);
    if (n <= 0) return -1;  // includes EINTR, which lets a resize be handled

    unsigned char b;
    ssize_t got = read(STDIN_FILENO, &b, 1);
    if (got != 1) return -1;

    return b;
}

int term_read_key(int timeout_ms) {
    int b = read_byte(timeout_ms);
    if (b < 0) return K_NONE;

    if (b != 0x1b) {
        // accept LF as Enter; with OPOST off the terminal sends CR
        if (b == '\n') return K_ENTER;
        return b;
    }

    // escape sequence, or a bare ESC keypress if nothing follows
    int b1 = read_byte(ESC_SEQ_WAIT_MS);
    if (b1 < 0) return K_ESC;
    if (b1 != '[' && b1 != 'O') return K_ESC;

    int b2 = read_byte(ESC_SEQ_WAIT_MS);
    if (b2 < 0) return K_ESC;

    switch (b2) {
        case 'A': return K_UP;
        case 'B': return K_DOWN;
        case 'C': return K_RIGHT;
        case 'D': return K_LEFT;
        case 'H': return K_HOME;
        case 'F': return K_END;
        default: break;
    }

    if (b2 >= '0' && b2 <= '9') {
        // numeric form: \e[<n>~
        int num = b2 - '0';

        int b3;
        while ((b3 = read_byte(ESC_SEQ_WAIT_MS)) >= 0) {
            if (b3 >= '0' && b3 <= '9') {
                num = num * 10 + (b3 - '0');
            } else {
                break;  // '~' or a modifier we do not care about
            }
        }

        switch (num) {
            case 1: return K_HOME;
            case 3: return K_DELETE;
            case 4: return K_END;
            case 5: return K_PGUP;
            case 6: return K_PGDN;
            case 7: return K_HOME;
            case 8: return K_END;
            default: return K_ESC;
        }
    }

    return K_ESC;
}

double term_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
