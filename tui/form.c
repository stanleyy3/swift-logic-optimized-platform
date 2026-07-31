/**
 * form.c - Hyperparameter collection screen
 *
 * A full-screen form rather than a sequence of prompts: every field is visible
 * at once, left/right cycles the enumerated ones, and the numeric ones clamp to
 * their bounds. Invalid input is therefore unrepresentable, which is what the
 * old `do/while (ret != 1 || ...)` re-prompt loops in main.c were for.
 *
 * The screen only redraws in response to a keypress or a resize, so it costs no
 * CPU while the user is deciding.
 */

#include "form.h"

#include <stdio.h>
#include <string.h>

#include "../config.h"
#include "term.h"
#include "tui.h"

// column layout within the body
#define COL_LABEL 4
#define COL_VALUE 26
#define COL_HINT 38
#define VALUE_WIDTH 7

// how many training samples a run will see per epoch, for the estimate line
// note: the datasets are not loaded until the run starts, so this is nominal
#define NOMINAL_TRAIN_SET_SIZE 60000

const int form_dim_choices[FORM_DIM_CHOICES] = { 16, 32, 64, 128, 256, 512 };
const float form_lr_choices[FORM_LR_CHOICES] = { 0.001f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f };

// presets that left/right steps through on the freely-typed fields
static const int epoch_presets[] = { 1, 2, 5, 10, 15, 20, 30, 50 };
static const int batch_presets[] = { 1, 8, 16, 32, 64, 100, 128, 256, 512, 1000 };

typedef enum {
    F_LAYERS,  // number of hidden layers
    F_DIM,     // one hidden layer's dimension
    F_EPOCHS,
    F_BATCH,
    F_LR
} Field_kind;

typedef struct {
    Field_kind kind;
    int layer;  // which hidden layer, for F_DIM
} Field;

// F_LAYERS + up to MAX_HIDDEN_LAYERS dimensions + epochs + batch + lr
#define MAX_FIELDS (MAX_HIDDEN_LAYERS + 4)

/**
 * @brief Builds the field list for a configuration
 *
 * - The layer dimension rows appear and disappear with the layer count, so the
 * list is rebuilt every frame
 *
 * @param[in]  cfg    Configuration being edited
 * @param[out] fields Field list to fill
 *
 * @return Number of fields
 */
static int build_fields(const TrainConfig *cfg, Field *fields) {
    int n = 0;

    fields[n].kind = F_LAYERS;
    fields[n].layer = 0;
    n++;

    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        fields[n].kind = F_DIM;
        fields[n].layer = i;
        n++;
    }

    fields[n].kind = F_EPOCHS;
    fields[n].layer = 0;
    n++;

    fields[n].kind = F_BATCH;
    fields[n].layer = 0;
    n++;

    fields[n].kind = F_LR;
    fields[n].layer = 0;
    n++;

    return n;
}

/**
 * @brief Steps to the neighbouring value in a sorted list
 *
 * - Works from any current value, so a typed number still steps sensibly
 *
 * @param[in] list Sorted list of values
 * @param[in] len  Length of the list
 * @param[in] cur  Current value
 * @param[in] dir  Direction to step (positive is up)
 *
 * @return The neighbouring value, or the nearest end of the list
 */
static int step_int_list(const int *list, int len, int cur, int dir) {
    if (dir > 0) {
        for (int i = 0; i < len; i++) {
            if (list[i] > cur) return list[i];
        }
        return list[len - 1];
    }

    for (int i = len - 1; i >= 0; i--) {
        if (list[i] < cur) return list[i];
    }
    return list[0];
}

static float step_float_list(const float *list, int len, float cur, int dir) {
    // compare with a tolerance, since these are exact decimals in binary floats
    const float eps = 1e-6f;

    if (dir > 0) {
        for (int i = 0; i < len; i++) {
            if (list[i] > cur + eps) return list[i];
        }
        return list[len - 1];
    }

    for (int i = len - 1; i >= 0; i--) {
        if (list[i] < cur - eps) return list[i];
    }
    return list[0];
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

////////////////////////////////////////////////////////////////////////////////
// EDITING
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Sets the hidden layer count, giving any newly revealed layer a width
 */
static void set_hidden_layers(TrainConfig *cfg, int n) {
    n = clamp_int(n, 0, MAX_HIDDEN_LAYERS);

    for (int i = cfg->num_hidden_layers; i < n; i++) {
        if (cfg->hidden_layer_dims[i] == 0) cfg->hidden_layer_dims[i] = form_dim_choices[3];
    }

    cfg->num_hidden_layers = n;
}

/**
 * @brief Applies a left/right step to the field under the cursor
 */
static void step_field(TrainConfig *cfg, Field field, int dir) {
    switch (field.kind) {
        case F_LAYERS:
            set_hidden_layers(cfg, cfg->num_hidden_layers + dir);
            break;

        case F_DIM:
            cfg->hidden_layer_dims[field.layer] =
                step_int_list(form_dim_choices, FORM_DIM_CHOICES,
                              cfg->hidden_layer_dims[field.layer], dir);
            break;

        case F_EPOCHS:
            cfg->num_epochs = step_int_list(epoch_presets,
                                            (int)(sizeof(epoch_presets) / sizeof(int)),
                                            cfg->num_epochs, dir);
            break;

        case F_BATCH:
            cfg->batch_size = step_int_list(batch_presets,
                                            (int)(sizeof(batch_presets) / sizeof(int)),
                                            cfg->batch_size, dir);
            break;

        case F_LR:
            cfg->learning_rate = step_float_list(form_lr_choices, FORM_LR_CHOICES,
                                                 cfg->learning_rate, dir);
            break;
    }
}

/**
 * @brief Appends a typed digit to a numeric field
 *
 * - Overflowing the field's maximum restarts the number from the new digit,
 * which is friendlier than clamping when someone is retyping a value
 */
static void type_digit(TrainConfig *cfg, Field field, int digit) {
    int *value;
    int lo;
    int hi;

    if (field.kind == F_LAYERS) {
        // a single digit is the whole value here, so set it outright
        if (digit <= MAX_HIDDEN_LAYERS) set_hidden_layers(cfg, digit);
        return;
    }

    if (field.kind == F_EPOCHS) {
        value = &cfg->num_epochs;
        lo = FORM_EPOCHS_MIN;
        hi = FORM_EPOCHS_MAX;
    } else if (field.kind == F_BATCH) {
        value = &cfg->batch_size;
        lo = FORM_BATCH_MIN;
        hi = FORM_BATCH_MAX;
    } else {
        return;  // enumerated fields are cycled, not typed
    }

    int next = *value * 10 + digit;
    if (next > hi) next = digit;

    *value = clamp_int(next, lo, hi);
}

/**
 * @brief Removes the last digit of a numeric field
 */
static void backspace_field(TrainConfig *cfg, Field field) {
    if (field.kind == F_EPOCHS) {
        cfg->num_epochs = clamp_int(cfg->num_epochs / 10, FORM_EPOCHS_MIN, FORM_EPOCHS_MAX);
    } else if (field.kind == F_BATCH) {
        cfg->batch_size = clamp_int(cfg->batch_size / 10, FORM_BATCH_MIN, FORM_BATCH_MAX);
    }
}

////////////////////////////////////////////////////////////////////////////////
// DRAWING
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Writes a field's label, current value and allowed range into one row
 *
 * @param[in] row      Row to draw on
 * @param[in] cfg      Configuration being edited
 * @param[in] field    Field to draw
 * @param[in] selected Whether the cursor is on this field
 */
static void draw_field(int row, const TrainConfig *cfg, Field field, bool selected) {
    const char *label = "";
    const char *hint = "";
    char label_buf[32];
    char value[32];
    bool cycles = true;

    switch (field.kind) {
        case F_LAYERS:
            label = "Hidden layers";
            snprintf(value, sizeof(value), "%d", cfg->num_hidden_layers);
            hint = "0 - 3";
            break;

        case F_DIM:
            snprintf(label_buf, sizeof(label_buf), "Layer %d dimension", field.layer + 1);
            label = label_buf;
            snprintf(value, sizeof(value), "%d", cfg->hidden_layer_dims[field.layer]);
            hint = "16 32 64 128 256 512";
            break;

        case F_EPOCHS:
            label = "Epochs";
            snprintf(value, sizeof(value), "%d", cfg->num_epochs);
            hint = "1 - 50";
            cycles = false;
            break;

        case F_BATCH:
            label = "Batch size";
            snprintf(value, sizeof(value), "%d", cfg->batch_size);
            hint = "1 - 1000";
            cycles = false;
            break;

        case F_LR:
            label = "Learning rate";
            snprintf(value, sizeof(value), "%.3f", cfg->learning_rate);
            hint = "0.001 0.01 0.02 0.05 0.10 0.20";
            break;
    }

    if (selected) term_puts(row, COL_LABEL - 2, ">", T_YELLOW, T_BOLD);

    term_puts(row, COL_LABEL, label, T_DEFAULT, selected ? T_BOLD : T_NORMAL);

    // centre the value in its column, wrapped in arrows while it has the cursor
    int len = (int)strlen(value);
    int pad = (VALUE_WIDTH - len) / 2;
    if (pad < 0) pad = 0;

    uint8_t value_attr = selected ? T_INVERSE : T_NORMAL;
    term_fill(row, COL_VALUE, VALUE_WIDTH, ' ', T_DEFAULT, value_attr);
    term_puts(row, COL_VALUE + pad, value, T_DEFAULT, value_attr);

    if (selected) {
        term_puts(row, COL_VALUE - 2, "<", T_YELLOW, T_NORMAL);
        term_puts(row, COL_VALUE + VALUE_WIDTH + 1, ">", T_YELLOW, T_NORMAL);
    }

    term_puts(row, COL_HINT, hint, T_DEFAULT, T_DIM);

    // mark which fields also accept typed digits
    if (!cycles && selected) {
        int col = COL_HINT + (int)strlen(hint) + 2;
        term_puts(row, col, "(or type digits)", T_DEFAULT, T_DIM);
    }
}

/**
 * @brief Draws the derived summary below the fields
 */
static void draw_summary(int row, const TrainConfig *cfg) {
    int col = term_printf(row, COL_LABEL, T_DEFAULT, T_DIM, "Architecture: ");

    col = term_printf(row, col, T_CYAN, T_NORMAL, "784");
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        col = term_printf(row, col, T_DEFAULT, T_DIM, " -> ");
        col = term_printf(row, col, T_CYAN, T_NORMAL, "%d", cfg->hidden_layer_dims[i]);
    }
    col = term_printf(row, col, T_DEFAULT, T_DIM, " -> ");
    term_printf(row, col, T_CYAN, T_NORMAL, "10");

    term_printf(row + 1, COL_LABEL, T_DEFAULT, T_DIM, "Dataset: %s", tui_dataset_name());

    // iteration count is what makes a configuration cheap or brutal, so show it
    int epoch_iters = NOMINAL_TRAIN_SET_SIZE / cfg->batch_size;
    long total = (long)epoch_iters * cfg->num_epochs;

    term_printf(row + 2, COL_LABEL, T_DEFAULT, T_DIM,
                "Iterations: ~%d per epoch, ~%ld total", epoch_iters, total);
}

/**
 * @brief Describes the field under the cursor on the status line
 */
static void draw_field_status(Field field) {
    switch (field.kind) {
        case F_LAYERS:
            tui_draw_status("left/right sets how many hidden layers", T_DEFAULT);
            break;
        case F_DIM:
            tui_draw_status("left/right cycles the layer width", T_DEFAULT);
            break;
        case F_EPOCHS:
            tui_draw_status("left/right steps common values, or type a number", T_DEFAULT);
            break;
        case F_BATCH:
            tui_draw_status("left/right steps common values, or type a number", T_DEFAULT);
            break;
        case F_LR:
            tui_draw_status("left/right cycles the learning rate", T_DEFAULT);
            break;
    }
}

static void draw_form(const TrainConfig *cfg, const Field *fields, int count, int cursor) {
    static const Tui_shortcut shortcuts[] = {
        { "^X", "Exit" },
        { "Enter", "Start run" },
        { "Up/Dn", "Field" },
        { "Lt/Rt", "Value" }
    };

    term_clear();
    tui_draw_title("Swift Logic Optimized Platform");

    if (tui_too_small()) {
        tui_draw_too_small();
        term_flush();
        return;
    }

    Tui_layout layout = tui_layout();

    int row = layout.body_row + 1;

    term_puts(row, COL_LABEL, "Configure a training run", T_DEFAULT, T_BOLD);
    row += 2;

    for (int i = 0; i < count; i++) {
        draw_field(row + i, cfg, fields[i], i == cursor);
    }

    int summary_row = row + count + 1;

    // only draw the summary if the body has room for all three of its lines
    if (summary_row + 3 <= layout.body_row + layout.body_rows) {
        draw_summary(summary_row, cfg);
    }

    draw_field_status(fields[cursor]);
    tui_draw_shortcuts(shortcuts, (int)(sizeof(shortcuts) / sizeof(shortcuts[0])));

    term_flush();
}

////////////////////////////////////////////////////////////////////////////////
// SCREEN LOOP
////////////////////////////////////////////////////////////////////////////////

bool tui_run_form(TrainConfig *cfg) {
    Field fields[MAX_FIELDS];
    int cursor = 0;

    for (;;) {
        int count = build_fields(cfg, fields);

        // the field list shrinks when hidden layers are removed
        if (cursor >= count) cursor = count - 1;

        draw_form(cfg, fields, count, cursor);

        // the form is static between keypresses, so block rather than poll
        // note: a resize interrupts the wait and returns K_NONE, which redraws
        int key = term_read_key(-1);

        switch (key) {
            case K_NONE:
                term_check_resize();
                break;

            case K_UP:
                if (cursor > 0) cursor--;
                break;

            case K_DOWN:
                if (cursor < count - 1) cursor++;
                break;

            case '\t':
                cursor = (cursor + 1) % count;
                break;

            case K_LEFT:
                step_field(cfg, fields[cursor], -1);
                break;

            case K_RIGHT:
                step_field(cfg, fields[cursor], +1);
                break;

            case K_BACKSPACE:
            case K_CTRL('H'):
                backspace_field(cfg, fields[cursor]);
                break;

            case K_ENTER:
                return true;

            case K_ESC:
            case K_CTRL('X'):
            case K_CTRL('C'):
                return false;

            case K_CTRL('L'):
                term_force_repaint();
                break;

            default:
                if (key >= '0' && key <= '9') type_digit(cfg, fields[cursor], key - '0');
                break;
        }
    }
}
