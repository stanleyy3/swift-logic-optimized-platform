/**
 * form.h - Hyperparameter collection screen
 *
 * `tui_run_form` is declared in tui.h; this header exists so the screen's
 * allowed value sets are stated in one place and can be shared with the
 * dashboard's summary line.
 */

#ifndef _TUI_FORM_H_
#define _TUI_FORM_H_

// hidden layer dimensions the user may pick from
#define FORM_DIM_CHOICES 6
extern const int form_dim_choices[FORM_DIM_CHOICES];

// learning rates the user may pick from
#define FORM_LR_CHOICES 6
extern const float form_lr_choices[FORM_LR_CHOICES];

// inclusive bounds on the freely-typed fields
#define FORM_EPOCHS_MIN 1
#define FORM_EPOCHS_MAX 50
#define FORM_BATCH_MIN 1
#define FORM_BATCH_MAX 1000

#endif
