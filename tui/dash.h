/**
 * dash.h - Live training dashboard
 *
 * `tui_run_training` is declared in tui.h. This header only exposes the body
 * layout constants, so that the minimum terminal size in config.h and the space
 * the dashboard actually needs stay in agreement.
 */

#ifndef _TUI_DASH_H_
#define _TUI_DASH_H_

// body rows the dashboard spends before the graph:
// two stat lines, the progress bar and a blank separator
#define DASH_HEADER_ROWS 4

#endif
