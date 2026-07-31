# Swift Logic Optimized Platform

## Project Structure

```
./
├── data/
│   └── mnist/
│       ├── test-images.idx3-ubyte
│       ├── test-labels.idx1-ubyte
│       ├── train-images.idx3-ubyte
│       └── train-labels.idx1-ubyte
├── tui/                  all terminal user interface code
│   ├── term.c/.h         raw mode, cell grid, damage-diff renderer, key input
│   ├── tui.c/.h          lifecycle and the nano-style chrome
│   ├── form.c/.h         hyperparameter screen
│   ├── dash.c/.h         live training dashboard
│   └── plot.c/.h         braille curve plotting
├── .gitignore
├── config.h              compile-time knobs
├── data_ops.c
├── data_ops.h
├── main.c
├── Makefile
├── mat_ops.c
├── mat_ops.h
├── metrics.c             thread-safe seam between a training run and the UI
├── metrics.h
├── model.c
├── model.h
├── README.md
├── train.c
└── train.h
```

The training code never touches the terminal. It publishes numbers through
`metrics.h`, and everything that reads from or writes to the terminal lives in
`tui/`. A run happens on a worker thread while the interface draws on the main
thread. That matters more here than anywhere: the matrix multiply on this branch
is untiled and single-threaded, so one iteration of a wide model takes seconds,
and the interface still answers the keyboard in well under a millisecond
throughout.

## Dependencies

The C standard library, plus POSIX (`termios`, `poll`, `ioctl`) and pthreads for
the interface. Both come with gcc and glibc, so there is still nothing to install.

Note that `-pthread` is for the interface only. The compute flags are left
deliberately unoptimized - no `-fopenmp`, no `-O3` - because this branch is the
slow reference point for the accelerator comparison.

## Usage

- To run normally: `make`, then `./train` from this directory - the dataset paths
are relative to it. An interactive terminal is required.
To run with profiling: `make PROFILE=1`

- To clean build files: `make clean`

### Keys

| Screen | Key | Does |
| --- | --- | --- |
| Form | up / down | move between fields |
| Form | left / right | change the field's value |
| Form | digits | type a value directly |
| Form | Enter | start the run |
| Form | `^X` | exit |
| Dashboard | `^X` | stop the run, then exit |
| Dashboard | `^P` | pause / resume |
| Dashboard | `^R` | stop and configure another run |
| Dashboard | `^C` | quit |
| Dashboard | `^L` | redraw |

Curves are drawn with braille glyphs, which need a UTF-8 locale and a font that
has them. If `LANG` does not look like UTF-8 the plot falls back to ASCII;
`TUI_ASCII_FALLBACK` in `config.h` forces the fallback on.

## Performance notes

- There are a number of unoptimized parts of the program, to say the least. We intentionally leave these to focus more on the hardware side of the project.
  - E.g., there are multiple instances of column-major access patterns (for flattened 2d matrices) that could be row-major accesses instead.

## Correctness verification

 - Final test accuracy (unseeded): 0.9257
   - 1 hidden layer with 10 neurons
   - 50 epochs
   - 1000 batch size
   - 0.1 learning rate
