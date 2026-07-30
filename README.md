# Swift Logic Optimized Platform

## Project Structure

```
./
├── data/
│   ├── mnist/
│   │   ├── test-images.idx3-ubyte
│   │   ├── test-labels.idx1-ubyte
│   │   ├── train-images.idx3-ubyte
│   │   └── train-labels.idx1-ubyte
│   └── fashion_mnist/
│       ├── test-images.idx3-ubyte
│       ├── test-labels.idx1-ubyte
│       ├── train-images.idx3-ubyte
│       └── train-labels.idx1-ubyte
├── baseline/
│   ├── Makefile
│   ├── config.h
│   ├── data_ops.c
│   ├── data_ops.h
│   ├── main.c
│   ├── mat_ops.c
│   ├── mat_ops.h
│   ├── model.c
│   ├── model.h
│   ├── train.c
│   └── train.h
├── accelerated/
│   ├── host/
│   │   └── ...
│   └── device/
│       ├── rtl/
│       │   └── ...
│       └── testbench/
│           └── ...
├── .gitignore
└── README.md
```

## Dependencies

Just the C standard library!

## Usage

Inside subdirectory of version you want to run (`baseline/` or `accelerated/host/`):

- To run normally: `make`
- To run with `gprof`: `make PROFILE=1`
  - Note: the executable is called `traina` for the accelerated version and `trainb` for the baseline version

- To clean build files: `make clean`

## Performance notes

There are various software optimizations left unaddressed to spend more time on the hardware side of the project.

## Approximate correctness verification*

### MNIST

```
Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz (Andrew machines):

|-------------------------|---------------------|
| Configuration           | Final test accuracy |
|-------------------------|---------------------|
| 784 -> 128 -> 10        | ~0.9720             |
| batch size: 100         |                     |
| learning rate: 0.1      |                     |
| epochs: 10              |                     |
|-------------------------|---------------------|
| 784 -> 256 -> 128 -> 10 | ~0.9817             |
| batch size: 32          |                     |
| learning rate: 0.05     |                     |
| epochs: 15              |                     |
|-------------------------|---------------------|
| 784 -> 512 -> 512 -> 10 | ~0.9831             |
| batch size: 128         |                     |
| learning rate: 0.2      |                     |
| epochs: 20              |                     |
|-------------------------|---------------------|
```

## Other

### Constrain OMP multi-threading for Cortex A53 core count

```
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```
