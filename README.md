# Swift Logic Optimized Platform

## Project Structure

```
./
├── data/
│   └── mnist/
│       ├── test-images.idx3-ubyte
│       ├── test-labels.idx1-ubyte
│       ├── train-images.idx3-ubyte
│       └── train-labels.idx1-ubyte
├── .gitignore
├── data_ops.c
├── data_ops.h
├── main.c
├── Makefile
├── mat_ops.c
├── mat_ops.h
├── model.c
├── model.h
├── README.md
├── train.c
└── train.h
```

## Dependencies

Just the C standard library!

## Usage

- To run normally: `make`
To run with profiling: `make PROFILE=1`

- To clean build files: `make clean`

## Performance notes

- There are a number of unoptimized parts of the program, to say the least. We intentionally leave these to focus more on the hardware side of the project.
  - E.g., there are multiple instances of column-major access patterns (for flattened 2d matrices) that could be row-major accesses instead.

## Correctness verification*

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
