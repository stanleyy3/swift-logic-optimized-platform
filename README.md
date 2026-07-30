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
│   │   ├── device_model.c/.h   bit-exact software model of the array
│   │   ├── fpga.c/.h           device lifecycle and block launches
│   │   ├── quant.c/.h          float32 <-> float16 conversion and scaling
│   │   ├── tests/
│   │   │   └── test_matmul.c
│   │   └── ...
│   └── device/
│       ├── CONTROL_INTERFACE.md  what the packaged RTL kernel must expose
│       ├── rtl/
│       │   └── ...
│       └── testbench/
│           └── ...
├── .gitignore
└── README.md
```

## Dependencies

The C standard library, plus XRT for the accelerated version when it is built
against real hardware (`make MODEL=1` needs neither XRT nor a board).

## Usage

Inside subdirectory of version you want to run (`baseline/` or `accelerated/host/`):

- To run normally: `make`
- To run with `gprof`: `make PROFILE=1`
  - Note: the executable is called `traina` for the accelerated version and `trainb` for the baseline version

- To clean build files: `make clean`

### Accelerated version

`accelerated/host/config.h` holds two switches:

- `USE_FPGA_MATMUL` sends the training pass's matmuls to the systolic array
  instead of the CPU. Off by default, so the two paths can be diffed.
- `FPGA_MODEL` runs the FPGA path against the bit-exact software model of the
  array in `device_model.c` rather than hardware. `make MODEL=1` sets it, which
  also drops the XRT link flags - useful for developing the host off the board.

- To check the FPGA matmul against a float32 reference: `make test`
  - Runs in model mode, so it needs neither XRT nor a board

The AXI4-Lite register map, DataMover settings and packaging metadata the RTL
kernel has to expose are specified in `accelerated/device/CONTROL_INTERFACE.md`.

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
