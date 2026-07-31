CC = gcc

CFLAGS = -fopenmp -O3 -Wall -Wextra -MMD -MP -march=native
LDLIBS = -lm

# `make MODEL=1` swaps the device out for the bit-exact software model in
# device_model.c, so the host builds and runs anywhere - no board, no XRT
ifeq ($(MODEL), 1)
CFLAGS += -DFPGA_MODEL=1
else
# requires XRT to be installed (e.g. build directly on the KV260, which
# already has it). Prefer pkg-config since XRT's install prefix varies by
# image; fall back to XILINX_XRT (override if it's installed elsewhere,
# e.g. `make XILINX_XRT=/usr`)
ifeq ($(shell pkg-config --exists xrt && echo yes),yes)
CFLAGS += $(shell pkg-config --cflags xrt)
LDLIBS += $(shell pkg-config --libs xrt) -luuid
else
# Two install layouts exist and they put xrt.h in different places:
#
#   AMD tarball   /opt/xilinx/xrt/include/xrt.h  + include/xrt/xrt_kernel.h
#   Ubuntu `xrt`  /usr/include/xrt/xrt.h         + /usr/include/xrt/xrt_kernel.h
#
# The sources use both `#include <xrt.h>` and `#include <xrt/xrt_kernel.h>`, so
# put <prefix>/include *and* <prefix>/include/xrt on the search path -- that
# satisfies both spellings under either layout. The KV260's Certified Ubuntu
# image ships the Ubuntu package (`dpkg -l xrt` -> 2.18.0-0ubuntu1, prefix
# /usr), which is why the prefix is auto-detected rather than assumed.
XILINX_XRT ?= $(if $(wildcard /opt/xilinx/xrt/include),/opt/xilinx/xrt,/usr)
CFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_XRT)/include/xrt
# no -L for a /usr prefix: the loader's default paths already cover the
# multiarch lib dir the Ubuntu package installs into
LDLIBS += $(if $(filter /usr,$(XILINX_XRT)),,-L$(XILINX_XRT)/lib) -lxrt_coreutil -luuid
endif
endif

TARGET = traina

# optionally add profiling flags
ifeq ($(PROFILE), 1)
CFLAGS += -g -pg
endif

BUILD_DIR = build

# get .o and .d dependencies
SRCS = $(wildcard *.c)
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS = $(SRCS:%.c=$(BUILD_DIR)/%.d)

# default
all: $(TARGET)

# create main executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)

# compile .c files into .o files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# pull in auto-generated header dependencies
-include $(DEPS)

# Checks the FPGA matmul against a plain float32 reference, running against the
# software model of the device rather than hardware, so it needs neither a board
# nor XRT. Always built in model mode regardless of config.h.
#
# No -fopenmp: the FPGA path is serial, and leaving it off lets this build with
# a stock Apple clang. The one OpenMP pragma is in the CPU matmul, which this
# does not exercise.
TEST_TARGET = test_matmul
TEST_SRCS = tests/test_matmul.c mat_ops.c quant.c device_model.c fpga.c

.PHONY: test
test:
	$(CC) -O2 -Wall -Wextra -Wno-unknown-pragmas -I. \
	    -DFPGA_MODEL=1 -DUSE_FPGA_MATMUL=1 \
	    -o $(TEST_TARGET) $(TEST_SRCS) -lm
	./$(TEST_TARGET)

# clean target
.PHONY: clean
clean:
	rm -rf $(TARGET) $(TEST_TARGET) $(BUILD_DIR) gmon.out
