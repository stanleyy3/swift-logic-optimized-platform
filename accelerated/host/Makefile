CC = gcc

CFLAGS = -fopenmp -O3 -Wall -Wextra -MMD -MP -march=native
LDLIBS = -lm

# requires XRT to be installed (e.g. build directly on the KV260, which
# already has it). Prefer pkg-config since XRT's install prefix varies by
# image; fall back to XILINX_XRT (override if it's installed elsewhere,
# e.g. `make XILINX_XRT=/usr`)
ifeq ($(shell pkg-config --exists xrt && echo yes),yes)
CFLAGS += $(shell pkg-config --cflags xrt)
LDLIBS += $(shell pkg-config --libs xrt) -luuid
else
XILINX_XRT ?= /opt/xilinx/xrt
CFLAGS += -I$(XILINX_XRT)/include
LDLIBS += -L$(XILINX_XRT)/lib -lxrt_coreutil -luuid
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

# clean target
.PHONY: clean
clean:
	rm -rf $(TARGET) $(BUILD_DIR) gmon.out
