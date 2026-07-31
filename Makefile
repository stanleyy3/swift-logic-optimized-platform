CC = gcc
# note: -pthread is for the interface only - the UI runs on the main thread while
#       training runs on a worker. The compute flags are deliberately left alone:
#       no -fopenmp and no -O3, because this branch is the slow reference point.
CFLAGS = -O2 -pthread -Wall -Wextra -MMD -MP
LDLIBS = -lm -pthread
TARGET = train

# optionally add profiling flags
ifeq ($(PROFILE), 1)
CFLAGS += -g -pg
endif

BUILD_DIR = build

# get .o and .d dependencies
# note: all terminal user interface code lives in tui/
SRCS = $(wildcard *.c) $(wildcard tui/*.c)
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS = $(SRCS:%.c=$(BUILD_DIR)/%.d)

# default
all: $(TARGET)

# create main executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)

# compile .c files into .o files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
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
