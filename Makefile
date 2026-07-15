CC = gcc
CFLAGS = -O1 -g -pg -MMD -MP
LDLIBS = -lm
TARGET = train

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
	rm -rf $(TARGET) $(BUILD_DIR)
