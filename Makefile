CC = gcc
CFLAGS = -O1 -g -pg -MMD -MP
LDLIBS = -lm
TARGET = mlp

# get .o and .d dependencies
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)

# default
all: $(TARGET)

# create main executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)

# compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# pull in auto-generated header dependencies
-include $(DEPS)

# clean target
.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
