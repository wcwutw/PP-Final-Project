CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS = 

# Source files
SOURCES = main.c dna_alignment.c brute_force.c kmp.c boyer_moore.c rabin_karp.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = dna_align

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files to object files
%.o: %.c dna_alignment.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run

