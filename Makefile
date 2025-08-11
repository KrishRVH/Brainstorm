CC = gcc
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS = -shared -lm

ifeq ($(OS),Windows_NT)
    TARGET = immolate.dll
    LDFLAGS += -Wl,--out-implib,immolate.lib
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        TARGET = immolate.dylib
        LDFLAGS = -dynamiclib -lm
    else
        TARGET = immolate.so
    endif
endif

all: $(TARGET)

$(TARGET): immolate.c
	$(CC) $(CFLAGS) immolate.c -o $(TARGET) $(LDFLAGS)

test: immolate.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST immolate.c -o immolate_test -lm
	./immolate_test

clean:
	rm -f $(TARGET) immolate_test immolate.lib

.PHONY: all test clean