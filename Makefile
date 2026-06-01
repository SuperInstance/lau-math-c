# lau-math-c Makefile
# Builds static library + test binary
# Targets: native, aarch64-linux-gnu (Jetson)
#
# Copyright (c) 2025 Lau Ecosystem

CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -std=c99 -Wall -Wextra -pedantic -O3 -ffast-math
LDFLAGS ?= -lm

# Architecture detection
ARCH ?= native

ifeq ($(ARCH),aarch64)
    CC     = aarch64-linux-gnu-gcc
    AR     = aarch64-linux-gnu-ar
    CFLAGS += -march=armv8-a -mtune=cortex-a72
else ifeq ($(ARCH),native)
    CFLAGS += -march=native
else ifeq ($(ARCH),riscv64)
    CC     = riscv64-linux-gnu-gcc
    AR     = riscv64-linux-gnu-ar
    CFLAGS += -march=rv64gc -mabi=lp64d
endif

# SIMD flags
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
ifeq ($(findstring x86_64,$(UNAME_M)),x86_64)
    CFLAGS += -mavx2 -mfma
endif
ifeq ($(findstring aarch64,$(UNAME_M)),aarch64)
    CFLAGS += -ftree-vectorize
endif

SRCS = lau_matrix.c lau_laplacian.c lau_dirichlet.c lau_dirac.c \
       lau_conservation.c lau_thermodynamic.c lau_agent.c
OBJS = $(SRCS:.c=.o)
TEST_SRC = test_main.c
TEST_BIN = lau_math_test
LIB      = liblaumath.a

.PHONY: all clean test install jetson riscv

all: $(LIB) $(TEST_BIN)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

jetson:
	$(MAKE) ARCH=aarch64 all

riscv:
	$(MAKE) ARCH=riscv64 all

install: $(LIB)
	mkdir -p $(DESTDIR)/usr/lib $(DESTDIR)/usr/include
	cp $(LIB) $(DESTDIR)/usr/lib/
	cp *.h $(DESTDIR)/usr/include/

clean:
	rm -f $(OBJS) $(LIB) $(TEST_BIN)
