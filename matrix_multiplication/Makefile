CC = mpicc

CFLAGS=-Wall -O3

OPENBLAS=${OPENBLAS_HOME}

TARGET=matrix_mul.o

SOURCES= matrix_mul.c 
OBJECTS=$(SOURCES:.c=.o)

ifeq ($(USE_CBLAS),1)
    LDFLAGS=-DUSE_CBLAS -I $(OPENBLAS)/include/ -L $(OPENBLAS)/lib -lopenblas
endif

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

clean: rm -f $(OBJECTS) $(TARGET)
