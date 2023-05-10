CC = mpicc
NVCC=nvcc

CFLAGS=-Wall -O3

OPENBLAS=${OPENBLAS_HOME}

TARGET=matrix_mul.o

SOURCES= matrix_mul.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE = my_program

ifeq ($(USE_CBLAS),1)
    LDFLAGS=-DUSE_CBLAS -I $(OPENBLAS)/include/ -L $(OPENBLAS)/lib -lopenblas
endif

ifeq ($(USE_GPU),1)
        LDFLAGS+=-DUSE_GPU
        OBJECTS+=gpu.o
        LDLIBS+=-L${CUDA_HOME}/lib64/ -lcudart -lcublas
endif

all: $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

gpu.o: gpu.cu
	$(NVCC) -c $< -o $@

clean:
	rm $(OBJECTS) 
