# inputs for the executables

ARGS = 10 100
ID_REF = 0
NP = 1

CC = mpicc
CFLAGS = -O3

SRCS = $(wildcard *.c)
EXE=$(SRCS:.c=.x)
EXECUTABLE = jacobi

all: $(EXECUTABLE)

$(EXECUTABLE): $(EXE)
	$(CC) $(CFLAGS) $^ -o $@

%.x: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: 
	mpirun -np $(NP) $(EXECUTABLE) $(ARGS) $(ID_REF)

acc: CFLAGS += -DACC -acc -Minfo=accel -ta=tesla
acc: clean
acc: $(EXECUTABLE)

clean:
	rm -f *~ $(EXECUTABLE) $(EXE) solution.dat

plot:
	@gnuplot -p plot.plt

.PHONY: clean plot all


