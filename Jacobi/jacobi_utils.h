#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// save matrix to file
void save_gnuplot( double *M, size_t dim );

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t dimension );

// return the elapsed time
double seconds( void );

// initialize matrices
void init_mat(double * matrix, double val, int row, int dimension, int rank);

