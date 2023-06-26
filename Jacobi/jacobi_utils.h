#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdbool.h>

#ifdef ACC
#include <openacc.h>
#endif

// print matrix
void print_mat(double* matrix, int n_row, int n_col);

// save matrix to file
void save_gnuplot( double *M, int n_row, int n_col, double offset, int rank );

// return the elapsed time
double seconds( void );

// compute the number of halo layers assigned to a given process
int my_n_halo(int rank, int size);

// compute the number of rows assigned to a given process
int my_n_row(int rank, int size, int dim, _Bool halo);

// compute the global index of the first row assigned to a given process
int my_first_row_idx_glob(int rank, int size, int dim, _Bool halo);

// compute the global index of the last row assigned to a given process
int my_last_row_idx_glob(int rank, int size, int dim, _Bool halo);

// compute the global index of the first row assigned to a given process
int my_first_row_idx_loc(int rank, _Bool halo);

// compute the global index of the last row assigned to a given process
int my_last_row_idx_loc(int rank, int size, int dim, _Bool halo);

// compute the global index of the first element assigned to a given process
int my_first_element_idx_glob(int rank, int size, int dim, _Bool halo);

// compute the global index of the last element assigned to a given process
int my_last_element_idx_glob(int rank, int size, int dim, _Bool halo);

// compute the global index of the first element assigned to a given process
int my_first_element_idx_loc(int rank, int dim, _Bool halo);

// compute the global index of the last element assigned to a given process
int my_last_element_idx_loc(int rank, int size, int dim, _Bool halo);

//check if a given global index corresponds to a local element for a given process
int is_my_element(int glob, int rank, int size, int dim, _Bool halo);

// from local coordinates to global ones
int global_from_local(int loc, int rank, int size, int dim);

// from global coordinates to local ones
int local_from_global(int glob, int rank, int size, int dim);

// compute the local indices of the central elements for a given process
// if halo==true the first SendRecive operation is not necessary
//int* central_elements_idx(int rank, int size, int dim, int n_ce, _Bool halo);

// initialize matrices
void init_mat(double * matrix, double val, int rank, int size, int dim, _Bool halo);

// compute the local indices of the border elements for a given process
int* border_elements_idx_loc(int rank, int size, int dim, int n_be, _Bool halo);

// impose the border conditions
void init_border_conditions(double* matrix, double increment, int rank, int size, int dim, _Bool halo);

// compute the neighbours of a given process
int neighbor(int rank, int size, int up);

// update the halos
void update_halos(double * matrix, int rank, int size, int dim, MPI_Comm Comm);

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, int rank, int size, int dim, MPI_Comm Comm );

// create offset vector
int* set_offset(int rank, int size, int dim);

// save results on the file solution.dat
void save_result(double* matrix, int rank, int size, int dim, MPI_Comm Comm);
