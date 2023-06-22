#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// save matrix to file
void save_gnuplot( double *M, size_t dim );

// return the elapsed time
double seconds( void );

// compute the number of halo layers assigned to a given process
int my_n_halo(int rank, int size);

// compute the number of rows assigned to a given process
int my_n_row(int rank, int size, int dim, int halo);

// compute the global index of the first row assigned to a given process
int my_first_row_idx_glob(int rank, int size, int dim, int halo);

// compute the global index of the last row assigned to a given process
int my_last_row_idx_glob(int rank, int size, int dim, int halo);

// compute the global index of the first row assigned to a given process
int my_first_row_idx_loc(int rank, int halo);

// compute the global index of the last row assigned to a given process
int my_last_row_idx_loc(int rank, int size, int dim, int halo);

// compute the global index of the first element assigned to a given process
int my_first_element_idx_glob(int rank, int size, int dim, int halo);

// compute the global index of the last element assigned to a given process
int my_last_element_idx_glob(int rank, int size, int dim, int halo);

// compute the global index of the first element assigned to a given process
int my_first_element_idx_loc(int rank, int dim, int halo);

// compute the global index of the last element assigned to a given process
int my_last_element_idx_loc(int rank, int size, int dim, int halo);

//check if a given global index corresponds to a local element for a given process
int is_my_element(int glob, int rank, int size, int dim, int halo);

// from local coordinates to global ones
int global_from_local(int loc, int rank, int size, int dim);

// from global coordinates to local ones
int local_from_global(int glob, int rank, int size, int dim);

// compute the local indices of the central elements for a given process
// if halo==true the first SendRecive operation is not necessary
int* central_elements_idx(int rank, int size, int dim, int n_ce, int halo);

// initialize matrices
void init_mat(double * matrix, double val, int rank, int size, int dim, int halo);

// compute the neighbours of a given process
int neighbor(int rank, int size, int up);

// update the halos
void update_halos(double * matrix, int rank, int size, int dim);

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t rank, int size, int dim );
