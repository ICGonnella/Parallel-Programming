#include "jacobi_utils.h"

void print_mat(double* matrix, int n_row, int n_col){
  printf("--------------------------------------------------\n");
  for(int i=0;i<n_col*n_row;++i){
    if ((i%n_col == 0) && (i!=0))
      printf("\n");
    printf("%.3e ", matrix[i]);
  }
  printf("\n--------------------------------------------------\n");
}

void save_gnuplot( double *M, int n_row, int n_col, double offset, int rank ){

  const double h = 0.1;
  FILE *file;

  if (rank==0) file = fopen("solution.dat", "w");
  else file = fopen("solution.dat", "a");

  for( int i = 0; i < n_row; ++i )
    for( int j = 0; j < n_col; ++j )
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * (i+offset), M[ i*n_col + j ] );

  fclose( file );

}

double seconds(){

  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;

}

int my_n_halo(int rank, int size){
  if (size==1)
    return 0;
  else
    return (rank==0 || rank==size-1) ? 1 : 2;
}

int my_n_row(int rank, int size, int dim, _Bool halo){
  return (rank >= (dim+2)%size) ? ((dim+2)/size + (halo)*my_n_halo(rank, size)) : ((dim+2)/size +1 + (halo)*my_n_halo(rank, size));
}

int my_first_row_idx_glob(int rank, int size, int dim, _Bool halo){
  int count=0;
  for (int i=0;i<rank;++i)
    count += my_n_row(i, size, dim, false);
  if (halo && rank>0)
    count -= 1;
  return count;
}

int my_last_row_idx_glob(int rank, int size, int dim, _Bool halo){
  return my_first_row_idx_glob(rank, size, dim, halo) + my_n_row(rank, size, dim, halo) - 1;
}

int my_first_row_idx_loc(int rank, _Bool halo){
  if (halo)
    return 0;
  else
    return rank==0 ? 0 : 1;
}

int my_last_row_idx_loc(int rank, int size, int dim, _Bool halo){
  return my_first_row_idx_loc(rank, halo) + my_n_row(rank, size, dim, halo) - 1;
}

int my_first_element_idx_glob(int rank, int size, int dim, _Bool halo){
  return my_first_row_idx_glob(rank, size, dim, halo)*(dim+2);
}

int my_last_element_idx_glob(int rank, int size, int dim, _Bool halo){
  return (my_last_row_idx_glob(rank, size, dim, halo)+1)*(dim+2) - 1;
}

int my_first_element_idx_loc(int rank, int dim, _Bool halo){
  return my_first_row_idx_loc(rank, halo)*(dim+2);
}

int my_last_element_idx_loc(int rank, int size, int dim, _Bool halo){
  return (my_last_row_idx_loc(rank, size, dim, halo)+1)*(dim+2) - 1;
}

int is_my_element(int glob, int rank, int size, int dim, _Bool halo){
  return ((glob >= my_first_element_idx_glob(rank, size, dim, halo)) && (glob <= my_last_element_idx_glob(rank, size, dim, halo)));
}

int global_from_local(int loc, int rank, int size, int dim){
  return my_first_element_idx_glob(rank, size, dim, true)+loc;
}

int local_from_global(int glob, int rank, int size, int dim){
  if (is_my_element(glob, rank, size, dim, true)==false)
    return -1;
  else 
    return glob - my_first_element_idx_glob(rank, size, dim, true);
}

void init_mat(double * matrix, double val, int rank, int size, int dim, _Bool halo){
  memset(matrix, 0, (dim*2)*(my_n_row(rank, size, dim,halo)));
  int n_ce = (my_n_row(rank, size, dim, halo) - is_my_element(0, rank, size, dim, halo) - is_my_element((dim+2)*(dim+2)-1, rank, size, dim, halo))*dim;
  int first_e = my_first_element_idx_loc(rank, dim, halo) + 1 + is_my_element(0, rank, size, dim, halo)*(dim+2);
  
  for (int i=0; i<n_ce; ++i)
    matrix[first_e + (i/dim)*(dim+2) +(i%dim)] = val;
}

void init_boundary_conditions(double* matrix, double increment, int rank, int size, int dim, _Bool halo){

  // first column vertical boundary conditions
  int n_be = my_n_row(rank, size, dim, halo);
  int first_increment_factor = my_first_row_idx_glob(rank, size, dim, halo);
  int first_e = my_first_element_idx_loc(rank, dim, halo);
  for (int i=0; i<n_be;++i)
    matrix[first_e + (dim+2)*i] = increment*(i+first_increment_factor);

  // last row horizontal boundary condition
  if (is_my_element((dim+2)*(dim+2)-1, rank, size, dim, halo)){
    int start = my_last_element_idx_loc(rank, size, dim, true) - (dim+1);
    for (int i=0; i<(dim+2); ++i)
      matrix[start+i] = increment*(dim+1-i);
  }
}

int neighbor(int rank, int size, int up){
  if (up)
    return rank==0 ? -1 : rank-1;
  else
    return rank==(size-1) ? -1 : rank+1;
}


void update_halos(double * matrix, int rank, int size, int dim, MPI_Comm Comm){
  int up_neigh = neighbor(rank, size,true);
  int down_neigh = neighbor(rank, size, false);
  
  if (up_neigh>=0)
    MPI_Sendrecv(matrix + my_first_element_idx_loc(rank, dim, false),
		 dim + 2, MPI_DOUBLE, up_neigh, 0,
		 matrix + my_first_element_idx_loc(rank, dim, true),
		 dim + 2, MPI_DOUBLE, up_neigh, 0, Comm, MPI_STATUS_IGNORE);
  
  if (down_neigh>=0)
    MPI_Sendrecv(matrix + my_last_element_idx_loc(rank, size, dim, false) - (dim + 1),
		 dim + 2, MPI_DOUBLE, down_neigh, 0,
		 matrix + my_last_element_idx_loc(rank, size, dim, true) - (dim + 1),
		 dim + 2, MPI_DOUBLE, down_neigh, 0, Comm, MPI_STATUS_IGNORE);
  
}

void evolve( double * matrix, double *matrix_new, int rank, int size, int dim, MPI_Comm Comm, float *t_comp, float *t_comm ){
  //This will be a row dominant program

  double tic, toc;

  //--------------------- exchange halos and update t_comm-----------------------
  tic = MPI_Wtime();
  #ifdef ACC
  int start_1 = my_first_element_idx_loc(rank, dim, false);
  int start_2 = my_last_element_idx_loc(rank, size, dim, false)-(dim+1);
  if (rank>0){
    #pragma acc update host(matrix[start_1:dim+1])
  }
  if (rank<size-1){
    #pragma acc update host(matrix[start_2:dim+1])
  }
  #endif

  update_halos(matrix, rank, size, dim, Comm);
  
  #ifdef ACC
  tic = MPI_Wtime();
  start_1 = my_first_element_idx_loc(rank, dim, true);
  start_2 = my_last_element_idx_loc(rank, size, dim, true)-(dim+1);
  if (rank>0){
    #pragma acc update device(matrix[start_1:dim+1])
  }
  if (rank<size-1){
#pragma acc update device(matrix[start_2:dim+1])
  }
  #endif
  toc = MPI_Wtime();
  *t_comm += toc-tic;

  //-------------------update the values and t_comp-------------------------------
  tic = MPI_Wtime();
  int start = my_first_element_idx_loc(rank, dim, false)+1;
  int n_row = my_n_row(rank, size, dim, false);
  if (rank==0){
    start += dim+2;
    n_row -= 1;
  }
  if (rank==size-1)
    n_row -= 1;
  int idx;

  #ifdef ACC
  #pragma acc parallel loop present(matrix[:my_last_element_idx_loc(rank, size, dim,true)], matrix_new[:my_last_element_idx_loc(rank, size, dim,true)])
  #endif
  for (int i=0; i<n_row*dim; ++i) {
    idx = start + i%dim + (dim+2)*(i/dim);
    matrix_new[idx] = (0.25) *
      ( matrix[idx - (dim + 2)] +
	matrix[idx + (dim + 2)] +
	matrix[idx - 1] +
	matrix[idx + 1]);
  }
  toc = MPI_Wtime();
  *t_comp += toc - tic;
}

int* set_offset(int rank, int size, int dim){
  int* offset = (int*)malloc(size * sizeof(int));
  offset[0] = 0;
  if(size>1){
    offset[1] = my_n_row(0,size,dim,false) + 1;
    for (int i = 2; i < size; i++) offset[i] = offset[i - 1] + my_n_row(i,size,dim,false);
  }
  return offset;
}

void save_result(double* matrix, int rank, int size, int dim, MPI_Comm Comm){

  int* offset = set_offset(rank, size, dim);
  
  for (int i = 0; i < size; ++i) {
    if (rank == i) {
      save_gnuplot( matrix+(rank>0)*(dim+2), my_n_row(rank, size, dim, false), dim+2, offset[i], rank);
      printf("DONE rank %zu\n", rank);
    }
    MPI_Barrier(Comm);  // Synchronize before the next process prints
  }

}

void save_times(int rank, int size, int dim, int iter, double t_comm, double t_comp, MPI_Comm Comm){

  FILE *file_comm, *file_comp;

  for (int i = 0; i < size; ++i) {
    if(rank==i) {
      file_comm = fopen("time_comm.dat", "a");
      file_comp = fopen("time_comp.dat", "a");
      fprintf(file_comm, "%zu\t%zu\t%zu\t%zu\t%f\n", rank, size, dim, iter, t_comm );
      fprintf(file_comp, "%zu\t%zu\t%zu\t%zu\t%f\n", rank, size, dim, iter, t_comp );
      fclose(file_comm);
      fclose(file_comp);
    }
    MPI_Barrier(Comm);
  }
  
}
