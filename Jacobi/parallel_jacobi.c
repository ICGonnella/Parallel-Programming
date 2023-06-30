#include "jacobi_utils.h"

#define ROOT 0
//#define ID_REF 0

int main(int argc, char* argv[]){

  // init MPI
  MPI_Init(&argc, &argv);

  // MPI variables
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // timing variables
  //double t_start, t_end;
  double increment;

  // indexes for loops
  int it;
  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0;
  size_t matrix_dimension = 0;
  float t_comm=0, t_comp=0;
  // check on input parameters
  if(argc != 4) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  int ID_REF = atoi(argv[3]);
  _Bool changed = false;
  // Parallelization variables
  MPI_Comm Comm = MPI_COMM_WORLD;
  if (size>(dimension+2)){
    changed=true;
    MPI_Comm_split(MPI_COMM_WORLD, (rank < (dimension+2)), rank, &Comm);
  } 
  MPI_Comm_rank(Comm, &rank);
  MPI_Comm_size(Comm, &size);

  if (size==dimension+2 || changed==false){

    if (rank==ID_REF) printf("size %zu\n",size);
    int n_row = my_n_row(rank, size, dimension, false);
    int n_halo = my_n_halo(rank, size);

    if (rank==ID_REF) {
      printf("matrix size = %zu\n", dimension);
      printf("number of iterations = %zu\n", iterations);
    }

    // Create local matrices
    matrix_dimension = sizeof(double) * ( n_row + n_halo ) * ( dimension + 2 );
    matrix = ( double* )malloc( matrix_dimension );
    matrix_new = ( double* )malloc( matrix_dimension );

    if (rank==ID_REF) {
      printf("number of rows = %zu\n", n_row);
      printf("number of halos = %zu\n", n_halo);
      printf("number of rows total = %zu\n", my_n_row(rank, size, dimension, true));
    }

    //fill initial values  
    init_mat(matrix, 0.5,rank, size, dimension, true);
    init_mat(matrix_new, 0.5,rank, size, dimension, true);

    // set up boundary 
    increment = 100.0 / ( dimension + 1 );
    init_boundary_conditions(matrix, increment, rank, size, dimension, true);
    init_boundary_conditions(matrix_new, increment, rank, size, dimension, true);
  
#ifdef ACC
    const acc_device_t devtype = acc_get_device_type(); // Device type (e.g. Tesla)
    const int num_devs = acc_get_num_devices(devtype); // Number of devices per node
    acc_set_device_num(rank % num_devs, devtype); // To run on multiple nodes
    acc_init(devtype);

#pragma acc enter data copyin(matrix[:my_last_element_idx_loc(rank, size, dimension,true)], matrix_new[:my_last_element_idx_loc(rank, size, dimension,true)])
#endif

    // start algorithm
    //t_start = MPI_Wtime();
  
    for( it = 0; it < iterations; ++it ){
   
      evolve(matrix, matrix_new, rank, size, dimension, Comm, &t_comp, &t_comm);
      // swap the pointers
      tmp_matrix = matrix;
      matrix = matrix_new;
      matrix_new = tmp_matrix;
    
    }
    
    //t_end = MPI_Wtime();
    MPI_Barrier(Comm);
    
#ifdef ACC
#pragma acc exit data copyout(matrix[:my_last_element_idx_loc(rank, size, dimension,true)], matrix_new[:my_last_element_idx_loc(rank, size, dimension,true)])
#endif

  
    //printf( "\nelapsed time = %f seconds\n", t_end - t_start );

    //save_result(matrix, rank, size, dimension, Comm);

  
    save_times(rank, size, dimension, iterations, t_comm, t_comp, Comm);
    
  
    free( matrix );
    free( matrix_new );
  }
  MPI_Finalize();
  return 0;
}
