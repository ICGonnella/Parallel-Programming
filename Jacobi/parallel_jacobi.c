# include "jacobi_utils.h"

#define ROOT 0
#define ID_REF 0

int main(int argc, char* argv[]){

  // init MPI
  MPI_Init(&argc, &argv);

  // MPI variables
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // timing variables
  double t_start, t_end, increment;

  // indexes for loops
  size_t i, j, it;
  
  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  if(argc != 5) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  row_peek = atoi(argv[3]);
  col_peek = atoi(argv[4]);

  if (rank==ID_REF) {
    printf("matrix size = %zu\n", dimension);
    printf("number of iterations = %zu\n", iterations);
    printf("element for checking = Mat[%zu,%zu]\n",row_peek, col_peek);
  }
  
  if((row_peek > dimension) || (col_peek > dimension)){
    if (rank==ID:REF){
      fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
      fprintf(stderr, "Arguments n and m must be smaller than %zu\n", dimension);
    }
    return 1;
  }

  // Parallelization variables
  size = size>dimension ? size : dimension;
  int n_row = rank<=((dimension+2)%size) ? (dimension+2)/size : (dimension+2)/size + 1;
  int n_halo = (rank==0 || rank==size-1) ? 1 : 2;

  // Create local matrices
  matrix_dimension = sizeof(double) * ( n_row + n_halo ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  //memset( matrix, 0, byte_dimension );
  //memset( matrix_new, 0, byte_dimension );

  //fill initial values  
  init_mat(matrix, 0.5,rank, size, dimension, true);
  init_mat(matrix_new, 0.5,rank, size, dimension, true);

  // set up borders 
  increment = 100.0 / ( dimension + 1 );
  init_border_conditions(matrix, increment, rank, size, dimension,true)
  init_border_conditions(matrix_new, increment, rank, size, dimension,true)
  
  // start algorithm
  //t_start = seconds();
  for( it = 0; it < iterations; ++it ){
    
    evolve(matrix, matrix_new, rank, size, dimension);

    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;

  }
  //t_end = seconds();
  
  //printf( "\nelapsed time = %f seconds\n", t_end - t_start );
  //printf( "\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[ ( row_peek + 1 ) * ( dimension + 2 ) + ( col_peek + 1 ) ] );

  //save_gnuplot( matrix, dimension );
  
  free( matrix );
  free( matrix_new );

  return 0;
}




