# include "jacobi_utils.h"

void save_gnuplot( double *M, size_t dimension ){

  size_t i , j;
  const double h = 0.1;
  FILE *file;

  file = fopen( "solution.dat", "w" );

  for( i = 0; i < dimension + 2; ++i )
    for( j = 0; j < dimension + 2; ++j )
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( dimension + 2 ) ) + j ] );

  fclose( file );

}

double seconds(){

  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;

}

void evolve( double * matrix, double *matrix_new, size_t dimension ){

  size_t i , j;
  
  //This will be a row dominant program.
  for( i = 1 ; i <= dimension; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) *
	( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] +
	  matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] +
	  matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] +
	  matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] );

}

void init_mat(double * matrix, double val, int n_row, int dimension, int rank){

  for( i = 1; i <= n_row; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix[( i * ( dimension + 2 ) ) + j] = val;

  if()
  
}