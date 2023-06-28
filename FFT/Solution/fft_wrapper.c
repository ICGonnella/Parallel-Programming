/* Assignement:
 * Here you have to modify the includes, the array sizes and the fftw calls, to use the fftw-mpi
 *
 * Regarding the fftw calls. here is the substitution 
 * fftw_plan_dft_3d -> fftw_mpi_plan_dft_3d
 * ftw_execute_dft  > fftw_mpi_execute_dft 
 * use fftw_mpi_local_size_3d for local size of the arrays
 * 
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 *
 */ 

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "utilities.h"


double seconds(){
/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970) */
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

/* 
 *  Index linearization is computed following row-major order.
 *  For more informtion see FFTW documentation:
 *  http://www.fftw.org/doc/Row_002dmajor-Format.html#Row_002dmajor-Format
 *
 */
int index_f ( int i1, int i2, int i3, int n1, int n2, int n3){

  return n3*n2*i1 + n3*i2 + i3; 
}

void init_fftw(fftw_dist_handler *fft, int n1, int n2, int n3, MPI_Comm comm){
  
  int npes, mype;
  
  fft->mpi_comm = comm;

  #ifdef FFTW3_MPI
  
  fftw_mpi_init();
  fft->local_size_grid = fftw_mpi_local_size_3d(n1, n2, n3, fft->mpi_comm, &fft->local_n1, &fft->local_n1_offset);
  
  #else
  
  MPI_Comm_size( comm, &npes );
  MPI_Comm_rank( comm, &mype );
  
  fft->n1 = n1;
  fft->n2 = n2;
  fft->n3 = n3;
  fft->local_n1 = fft->n1/npes;
  fft->local_n1_offset = fft->local_n1*mype;
  fft->local_n2 = fft->n2/npes;
  fft->local_size_grid = fft->local_n1*fft->n2*fft->n3;
  fft->local_square = fft->local_n1 * fft->local_n2;
  
  MPI_Type_vector(fft->local_n1, fft->local_n2 * fft->n3, fft->n2 * fft->n3, MPI_C_DOUBLE_COMPLEX,
                  &(fft->Elem_Pack));
  MPI_Type_commit(&(fft->Elem_Pack));

  fft->fftw_data_T = (fftw_complex *)fftw_malloc(fft->local_size_grid * sizeof(fftw_complex));


  fft->counts = (int *)malloc(npes * sizeof(int));
  fft->counts_T = (int *)malloc(npes * sizeof(int));
  fft->displacement = (int *)malloc(npes * sizeof(int));
  fft->displacement_T = (int *)malloc(npes * sizeof(int));
  fft->Elem_Type = (MPI_Datatype *)malloc(npes * sizeof(MPI_Datatype));
  fft->Elem_Type_T = (MPI_Datatype *)malloc(npes * sizeof(MPI_Datatype));

  for (int i = 0; i < npes; i++) {
    fft->counts[i] = 1;
    fft->counts_T[i] = fft->local_square;
    fft->displacement[i] = i==0 ? 0 : fft->displacement[i] + fft->local_n2*fft->n3*sizeof(fftw_complex);
    fft->displacement_T[i] = i==0 ? 0 : fft->displacement_T[i] + fft->local_square*sizeof(fftw_complex); 
    fft->Elem_Type[i] = fft->Elem_Pack;
    fft->Elem_Type_T[i] = MPI_C_DOUBLE_COMPLEX;
  }
  
  
  int dim_2D[] = {n2, n3}, dim_1D[] = {n1};

  #endif

  if( ( ( n1 % npes ) || ( n2 % npes ) ) && !mype ){
    
    fprintf( stdout, "\nN1 dimension must be multiple of the number of processes. The program will be aborted...\n\n" );
    MPI_Abort( comm, 1 );
  }
  
  fft->global_size_grid = n1*n2*n3;
  fft->fftw_data = (fftw_complex *)fftw_malloc(fft->local_size_grid * sizeof(fftw_complex));
  
  #ifdef FFTW3_MPI
  
  fft->fw_plan = fftw_mpi_plan_dft_3d(n1, n2, n3,
				      fft->fftw_data,
				      fft->fftw_data,
				      fft->mpi_comm,
				      FFTW_FORWARD, FFTW_ESTIMATE);
  fft->bw_plan = fftw_mpi_plan_dft_3d( n1, n2, n3,
				       fft->fftw_data,
				       fft->fftw_data,
				       fft->mpi_comm,
				       FFTW_BACKWARD, FFTW_ESTIMATE);

  #else

  fft->fw_plan_1D = fftw_plan_many_dft(1, dim_1D,
				       fft->local_n2*fft->n3,
				       fft->fftw_data_T,
				       dim_1D,
				       fft->local_n2*fft->n3,
				       1,
				       fft->fftw_data_T,
				       dim_1D,
				       fft->local_n2*fft->n3,
				       1,
				       FFTW_FORWARD, FFTW_ESTIMATE);
  fft->fw_plan_2D = fftw_plan_many_dft(2, dim_2D,
				       fft->local_n1,
				       fft->fftw_data,
				       dim_2D,
				       1,
				       fft->n2*fft->n3,
				       fft->fftw_data,
				       dim_2D,
				       1,
				       fft->n2*fft->n3,
				       FFTW_FORWARD, FFTW_ESTIMATE);
  fft->bw_plan_1D = fftw_plan_many_dft(1, dim_1D,
				       fft->local_n2*fft->n3,
				       fft->fftw_data_T,
				       dim_1D,
				       fft->local_n2*fft->n3,
				       1,
				       fft->fftw_data_T,
				       dim_1D,
				       fft->local_n2*fft->n3,
				       1,
				       FFTW_BACKWARD, FFTW_ESTIMATE);
  fft->bw_plan_2D = fftw_plan_many_dft(2, dim_2D,
				       fft->local_n1,
				       fft->fftw_data,
				       dim_2D,
				       1,
				       fft->n2*fft->n3,
				       fft->fftw_data,
				       dim_2D,
				       1,
				       fft->n2*fft->n3,
				       FFTW_BACKWARD, FFTW_ESTIMATE);

  #endif

}

void close_fftw( fftw_dist_handler *fft ){
  
  #ifdef _FFTW3_MPI

  fftw_destroy_plan(fft->fw_plan);
  fftw_destroy_plan(fft->bw_plan);

  #else

  fftw_destroy_plan(fft->fw_plan_1D);
  fftw_destroy_plan(fft->fw_plan_2D);
  fftw_destroy_plan(fft->bw_plan_1D);
  fftw_destroy_plan(fft->bw_plan_2D);
  fftw_free(fft->fftw_data_T);
  free(fft->counts);
  free(fft->counts_T);
  free(fft->displacement);
  free(fft->displacement_T);
  MPI_Type_free(fft->Elem_Type);
  MPI_Type_free(fft->Elem_Type_T);
  MPI_Type_free(&(fft->Elem_Pack));

  #endif

  fftw_free(fft->fftw_data);
}

/* This subroutine uses fftw to calculate 3-dimensional discrete FFTs.
 * The data in direct space is assumed to be real-valued
 * The data in reciprocal space is complex. 
 * direct_to_reciprocal indicates in which direction the FFT is to be calculated
 * 
 * Note that for real data in direct space (like here), we have
 * F(N-j) = conj(F(j)) where F is the array in reciprocal space.
 * Here, we do not make use of this property.
 * Also, we do not use the special (time-saving) routines of FFTW which
 * allow one to save time and memory for such real-to-complex transforms.
 *
 * f: array in direct space
 * F: array in reciprocal space
 * 
 * F(k) = \sum_{l=0}^{N-1} exp(- 2 \pi I k*l/N) f(l)
 * f(l) = 1/N \sum_{k=0}^{N-1} exp(+ 2 \pi I k*l/N) F(k)
 * 
 */

void fft_3d( fftw_dist_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal ){

  double fac;
  int npes;

  /* Allocate buffers to send and receive data */

  MPI_Comm_size( fft->mpi_comm, &npes );
    
  // Now distinguish in which direction the FFT is performed
  if( direct_to_reciprocal ){

    for(int i = 0; i < fft->local_size_grid; i++)
	    fft->fftw_data[i]  = data_direct[i] + 0.0 * I;
	  
    #ifdef FFTW3_MPI

    fftw_execute_dft(fft->fw_plan, fft->fftw_data, fft->fftw_data);
    
    #else

    fftw_execute(fft->fw_plan_2D); 
    MPI_Alltoallw(fft->fftw_data, fft->counts, fft->displacement, fft->Elem_Type, fft->fftw_data_T, fft->counts_T, fft->displacement_T, fft->Elem_Type_T, MPI_COMM_WORLD);
    fftw_execute(fft->fw_plan_1D); 
    MPI_Alltoallw(fft->fftw_data_T, fft->counts_T, fft->displacement_T, fft->Elem_Type_T, fft->fftw_data, fft->counts, fft->displacement, fft->Elem_Type, MPI_COMM_WORLD);

    #endif

    memcpy(data_rec, fft->fftw_data, fft->local_size_grid * sizeof(fftw_complex));

  }
  else{
    
    memcpy(fft->fftw_data, data_rec, fft->local_size_grid * sizeof(fftw_complex));

    #ifdef _FFTW3_MPI
    fftw_execute(fft->bw_plan);
    #else
    fftw_execute(fft->bw_plan_2D); 
    MPI_Alltoallw(fft->fftw_data, fft->counts, fft->displacement, fft->Elem_Type, fft->fftw_data_T, fft->counts_T, fft->displacement_T, fft->Elem_Type_T, MPI_COMM_WORLD);
    fftw_execute(fft->bw_plan_1D); 
    MPI_Alltoallw(fft->fftw_data_T, fft->counts_T, fft->displacement_T, fft->Elem_Type_T, fft->fftw_data, fft->counts, fft->displacement, fft->Elem_Type, MPI_COMM_WORLD);
    #endif

    fac = 1.0 / fft->global_size_grid;

    for (int i = 0; i < fft->local_size_grid; i++)
      data_direct[i] = creal(fft->fftw_data[i]) * fac;
    
    
  }
  
}

