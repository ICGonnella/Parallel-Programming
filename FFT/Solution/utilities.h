/* 
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 */

#ifndef _FFTW_UTLITIES_
#define _FFTW_UTLITIES_
#include <complex.h>
#include <fftw3.h>
#include <sys/time.h>
#include <stdbool.h>
#define pi 3.14159265358979323846

#include <mpi.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

typedef struct {

  #ifdef FFTW3_MPI
  fftw_plan fw_plan; 
  fftw_plan bw_plan;
  #else
  fftw_plan fw_plan_1D; 
  fftw_plan fw_plan_2D;
  fftw_plan bw_plan_1D;
  fftw_plan bw_plan_2D;
  ptrdiff_t n1;
  ptrdiff_t n2;
  ptrdiff_t n3;
  ptrdiff_t local_n2;
  ptrdiff_t local_square;
  fftw_complex *fftw_data_T;
  int* counts;
  int* counts_T;
  int* displacement;
  int* displacement_T;
  MPI_Datatype* Elem_Type;
  MPI_Datatype* Elem_Type_T;
  MPI_Datatype Elem_Pack;
  #endif
  
  fftw_complex *fftw_data;
  ptrdiff_t global_size_grid;
  ptrdiff_t local_size_grid;
  ptrdiff_t local_n1;
  ptrdiff_t local_n1_offset;
  MPI_Comm mpi_comm;
  
} fftw_dist_handler;



double seconds();
int index_f ( int i1, int i2, int i3, int n1, int n2, int n3 );


void plot_data_1d( char* name, int n1, int n2, int n3, int dir, double* data );
void plot_data_2d( char* name, int n1, int n2, int n3, int dir, double* data );
void init_fftw( fftw_dist_handler* fft, int n1, int n2, int n3, MPI_Comm comm );
void close_fftw( fftw_dist_handler* fft );

void derivative( fftw_dist_handler* fft,int n1, int n2, int n3, double L1, double L2, double L3, int ipol, double* data, double* deriv );
void fft_3d( fftw_dist_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal );

#endif
