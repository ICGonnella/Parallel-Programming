#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

float gpu_mm(double* A_dev, double* B_dev, double* C_dev,double* C_star,int* n_loc_vect,int n_loc,int i,int* n_col_sum,double alpha,double beta, int N, int n_prc, cublasHandle_t handle) {

     cudaEvent_t start, stop;
     float time;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     
     cudaEventRecord(start, 0);
     cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_loc_vect[i], n_loc, N, &alpha, B_dev, n_loc_vect[i],A_dev, N, &beta, &C_dev[n_col_sum[i]], N);
     cudaEventRecord(stop, 0);

     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&time, start, stop);
 
     return time;
}