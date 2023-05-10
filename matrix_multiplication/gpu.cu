#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

void gpu_mm(double* A_dev, double* B_dev, double* C_dev,double* C_star,int* n_loc_vect,int n_loc,int i,int* n_col_sum,double alpha,double beta, int N, int n_prc, cublasHandle_t handle) {
     cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_loc_vect[i], n_loc, N, &alpha, B_dev, n_loc_vect[i],A_dev, N, &beta, &C_dev[n_col_sum[i]], N);
     if (i==n_prc-1) cudaMemcpy(C_star,C_dev,n_loc*N*sizeof(double),cudaMemcpyDeviceToHost);
}