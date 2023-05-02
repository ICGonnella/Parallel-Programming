#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N 10
#define TH_PER_B 8

void print_matrix(double* A, int n_loc, int N_){
  int i, j;
  for(i=0;i<n_loc;i++) {
    for (j=0;j<N_;j++) {
      fprintf(stdout, "%.3g ", A[j+(i*N_)]);
    }
    fprintf(stdout,"\n");
  }
  printf("\n");
}

__global__ void transpose(double *div_A) {

    double tmp;
    int idx = threadIdx.x+blockIdx.x*blockDim.x;

    int i = idx/N;
    int j = idx%N;

    if(i<j) {
        tmp = div_A[i*N+j];
        div_A[i*N+j] = div_A[j*N+i];
        div_A[j*N+i] = tmp;
    }
}

int main(int argc, char* argv[]){

    /* for loop variable */
    int k;

    /* allocate the A matrix on both host and device */
    double *A, *dev_A;
    A = (double*)malloc(N*N*sizeof(double));
    cudaMalloc((void**)&dev_A, N*N*sizeof(double));

    /* read the matrix from a file and print it*/
    FILE *file;
    file = fopen("matrixB.csv","r");
    for (k=0;k<N*N;k++) fscanf(file,"%lf", &A[k]);
    print_matrix(A,N,N);
    
    /* copy the A matrix on the GPU device */
    cudaMemcpy(dev_A,A,N*N*sizeof(double),cudaMemcpyHostToDevice);

    /* transpose the matrix and copy back on the CPU */
    transpose<<<N*N/TH_PER_B,TH_PER_B>>> (dev_A);
    cudaMemcpy(A,dev_A,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    print_matrix(A,N,N);

    cudaFree(dev_A);
    free(A);
}