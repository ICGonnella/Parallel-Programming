#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_GPU
#include "gpu.cu"
#endif

#define ID_REF 0
#define ROOT 0
#define N 5

void print_matrix(double* A, int n_loc, int N_){
  int i, j;
  for(i=0;i<n_loc;i++) {
    for (j=0;j<N_;j++) {
      fprintf(stdout, "%.3g ", A[j+(i*N_)]);
    }
    fprintf(stdout,"\n");
  }
}

int main(int argc, char* argv[]){

  /* process related variables */
  int id, n_prc; 

  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_prc);

  /* time variables */
  double tic1,tic2,tic3, communication_t=0, computation_t=0;
  
  /* indices needed for the loops */
  int i,j;
  #ifndef USE_CBLAS
  #ifndef USE_GPU
  /* variable needed for the right allocation of C elements and for the loops in the baseline version*/
  int idx,k,w;
  #endif
  #endif
  /* variable needed for the files reading */
  double trash;

  /* number of processes having one more row allocated */
  int rest = N%n_prc;
  /* number of rows of the matrices associated to a certain process */
  int n_loc= (id<rest) ? N/n_prc+1 : N/n_prc;

  /* array storing the number of rows assigned to each process */
  int *n_loc_vect;
  n_loc_vect = (int*) malloc(n_prc*sizeof(int));
  MPI_Allgather(&n_loc,1,MPI_INT,n_loc_vect,1,MPI_INT,MPI_COMM_WORLD);
  /* number of columns already considered in the past iterations */
  int *n_col_sum;
  n_col_sum = (int*) malloc(n_prc*sizeof(int));
  for(i=0;i<n_prc;i++) n_col_sum[i]= i==0 ? 0:n_col_sum[i-1]+n_loc_vect[i-1];

  /* matrices allocation */                                                                                                                                                               
  double *A, *B, *C;
  A = (double*) malloc(n_loc*N*sizeof(double));
  B = (double*) malloc(n_loc*N*sizeof(double));
  if (id==ROOT) C = (double*) malloc(N*N*sizeof(double));
  

  /* local copy for the process of n_loc_vect[i] columns of the local B matrix */
  double *B_tmp;
  B_tmp = (double*) malloc(n_loc*n_loc_vect[0]*sizeof(double));
  /* local copy for the process of n_loc_vect[i] columns of the whole B matrix */
  double *B_star;
  B_star = (double*) malloc(n_loc_vect[0]*N*sizeof(double));
  /* local copy for the process of n_loc rows of the whole C matrix */
  double *C_star;
  C_star = (double*) malloc(n_loc*N*sizeof(double));

  #ifdef USE_GPU
  double *A_dev, *B_dev, *C_dev;
  double alpha=1, beta=0;
  cudaMalloc((void**)&A_dev, n_loc*N*sizeof(double));
  cudaMalloc((void**)&B_dev, n_loc_vect[0]*N*sizeof(double));
  cudaMalloc((void**)&C_dev, n_loc*N*sizeof(double));
  #endif
  
  /* count and displacement arrays needed for MPI_Allgatherv function in case of the first columns considered (having also the rest contribute) */
  int *count_allgatherv_rest;
  int *displ_allgatherv_rest;
  count_allgatherv_rest = (int*) malloc(n_prc*sizeof(int));
  displ_allgatherv_rest = (int*) malloc(n_prc*sizeof(int));
  for(i=0;i<n_prc;i++){
    count_allgatherv_rest[i]=n_loc_vect[0]*n_loc_vect[i];
    displ_allgatherv_rest[i]= i>0 ? displ_allgatherv_rest[i-1]+count_allgatherv_rest[i-1] : 0;
  }

  /* count and displacement arrays needed for MPI_Allgatherv function in case of the last columns considered (not having the rest contribute) */
  int *count_allgatherv;
  int *displ_allgatherv;
  count_allgatherv = (int*) malloc(n_prc*sizeof(int));
  displ_allgatherv = (int*) malloc(n_prc*sizeof(int));
  for(i=0;i<n_prc;i++){
    count_allgatherv[i]=n_loc_vect[i]*n_loc_vect[n_prc-1];
    displ_allgatherv[i]= i>0 ? displ_allgatherv[i-1]+count_allgatherv[i-1] : 0;
  }

  /* count and displacement arrays needed for MPI_Gatherv function */
  int *count_gatherv;
  int *displ_gatherv;
  count_gatherv = (int*) malloc(n_prc*sizeof(int));
  displ_gatherv = (int*) malloc(n_prc*sizeof(int));
  for(i=0;i<n_prc;i++){
    count_gatherv[i]=n_loc_vect[i]*N;
    displ_gatherv[i]= i>0 ? displ_gatherv[i-1]+count_gatherv[i-1] : 0;
  }

  /* each process reads the correct portion of the input matrices */
  FILE *fileA, *fileB;
  fileA = fopen("matrixA.csv","r");
  fileB = fopen("matrixB.csv","r");
  for(i=0;i<(n_col_sum[id]+n_loc)*N;i++) {
    if (i>=n_col_sum[id]*N) {
      fscanf(fileA,"%lf", &A[i-n_col_sum[id]*N]);
      fscanf(fileB,"%lf", &B[i-n_col_sum[id]*N]);
    }
    else {
      fscanf(fileA,"%lf", &trash);
      fscanf(fileB,"%lf", &trash);
    }
  }
  fclose(fileA);
  fclose(fileB);

  /* the process ID_REF prints its portion of the input matrices */
  /*if(id==ID_REF){
    printf("------ matrix A (proc %i) ------\n",ID_REF);
    print_matrix(A,n_loc,N);
    printf("------ matrix B (proc %i) ------\n",ID_REF);
    print_matrix(B,n_loc,N);
  }*/

  #ifdef USE_GPU
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMemcpy(A_dev,A,n_loc*N*sizeof(double),cudaMemcpyHostToDevice);
  //cudaMemcpy(C_dev,C_star,n_loc*N*sizeof(double),cudaMemcpyHostToDevice);
  #endif
  
  /* COMPUTATIONAL LOOP */
  for(i=0;i<n_prc;i++) {
    /* create the portion of local B matrix to be gathered */
    for(j=0;j<n_loc*n_loc_vect[i];j++) B_tmp[j]=B[n_col_sum[i]+(j/n_loc_vect[i])*N+(j%n_loc_vect[i])];

    tic1 = MPI_Wtime();
    /* Allgatherv */
    if(i<rest || n_prc==1) MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv_rest,displ_allgatherv_rest,MPI_DOUBLE,MPI_COMM_WORLD);
    else MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv,displ_allgatherv,MPI_DOUBLE,MPI_COMM_WORLD);
    tic2 = MPI_Wtime();
    #ifdef USE_CBLAS
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_loc,n_loc_vect[i],N,1.0,A,N,B_star,n_loc_vect[i],0.0,&C_star[n_col_sum[i]],N);
    #elif defined(USE_GPU)
    cudaMemcpy(B_dev,B_star,n_loc_vect[i]*N*sizeof(double),cudaMemcpyHostToDevice);
    gpu_mm(A_dev,B_dev,C_dev,C_star,n_loc_vect,n_loc,i,n_col_sum,alpha,beta,N,n_prc,handle);
    #else
    /* compute the C_tmp */
    for(j=0;j<n_loc;j++)   // row of A
      for(k=0;k<n_loc_vect[i];k++)  // column of B
	for(w=0;w<N;w++) {
	  idx = j*n_loc_vect[i]+k;
	  C_star[n_col_sum[i]+(idx/n_loc_vect[i])*N+(idx%n_loc_vect[i])] += A[j*N+w]*B_star[w*n_loc_vect[i]+k];
	}
    #endif
    tic3=MPI_Wtime();
    communication_t += tic2-tic1;
    computation_t += tic3-tic2;
  }

  tic1 = MPI_Wtime();
  /* Gatherv of C_star matrices */
  MPI_Gatherv(C_star,N*n_loc,MPI_DOUBLE,C,count_gatherv,displ_gatherv,MPI_DOUBLE,ROOT,MPI_COMM_WORLD);
  tic2 = MPI_Wtime();
  communication_t += tic2-tic1;
  
  /* write the result on a file and print it */
  if(id==ROOT) {
    FILE *fileC;
    remove("matrixC.csv");
    fileC = fopen("matrixC.csv","w");
    for(i=0;i<N*N;i++) fprintf(fileC,"%lf ",C[i]);
    printf("------ matrix C ------\n");
    print_matrix(C,N,N);
  }

  printf("communication_t = %.3g\ncomputation_t = %.3g \n",communication_t, computation_t);
  
  MPI_Finalize();
}
