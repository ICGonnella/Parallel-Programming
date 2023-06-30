#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_GPU
#include "gpu.cu"
#endif

void save_times(int rank, int size, int dim, double t_tot, double t_comp, MPI_Comm Comm);

void print_matrix(double* A, int n_loc, int N_);

void create_matrix(char* name, int val, int N_);

#define ID_REF 0
#define ROOT 0

int main(int argc, char* argv[]){

  /* process related variables */
  int id, n_prc; 

  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_prc);

  if(argc != 4) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim create val\n");
    return 1;
  }
  
  /* time variables */
  float tic, toc, tic1, toc1, tic2, toc2, tic3, toc3, tic4, toc4, t_comm=0, t_comp=0, single_time=0;
  int N = atoi(argv[1]);
  int create = atoi(argv[2]);
  int val = atoi(argv[3]);
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
  _Bool changed=false;
  MPI_Comm Comm = MPI_COMM_WORLD;
  if (n_prc>N){
    changed=true;
    MPI_Comm_split(MPI_COMM_WORLD, (id < N), id, &Comm);
  } 
  MPI_Comm_rank(Comm, &id);
  MPI_Comm_size(Comm, &n_prc);

  if (n_prc==N || changed==false){
    /* number of processes having one more row allocated */
    int rest = N%n_prc;
    /* number of rows of the matrices associated to a certain process */
    int n_loc= (id<rest) ? N/n_prc+1 : N/n_prc;

    /* array storing the number of rows assigned to each process */
    int *n_loc_vect;
    n_loc_vect = (int*) malloc(n_prc*sizeof(int));
    MPI_Allgather(&n_loc,1,MPI_INT,n_loc_vect,1,MPI_INT,Comm);
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

    /* if it has to be created, create the matrices files */
    if (id==ROOT && create==1){
      create_matrix("matrixA.csv",val,N);
      create_matrix("matrixB.csv",val,N);
      printf("MATRICES HAVE BEEN CREATED\n");
    }
    MPI_Barrier(Comm);

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
    cudaEvent_t start, stop, start1, stop1, start2, stop2, start3, stop3, start4, stop4, start_comp, stop_comp;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    #endif

    /* COMPUTATIONAL LOOP */
    for(i=0;i<n_prc;i++) {
      /* create the portion of local B matrix to be gathered */
      //--------------------------update t_comm --------------------------
      MPI_Barrier(Comm);
      #ifdef USE_GPU
      cudaEventRecord(start, 0);
      #else
      tic = MPI_Wtime();
      #endif
      for(j=0;j<n_loc*n_loc_vect[i];j++) B_tmp[j]=B[n_col_sum[i]+(j/n_loc_vect[i])*N+(j%n_loc_vect[i])];
      /* Allgatherv */
      if(i<rest || n_prc==1) MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv_rest,displ_allgatherv_rest,MPI_DOUBLE,Comm);
      else MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv,displ_allgatherv,MPI_DOUBLE,Comm);
      #ifdef USE_GPU
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&single_time, start, stop);
      t_comm+=single_time/1000;
      #else
      toc = MPI_Wtime();
      t_comm +=toc-tic;
      #endif
      //if(id==0) printf("t_comm=%f\n",t_comm);
      // COMPUTATIONAL CORE
#ifdef USE_CBLAS
      //--------------------------update t_comp CBLAS--------------------------
      tic1 = MPI_Wtime();
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n_loc,n_loc_vect[i],N,1.0,A,N,B_star,n_loc_vect[i],0.0,&C_star[n_col_sum[i]],N);
      toc1=MPI_Wtime();
      t_comp+=toc1-tic1;
#elif defined(USE_GPU)
      //--------------------------update t_comm GPU_CASE--------------------------
      cudaEventRecord(start1, 0);
      cudaMemcpy(B_dev,B_star,n_loc_vect[i]*N*sizeof(double),cudaMemcpyHostToDevice);
      cudaEventRecord(stop1, 0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&single_time, start1, stop1);
      t_comm+=single_time/1000;
      //--------------------------update t_comp CUBLAS--------------------------
      cudaEventRecord(start2, 0);
      gpu_mm(A_dev,B_dev,C_dev,C_star,n_loc_vect,n_loc,i,n_col_sum,alpha,beta,N,n_prc,handle);
      cudaEventRecord(stop2, 0);
      cudaEventSynchronize(stop2);
      cudaEventElapsedTime(&single_time, start2, stop2);
      t_comp+=single_time/1000;
      //if(id==0) printf("t_commGPU1=%f\n",t_comm);
      //--------------------------update t_comm GPU_CASE--------------------------
      if (i==n_prc-1){
     	cudaEventRecord(start3, 0);
     	cudaMemcpy(C_star,C_dev,n_loc*N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop3, 0);
	cudaEventSynchronize(stop3);
        cudaEventElapsedTime(&single_time, start3, stop3);
	t_comm+=single_time/1000;
	//if(id==0) printf("t_commGPU2=%f\n",t_comm);
      }
#else
      //--------------------------update t_comp NAIVE--------------------------
      tic2 = MPI_Wtime();
      for(j=0;j<n_loc;j++)   // row of A
	for(k=0;k<n_loc_vect[i];k++)  // column of B
	  for(w=0;w<N;w++) {
	    idx = j*n_loc_vect[i]+k;
	    C_star[n_col_sum[i]+(idx/n_loc_vect[i])*N+(idx%n_loc_vect[i])] += A[j*N+w]*B_star[w*n_loc_vect[i]+k];
	  }
      toc2=MPI_Wtime();
      t_comp += toc2-tic2;
#endif
    }
    
    /* Gatherv of C_star matrices */
    //--------------------------update t_comm--------------------------
    /*#ifdef USE_GPU
    cudaEventRecord(start4, 0);
    #else
    tic3 = MPI_Wtime();
    #endif*/
    MPI_Gatherv(C_star,N*n_loc,MPI_DOUBLE,C,count_gatherv,displ_gatherv,MPI_DOUBLE,ROOT,Comm);
    /*#ifdef USE_GPU
    cudaEventRecord(stop4, 0);
    cudaEventSynchronize(stop4);
    cudaEventElapsedTime(&single_time, start4, stop4);
    t_comm+=single_time/1000;
    #else
    toc3 = MPI_Wtime();
    t_comm +=toc3-tic3;
    #endif*/

    /* write the result on a file and print it */
    if(id==ROOT && create==1) {
      FILE *fileC;
      remove("matrixC.csv");
      fileC = fopen("matrixC.csv","w");
      for(i=0;i<N*N;i++) fprintf(fileC,"%lf ",C[i]);
      printf("------ matrix C ------\n");
      print_matrix(C,N,N);
    }
    save_times(id, n_prc, N, t_comm, t_comp, Comm);
    //printf("communication_t = %.3g\ncomputation_t = %.3g \n",communication_t, computation_t);
  }
  
  MPI_Finalize();
}


void save_times(int rank, int size, int dim, double t_comm, double t_comp, MPI_Comm Comm){

  FILE *file_comm, *file_comp;
  char *name_comm = "time_comm.dat", *name_comp = "time_comp.dat";
  #ifdef USE_GPU
  name_comm = "time_comm_GPU.dat";
  name_comp = "time_comp_GPU.dat";
  #endif
  #ifdef USE_CBLAS
  name_comm = "time_comm_CBLAS.dat";
  name_comp = "time_comp_CBLAS.dat";
  #endif
  
  for (int i = 0; i < size; ++i) {
    if(rank==i) {
      file_comm = fopen(name_comm, "a");
      file_comp = fopen(name_comp, "a");
      fprintf(file_comm, "%u\t%u\t%u\t%f\n", rank, size, dim, t_comm );
      fprintf(file_comp, "%u\t%u\t%u\t%f\n", rank, size, dim, t_comp );
      fclose(file_comm);
      fclose(file_comp);
    }
    MPI_Barrier(Comm);
  }
}

void print_matrix(double* A, int n_loc, int N_){
  int i, j;
  for(i=0;i<n_loc;i++) {
    for (j=0;j<N_;j++) {
      fprintf(stdout, "%.3g ", A[j+(i*N_)]);
    }
    fprintf(stdout,"\n");
  }
}

void create_matrix(char* name, int val, int N_){

  FILE *file = fopen(name,"w");
  for (int i=0;i<N_*N_;i++)
    fprintf(file, "%u ", val);
  fclose(file);
  
}
