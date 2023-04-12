#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <math.h>

#define N 10
#define ID_REF 1
#define ROOT 0

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

  /* indices needed for the loops */
  int i,j,k,w;
  /* variable needed for the right allocation of C elements */
  int idx;
  /* variable needed for the files reading */
  double trash;

  /* number of processes having one more row allocated */
  int rest = N%n_prc;
  /* number of rows of the matrices associated to a certain process */
  int n_loc= (id<rest) ? N/n_prc+1 : N/n_prc;

  /* array storing the number of rows assigned to each process */
  int n_loc_vect[n_prc]; 
  MPI_Allgather(&n_loc,1,MPI_INT,n_loc_vect,1,MPI_INT,MPI_COMM_WORLD);
  /* number of columns already considered in the past iterations */
  int n_col_sum[n_prc];
  for(i=0;i<n_prc;i++) n_col_sum[i]= i==0 ? 0:n_col_sum[i-1]+n_loc_vect[i-1];

  /* matrices allocation */
  double A[n_loc*N], B[n_loc*N];                                                                                                                                                               
  double *C; 
  if (id==ROOT) C = (double*) malloc(N*N*sizeof(double));

  /* local copy for the process of n_loc_vect[i] columns of the local B matrix */
  double B_tmp[n_loc*n_loc_vect[0]];
  /* local copy for the process of n_loc_vect[i] columns of the whole B matrix */
  double B_star[n_loc_vect[0]*N];
  /* local copy for the process of n_loc rows of the whole C matrix */
  double C_star[n_loc*N];
  
  /* count and displacement arrays needed for MPI_Allgatherv function in case of the first columns considered (having also the rest contribute) */
  int count_allgatherv_rest[n_prc];
  int displ_allgatherv_rest[n_prc];
  for(i=0;i<n_prc;i++){
    count_allgatherv_rest[i]=n_loc_vect[0]*n_loc_vect[i];
    displ_allgatherv_rest[i]= i>0 ? displ_allgatherv_rest[i-1]+count_allgatherv_rest[i-1] : 0;
  }

  /* count and displacement arrays needed for MPI_Allgatherv function in case of the last columns considered (not having the rest contribute) */
  int count_allgatherv[n_prc];
  int displ_allgatherv[n_prc];
  for(i=0;i<n_prc;i++){
    count_allgatherv[i]=n_loc_vect[i]*n_loc_vect[n_prc-1];
    displ_allgatherv[i]= i>0 ? displ_allgatherv[i-1]+count_allgatherv[i-1] : 0;
  }

  /* count and displacement arrays needed for MPI_Scatterv function */
  int count_gatherv[n_prc];
  int displ_gatherv[n_prc];
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
  if(id==ID_REF){
    printf("------ matrix A (proc %i) ------\n",ID_REF);
    print_matrix(A,n_loc,N);
    printf("------ matrix B (proc %i) ------\n",ID_REF);
    print_matrix(B,n_loc,N);
  }

  /* COMPUTATIONAL LOOP */
  for(i=0;i<n_prc;i++) {
    /* create the portion of local B matrix to be gathered */
    for(j=0;j<n_loc*n_loc_vect[i];j++) B_tmp[j]=B[n_col_sum[i]+(j/n_loc_vect[i])*N+(j%n_loc_vect[i])];

    /* Allgatherv */
    if(i<rest || n_prc==1) MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv_rest,displ_allgatherv_rest,MPI_DOUBLE,MPI_COMM_WORLD);
    else MPI_Allgatherv(B_tmp,n_loc*n_loc_vect[i],MPI_DOUBLE,B_star,count_allgatherv,displ_allgatherv,MPI_DOUBLE,MPI_COMM_WORLD);

    /* compute the C_tmp */
    for(j=0;j<n_loc;j++)   // row of A
      for(k=0;k<n_loc_vect[i];k++)  // column of B
	for(w=0;w<N;w++) {
	  idx = j*n_loc_vect[i]+k;
	  C_star[n_col_sum[i]+(idx/n_loc_vect[i])*N+(idx%n_loc_vect[i])] += A[j*N+w]*B_star[w*n_loc_vect[i]+k];
	}
  }

  /* Gatherv of C_star matrices */
  MPI_Gatherv(C_star,N*n_loc,MPI_DOUBLE,C,count_gatherv,displ_gatherv,MPI_DOUBLE,ROOT,MPI_COMM_WORLD);

  /* write the result on a file and print it */
  if(id==ROOT) {
    FILE *fileC;
    remove("matrixC.csv");
    fileC = fopen("matrixC.csv","w");
    for(i=0;i<N*N;i++) fprintf(fileC,"%lf ",C[i]);
    printf("------ matrix C ------\n");
    print_matrix(C,N,N);
  }
  
  MPI_Finalize();
}
