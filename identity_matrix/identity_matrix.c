#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define N 32

void print_matrix(double* A, int n_loc){
  int i, j;
  for(i=0;i<n_loc;i++) {
    for (j=0;j<N;j++) {
      fprintf(stdout, "%.3g ", A[j+(i*N)]);
    }
    fprintf(stdout,"\n");
  }
}

int main(int argc, char* argv[]){

  int me=0, npes=1;
  int n_loc=N;
  int count=0;

  int offset=0, rest=0;

  int i_loc=0,j_glob=0;
  double* A;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  n_loc=N/npes;
  rest=N%npes;
  offset=0;

  if(me<rest) n_loc++;
  else offset=rest;

  A = (double*) malloc(N * n_loc * sizeof(double));
  memset(A,2,N*n_loc*sizeof(double));

  for(i_loc=0;i_loc<n_loc;i_loc++) {
    j_glob = i_loc+(n_loc*me) + offset;
    A[j_glob +(i_loc*N)]=1.0;
  }

  if(me==0) {
    print_matrix(A, n_loc);
    for (count=1;count<npes;count++){
      if(count==rest) n_loc-=1; // va bene perchè il codice finisce qui. Altrimenti attenti che si modifica l'info locale su n_loc
      MPI_Recv(A, n_loc*N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      print_matrix(A, n_loc);
    }
  }
  else MPI_Send(A, n_loc*N,MPI_DOUBLE,0,me,MPI_COMM_WORLD);
  
  MPI_Finalize();
}
