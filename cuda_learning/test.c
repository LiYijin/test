#include <stdlib.h>
#include <stdio.h>
void initialData(float* ip,int size)
{
  for(int i=0;i<size;i++)
    ip[i]=1;
}

void printMatrix(float * C,const int nx,const int ny)
{
  printf("Matrix<%d,%d>:\n",nx,ny);
  for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++)
      printf("%6f ",C[i*nx+j]);
    printf("\n");
  }
}

void cpu_sgemm(const float *A, const float *B, float *C, int N, int M, int K)
{
  for(int i=0;i<N;i++)
    for(int j=0;j<M;j++)
      C[i*N+j]=0;
  for(int i=0;i<N;i++){
    for(int k=0;k<K;k++){
      for(int j=0;j<M;j++){
        C[i*N+j] += A[i*N+k] * B[k*K+j];
      }
    }
  }
}

int main(int argc,char **argv)
{
  float *A,*B,*C;
  int N=2,M=3,K=4;
  if(argc==4){
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  A = (float*)malloc(sizeof(float)*N*K);
  B = (float*)malloc(sizeof(float)*K*M);
  C = (float*)malloc(sizeof(float)*N*M);
  initialData(A,N*K);
  initialData(B,K*M);
  cpu_sgemm(A,B,C,N,M,K);
  printMatrix(C,N,M);
  return 0;
}