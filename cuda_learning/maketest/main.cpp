#include "../include/cuda_utils.h"
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
void gpu_sgemm(const float *A, const float *B, float *C, int M, int N, int K);
int main(int argc,char **argv)
{
  float *A,*B,*C,*C_ref;
  int M=1000,N=1002,K=1700;
  if(argc==4){
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  A = (float*)malloc(sizeof(float)*M*K);
  B = (float*)malloc(sizeof(float)*K*N);
  C = (float*)malloc(sizeof(float)*M*N);
  C_ref = (float*)malloc(sizeof(float)*M*N);
  initialData(A,M*K);
  initialData(B,K*N);

  float *A_d, *B_d, *C_d;
  CHECK(cudaMalloc((void**)&A_d,M*K*sizeof(float)));
  CHECK(cudaMalloc((void**)&B_d,K*N*sizeof(float)));
  CHECK(cudaMalloc((void**)&C_d,M*N*sizeof(float)));
  CHECK(cudaMemcpy(A_d,A,M*K*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d,B,K*N*sizeof(float),cudaMemcpyHostToDevice));

  gpu_sgemm(A_d,B_d,C_d,M,N,K);
  CHECK(cudaMemcpy(C,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));

  //cublas_sgemm(A_d,B_d,C_d,M,N,K);
  //CHECK(cudaMemcpy(C_ref,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  
  //checkResult(C,C_ref,M*N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  //printMatrix(C,M,N);
  //printMatrix(C_ref,M,N);
  return 0;
}