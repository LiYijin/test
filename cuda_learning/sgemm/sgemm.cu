#include "cuda_utils.h"
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

//A: M*K
//B: K*N
//C: M*N
#define BLOCKDIM 16

void cpu_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  double tStart = cpuSecond();
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++)
      C[i*N+j]=0;
    for(int k=0;k<K;k++){
      for(int j=0;j<N;j++){
        C[i*N+j] += A[i*K+k] * B[k*N+j];
      }
    }
  }
  double tLast = cpuSecond()-tStart;
  printf("cpu:%.6f\n",tLast*1000.0);
}

__global__ void sgemm_v0(const float *A, const float *B, float *C, int M, int N, int K)
{
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<M && j<N){
    float sum = 0;
    for(int k=0;k<K;k++)
      sum += A[i*K+k] * B[k*N+j];
    C[i*N+j] = sum;
  }
}
/*
__global__ void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ mm[]
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<M && j<N){
    float sum = 0;
    for(int k=0;k<K;k++)
      sum += A[i*K+k] * B[k*N+j];
    C[i*N+j] = sum;
  }
}
*/
void gpu_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  dim3 block(BLOCKDIM,BLOCKDIM);
  dim3 grid((N-1)/block.x+1,(M-1)/block.y+1);
  double tStart,tLast;
  tStart = cpuSecond();
  sgemm_v0<<<grid,block>>>(A,B,C,M,N,K);
  cudaDeviceSynchronize();
  tLast = cpuSecond()-tStart;
  printf("gpu:%.6f\n",tLast*1000.0);

  tStart = cpuSecond();
  sgemm_v1<<<grid,block>>>(A,B,C,M,N,K);
  cudaDeviceSynchronize();
  tLast = cpuSecond()-tStart;
  printf("gpu:%.6f\n",tLast*1000.0);
}
void cublas_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  cublasHandle_t handle;
  float alpha=1, beta=0;
  CHECK_CUBLAS(cublasCreate(&handle));
  double tStart = cpuSecond();
  //(BtAt)t = AB
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      N, M, K, 
      &alpha, 
      B, N,
      A, K, 
      &beta, 
      C, N);
  cudaDeviceSynchronize();
  double tLast = cpuSecond()-tStart;
  cublasDestroy(handle);
}


int main(int argc,char **argv)
{
  float *A,*B,*C,*C_ref;
  int M=640,N=650,K=400;
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

  cpu_sgemm(A,B,C,M,N,K);

  gpu_sgemm(A_d,B_d,C_d,M,N,K);
  CHECK(cudaMemcpy(C,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));

  cublas_sgemm(A_d,B_d,C_d,M,N,K);
  CHECK(cudaMemcpy(C_ref,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  
  checkResult(C,C_ref,M*N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  //printMatrix(C,M,N);
  //printMatrix(C_ref,M,N);
  return 0;
}