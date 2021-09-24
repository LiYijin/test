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

__global__ void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[BLOCKDIM][BLOCKDIM];
  __shared__ float mm2[BLOCKDIM][BLOCKDIM];
  float sum=0;

  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += BLOCKDIM){
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int j = tileidx + threadIdx.x;
    if(iy<M && j<K)
      mm1[threadIdx.y][threadIdx.x] = A[iy*K+j];
    if(j<K && ix<N)
      mm2[threadIdx.y][threadIdx.x] = B[j*N+ix];
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<BLOCKDIM && k+tileidx<K;k++)
      sum += mm1[threadIdx.y][k] * mm2[k][threadIdx.x];
    __syncthreads();
  }
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<M && j<N)
    C[(i)*N+(j)] = sum;
}

#define UNROLLSIZE 4
#define TILESIZE (BLOCKDIM*UNROLLSIZE)
__global__ void sgemm_v2(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[TILESIZE][TILESIZE];
  __shared__ float mm2[TILESIZE][TILESIZE];

  float sum[UNROLLSIZE][UNROLLSIZE]={{0}};
  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += TILESIZE){
    #pragma unroll 4
    for(int i2=0;i2<UNROLLSIZE;i2++){
      #pragma unroll 4
      for(int i1=0;i1<UNROLLSIZE;i1++){
        int iy = (blockIdx.y * blockDim.y + threadIdx.y)*UNROLLSIZE+i1;
        int ix = (blockIdx.x * blockDim.x + threadIdx.x)*UNROLLSIZE+i1;
        int j = tileidx + threadIdx.x*UNROLLSIZE+i2;
        if(iy<M && j<K)
          mm1[threadIdx.y*UNROLLSIZE+i1][threadIdx.x*UNROLLSIZE+i2] = A[iy*K+j];
        if(j<K && ix<N)
          mm2[threadIdx.y*UNROLLSIZE+i1][threadIdx.x*UNROLLSIZE+i2] = B[j*N+ix];
      }
    }
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<TILESIZE && k+tileidx<K;k++)
      #pragma unroll 4
      for(int i2=0;i2<UNROLLSIZE;i2++)
        #pragma unroll 4
        for(int i1=0;i1<UNROLLSIZE;i1++)
          sum[i2][i1] += mm1[threadIdx.y*UNROLLSIZE+i1][k] * mm2[k][threadIdx.x*UNROLLSIZE+i2];
    __syncthreads();
  }
  #pragma unroll 4
  for(int i2=0;i2<UNROLLSIZE;i2++){
    #pragma unroll 4
    for(int i1=0;i1<UNROLLSIZE;i1++){
      int i = (blockIdx.y*blockDim.y+threadIdx.y)*UNROLLSIZE+i2;
      int j = (blockIdx.x*blockDim.x+threadIdx.x)*UNROLLSIZE+i1;
      if(i<M && j<N)
        C[(i)*N+(j)] = sum[i2][i1];
    }
  }
}

void gpu_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  dim3 block(BLOCKDIM,BLOCKDIM);
  dim3 grid((N-1)/block.x+1,(M-1)/block.y+1);
  double tStart,tLast;
  tStart = cpuSecond();
  sgemm_v0<<<grid,block>>>(A,B,C,M,N,K);
  cudaDeviceSynchronize();
  tLast = cpuSecond()-tStart;
  printf("gpuv0:%.6f\n",tLast*1000.0);

  tStart = cpuSecond();
  sgemm_v2<<<grid,block>>>(A,B,C,M,N,K);
  cudaDeviceSynchronize();
  tLast = cpuSecond()-tStart;
  printf("gpuv1:%.6f\n",tLast*1000.0);
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
  printf("cublas:%.6f\n",tLast*1000.0);
  cublasDestroy(handle);
}


int main(int argc,char **argv)
{
  float *A,*B,*C,*C_ref;
  int M=1000,N=1024,K=1700;
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

  cpu_sgemm(A,B,C_ref,M,N,K);

  gpu_sgemm(A_d,B_d,C_d,M,N,K);
  CHECK(cudaMemcpy(C,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));

  cublas_sgemm(A_d,B_d,C_d,M,N,K);
  //CHECK(cudaMemcpy(C_ref,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  
  checkResult(C,C_ref,M*N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  //printMatrix(C,M,N);
  //printMatrix(C_ref,M,N);
  return 0;
}