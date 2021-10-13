#include "cuda_utils.h"
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "ref.cuh"
//A: M*K
//B: K*N
//C: M*N
#define BLOCKDIM 64
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
//naive 1.53
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
// tiling and shared memory 0.98
__global__ void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[BLOCKDIM][BLOCKDIM];
  __shared__ float mm2[BLOCKDIM][BLOCKDIM];
  float sum=0;

  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += BLOCKDIM){
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int jx = tileidx + threadIdx.x;
    int jy = tileidx + threadIdx.y;
    if(iy<M && jx<K)
      mm1[threadIdx.y][threadIdx.x] = A[iy*K+jx];
    if(jy<K && ix<N)
      mm2[threadIdx.y][threadIdx.x] = B[jy*N+ix];
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


// unroll 4*1 0.67ms
__global__ void sgemm_v5(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[BLOCKDIM][BLOCKDIM];
  __shared__ float mm2[BLOCKDIM][BLOCKDIM];
  float sum[4]={0};
  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += BLOCKDIM){
    for(int id=0;id<4;id++){
      int iy = blockIdx.y * blockDim.y*4 + threadIdx.y*4+id;
      int ix = blockIdx.x * blockDim.x + threadIdx.x;
      int jx = tileidx + threadIdx.x;
      int jy = tileidx + threadIdx.y*4+id;
      if(iy<M && jx<K)
        mm1[threadIdx.y*4+id][threadIdx.x] = A[iy*K+jx];
      if(jy<K && ix<N)
        mm2[threadIdx.y*4+id][threadIdx.x] = B[jy*N+ix];
    }
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<BLOCKDIM && k+tileidx<K;k++){
      /* 1.38ms
        sum[0] += mm1[threadIdx.y*4+0][k] * mm2[k][threadIdx.x];
        sum[1] += mm1[threadIdx.y*4+1][k] * mm2[k][threadIdx.x];
        sum[2] += mm1[threadIdx.y*4+2][k] * mm2[k][threadIdx.x];
        sum[3] += mm1[threadIdx.y*4+3][k] * mm2[k][threadIdx.x];
      */
     //0.67ms
      for(int id=0;id<4;id++){
        sum[id] += mm1[threadIdx.y*4+id][k] * mm2[k][threadIdx.x];
      }
    }
    __syncthreads();
  }
  for(int id=0;id<4;id++){
    int i = blockIdx.y * blockDim.y*4 + threadIdx.y*4+id;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<M && j<N)
      C[(i)*N+(j)] = sum[id];
  }
}
// unroll 1*4 1.35
__global__ void sgemm_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[BLOCKDIM][BLOCKDIM];
  __shared__ float mm2[BLOCKDIM][BLOCKDIM];
  float sum[4]={0};
  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += BLOCKDIM){
    for(int id=0;id<4;id++){
      int iy = blockIdx.y * blockDim.y + threadIdx.y;
      int ix = blockIdx.x * blockDim.x*4 + threadIdx.x*4+id;
      int jx = tileidx + threadIdx.x*4+id;
      int jy = tileidx + threadIdx.y;
      if(iy<M && jx<K)
        mm1[threadIdx.y][threadIdx.x*4+id] = A[iy*K+jx];
      if(jy<K && ix<N)
        mm2[threadIdx.y][threadIdx.x*4+id] = B[jy*N+ix];
    }
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<BLOCKDIM && k+tileidx<K;k++){
      /* 1.35ms
        sum[0] += mm1[threadIdx.y][k] * mm2[k][threadIdx.x*4+0];
        sum[1] += mm1[threadIdx.y][k] * mm2[k][threadIdx.x*4+1];
        sum[2] += mm1[threadIdx.y][k] * mm2[k][threadIdx.x*4+2];
        sum[3] += mm1[threadIdx.y][k] * mm2[k][threadIdx.x*4+3];
      */
     //1.35ms
      for(int id=0;id<4;id++){
        sum[id] += mm1[threadIdx.y][k] * mm2[k][threadIdx.x*4+id];
      }
      
    }
    __syncthreads();
  }
  for(int id=0;id<4;id++){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x*4 + threadIdx.x*4+id;
    if(i<M && j<N)
      C[(i)*N+(j)] = sum[id];
  }
}
// coalescing trial failed 2.53
__global__ void sgemm_v3(const float *A, const float *B, float *C, int M, int N, int K)
{
  __shared__ float mm1[BLOCKDIM][BLOCKDIM];
  __shared__ float mm2[BLOCKDIM][BLOCKDIM];
  float sum=0;

  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += BLOCKDIM){
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int jx = tileidx + threadIdx.x;
    int jy = tileidx + threadIdx.y;
    if(iy<M && jx<K)
      mm1[threadIdx.y][threadIdx.x] = A[iy*K+jx];
    if(jy<K && ix<N)
      mm2[threadIdx.x][threadIdx.y] = B[jy*N+ix];
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<BLOCKDIM && k+tileidx<K;k++)
      sum += mm1[threadIdx.y][k] * mm2[threadIdx.x][k];
    __syncthreads();
  }
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<M && j<N)
    C[(i)*N+(j)] = sum;
}
// unroll 1*4 with float4  BLOCKDIM16: 0.81
// unroll 4*4 with float4  BLOCKDIM16: 0.65  BLOCKDIM32: 0.33 BLOCKDIM64: 0.25
// M N K should be a multiple of 4
__global__ void sgemm_v6(const float *A, const float *B, float *C, int M, int N, int K)
{
  #define STEPS (32)
  __shared__ float mm1[64][STEPS];
  __shared__ float mm2[STEPS][64];
  float sum[4][4]={0};

  #pragma unroll 4
  for(int tileidx = 0;tileidx<K;tileidx += STEPS){
    for(int id=0;id<4;id++){
      int iy = blockIdx.y * blockDim.y*4 + threadIdx.y*4+id;
      int ix = blockIdx.x * blockDim.x*4 + threadIdx.x*4;
      int jx = tileidx + threadIdx.x*4;
      int jy = tileidx + threadIdx.y*4+id;
      if(iy<M && jx<K && threadIdx.x*4<STEPS)
        *(float4*)&mm1[threadIdx.y*4+id][threadIdx.x*4] = *(float4*)&A[iy*K+jx];
      if(jy<K && ix<N && threadIdx.y*4<STEPS)
        *(float4*)&mm2[threadIdx.y*4+id][threadIdx.x*4] = *(float4*)&B[jy*N+ix];
    }
    __syncthreads();
    #pragma unroll 4
    for(int k=0;k<STEPS && k+tileidx<K;k++){
      for(int id=0;id<4;id++){
        for(int dd=0;dd<4;dd++)
          sum[id][dd] += mm1[threadIdx.y*4+id][k] * mm2[k][threadIdx.x*4+dd];
      }
    }
    __syncthreads();
  }
  //#pragma unroll 4
  for(int id=0;id<4;id++){
    int i = blockIdx.y * blockDim.y*4 + threadIdx.y*4+id;
    int j = blockIdx.x * blockDim.x*4 + threadIdx.x*4;
    if(i<M && j<N)
      *(float4*)&C[(i)*N+(j)] = *(float4*)&sum[id][0];
  }
}



const int loop=500;
void gpu_sgemm(float *A, float *B, float *C, int M, int N, int K)
{
  
  dim3 block(BLOCKDIM,BLOCKDIM);
  dim3 grid((N-1)/block.x+1,(M-1)/block.y+1);
  /*
  for(int i=0;i<loop;i++){
    sgemm_v1<<<grid,block>>>(A,B,C,M,N,K);
    cudaDeviceSynchronize();
  }
  */
  double tStart,tLast;
  /*
  tStart = cpuSecond();
  for(int i=0;i<loop;i++){
    test_mysgemm_v7(M,N,K,1,A,B,0,C);
    cudaDeviceSynchronize();
  }
  tLast = cpuSecond()-tStart;
  printf("gpu ref:%.6f\n",tLast*1000.0/loop);
  */
  
  tStart = cpuSecond();
  for(int i=0;i<loop;i++){
    sgemm_v5<<<grid,dim3(BLOCKDIM,BLOCKDIM/4,1)>>>(A,B,C,M,N,K);
    cudaDeviceSynchronize();
  }
  tLast = cpuSecond()-tStart;
  printf("gpuv0:%.6f\n",tLast*1000.0/loop);
  
  tStart = cpuSecond();
  dim3 grid1((N-1)/64+1,(M-1)/64+1);
  for(int i=0;i<loop;i++){
    sgemm_v6<<<grid1,dim3(16,16,1)>>>(A,B,C,M,N,K);
    cudaDeviceSynchronize();
  }
  tLast = cpuSecond()-tStart;
  printf("gpuv0:%.6f\n",tLast*1000.0/loop);
  
}
void cublas_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  cublasHandle_t handle;
  float alpha=1, beta=0;
  CHECK_CUBLAS(cublasCreate(&handle));
  double tStart = cpuSecond();
  //(BtAt)t = AB
  for(int i=0;i<loop;i++){
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N, M, K, 
        &alpha, 
        B, N,
        A, K, 
        &beta, 
        C, N);
    cudaDeviceSynchronize();
  }
  double tLast = cpuSecond()-tStart;
  printf("cublas:%.6f\n",tLast*1000.0/loop);
  cublasDestroy(handle);
}


int main(int argc,char **argv)
{
  float *A,*B,*C,*C_ref;
  #define WIDTH 1024
  int M=WIDTH,N=WIDTH,K=WIDTH;
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
  CHECK(cudaMemcpy(C_ref,C_d,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  
  int suc = checkResult(C,C_ref,M*N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  
  if(!suc){
    printMatrix(A,M,K);
    printMatrix(B,K,N);
    printMatrix(C,M,N);
    printMatrix(C_ref,M,N);
  }
  
  return 0;
}