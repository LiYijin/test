#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel1.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel2.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel5.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel6.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel7.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel8.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel9.cuh"
#include "Optimizing-SGEMM-on-NVIDIA-Turing-GPUs/include/kernel10.cuh"

#define FLOAT float
#define INT int
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)
void test_mysgemm_v1(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(32,32);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v1<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v2(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(32,32);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v2<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}
void test_mysgemm_v5(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v5<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v6(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v6<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v7(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,64),CEIL_DIV(N,64));
    mysgemm_v7<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v8(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v8<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v9(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v9<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v10(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v10<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}