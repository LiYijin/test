#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>

extern __device__ __noinline__ void myprint();
__global__ void test()
{
  myprint();
}


void gpu_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  test<<<1,10>>>();
}