#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <iostream>
extern "C" __global__  void conv_test(float* input0, float* input1, float* output0);


int main()
{
    float *input0,*input1,*output;
    cudaMalloc((void**)&input0,384*16*16*4);
    cudaMalloc((void**)&input1,128*384*4);
    cudaMalloc((void**)&output,128*16*16*4);
    for(int i=0;i<1000;i++){
    conv_test<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(input0,input1,output);
    cudaDeviceSynchronize();
    }
    double tStart = cpuSecond();
    for(int i=0;i<1000;i++){
    conv_test<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(input0,input1,output);
    cudaDeviceSynchronize();
    }
    double tLast = cpuSecond()-tStart;
    printf("time:%.6fms mean:%.6fus\n",tLast*1000.0,tLast*1000);
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    cudaProfilerStop();
  return 0;
}