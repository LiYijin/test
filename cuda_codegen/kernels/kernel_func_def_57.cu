// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: Constant_12_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_397
// Description:	Constant
// Input:
// Output:
//	- name: Constant_397_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_397(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_397_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_397_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3150
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3150_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3150(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3150_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3150_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2524
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2524_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2524(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2524_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2524_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2548
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2548_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2548(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2548_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2548_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2813
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2813_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2813(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2813_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2813_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_120
// Description:	Constant
// Input:
// Output:
//	- name: Constant_120_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2050
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2050_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2050(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2050_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2050_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2107
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2107(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2107_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2107_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2275
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2275_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2275(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2275_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2275_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2182_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2762
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2762_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2762(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2762_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2762_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1745_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2731_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1746_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2734_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1748_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1750_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1748<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1745_0, Constant_2731_0, Convolution_1748_0);
// Convolution_float_float_float_cuda_Convolution_1750<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1746_0, Constant_2734_0, Convolution_1750_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1750 : Convolution_float_float_float_cuda_Convolution_1748

// Node name:	Convolution_1748
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1745_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2731_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1748_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1748_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(8, 2, 8);
    const dim3 gridDim(1, 4, 16);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 4, block_id / 4);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 1024);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1024)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1025)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2048)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2049)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3072)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3073)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 4096)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 4097)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 64)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 5120)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 5121)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 80)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 6144)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 6145)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 96)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 7168)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 7169)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 112)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_178(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[1536];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1748_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1748_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_178_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_178<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	MaxPool_898
// Description:	MaxPool
// Input:
//	- name: Slice_895_0	type: float	shape: Shape{1, 64, 32, 32}
// Output:
//	- name: MaxPool_898_0	type: float	shape: Shape{1, 64, 16, 16}
void MaxPool_float_float_cuda_lib_MaxPool_898(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 32, 32));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 64, 16, 16));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2933_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2934_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1391_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1393_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2935_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_266_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: Add_1401_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1406_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1389_0, Constant_2933_0, Convolution_1391_0, Constant_2934_0, Add_1401_0);
// Add_float_float_float_cuda_Add_2535<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1393_0, Constant_2935_0, BatchNormInference_1398_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1405_0, Constant_266_0, DepthwiseConv2dNative_1406_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1389_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2933_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1391_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2934_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1401_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2529<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1389_0, Constant_2933_0, BatchNormInference_1396_0);
// Add_float_float_float_cuda_Add_2532<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1391_0, Constant_2934_0, BatchNormInference_1397_0);
// Add_float_float_float_cuda_Add_1401<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1396_0, BatchNormInference_1397_0, Add_1401_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(input2[tid], input3[tid]);
    float temp2 = add(temp0, temp1);
    output0[tid] = temp2;

}
// Node name:	Add_2535
// Description:	Add
// Input:
//	- name: Convolution_1393_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2935_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2535_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	DepthwiseConv2dNative_1406
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1405_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_266_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1406_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 8;
        const int in_width = 8;
        const int in_depth = 128;
        const int filter_height = 3;
        const int filter_width = 3;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 1;
        const int pad_width = 1;
        const int out_height = 8;
        const int out_width = 8;
        const int out_depth = 128;
        const int num_outputs = 8192;

        for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < num_outputs;
             thread_id += blockDim.x * gridDim.x)
        {
            // Compute the indexes of this thread in the output.
            //
            // We want coalesced reads so we make sure that each warp reads
            // a contiguous chunk of memory.
            //
            // THIS IS PROBABLY WRONG, we are not doing coalesced reads
            // into the input, because of the depth multiplier division...
            const int out_col = thread_id % out_width;
            const int out_row = (thread_id / out_width) % out_height;
            const int out_channel = (thread_id / out_width / out_height) % out_depth;
            const int batch = thread_id / out_width / out_height / out_depth;

            // Compute the input depth and the index of depth multiplier
            // based off the output depth index that this thread is
            // computing n.
            const int in_channel = out_channel / depth_multiplier;
            const int multiplier = out_channel % depth_multiplier;

            // Data is stored in the following format (let's assume we
            // flatten the height and width into one contiguous dimension
            // called "P".
            //
            // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
            // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
            //
            // Each row contains in_depth * in_height * in_width values
            // for each sample in the batch.
            //
            // We can further flatten it into:
            //
            // B1C1P1 B1C1P2 .....
            // B1C2P1 B1C2P2 ....
            // B2C1P1 B2C1P2 .....
            // B2C2P1 B2C2P2 ....
            //
            // where each row is a contiguous array of all of the spatial
            // pixels for a given batch and input depth.  The following
            // loop #pragma unrolls across the filter dimensions for a given thread,
            // indexing into the filter value and the corresponding input
            // patch.
            //
            // We can compute the index into the patch once right here.
            const int input_offset_temp = (batch * in_depth + in_channel) * (in_height * in_width);

            // Finally, we can iterate over the spatial dimensions and perform the
            // convolution, writing into the output at the end.
            //
            // We perform an additional optimization, where we can determine
            // whether the patch fits within the image indices statically, and
            // avoid boundary checking within the loop.
            const int input_row_start = out_row * stride - pad_height;
            const int input_col_start = out_col * stride - pad_width;
            const int input_row_end = input_row_start + filter_height;
            const int input_col_end = input_col_start + filter_width;

            S sum = static_cast<S>(0);
            if (input_row_start >= 0 && input_col_start >= 0 && input_row_end < in_height &&
                input_col_end < in_width)
            {
                // Loop that doesn't need to check for boundary conditions.
                #pragma unroll
                for (int filter_row = 0; filter_row < filter_height; ++filter_row)
                {
                    const int in_row = input_row_start + filter_row;
                    const int filter_offset_temp = filter_width * filter_row;
                    #pragma unroll
                    for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                    {
                        const int in_col = input_col_start + filter_col;

                        const int input_offset = (input_offset_temp) + (in_row * in_width) + in_col;
                        const int filter_offset =
                            multiplier +
                            depth_multiplier *
                                (in_channel + in_depth * (filter_col + filter_offset_temp));
                        sum += static_cast<S>(__ldg(input + input_offset)) *
                               static_cast<S>(__ldg(filter + filter_offset));
                    }
                }
            }
            else
            {
                // Loop that needs to check for boundary conditions.
                #pragma unroll 
                for (int filter_row = 0; filter_row < filter_height; ++filter_row)
                {
                    const int in_row = input_row_start + filter_row;
                    const int filter_offset_temp = filter_width * filter_row;
                    #pragma unroll 
                    for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                    {
                        const int in_col = input_col_start + filter_col;
                        // TODO(vrv): the in_row check can be done outside of this loop;
                        // benchmark both methods to determine the better decision.
                        if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width)
                        {
                            const int in_col = input_col_start + filter_col;

                            // input_offset_temp indexes into the start of memory
                            // where the spatial data starts.
                            const int input_offset = (input_offset_temp) + (in_row * in_width) + in_col;

                            const int filter_offset =
                                multiplier +
                                depth_multiplier *
                                    (in_channel + in_depth * (filter_col + filter_offset_temp));
                            sum += static_cast<S>(__ldg(input + input_offset)) *
                                   static_cast<S>(__ldg(filter + filter_offset));
                        }
                    }
                }
            }

            output[thread_id] = static_cast<S>(sum);
        }
        
}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_50_block_kernel(input0, input1, input3, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_2535_block_kernel(input4, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1406_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_129<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, output0, output1, output2);
}
