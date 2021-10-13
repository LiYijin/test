// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
}
// Node name:	Constant_2988
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2988_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2988(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2988_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2988_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2945
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2945_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2945(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2945_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2945_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_354
// Description:	Constant
// Input:
// Output:
//	- name: Constant_354_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_354(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_354_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_354_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_342
// Description:	Constant
// Input:
// Output:
//	- name: Constant_342_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_342(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_342_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_342_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_335
// Description:	Constant
// Input:
// Output:
//	- name: Constant_335_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_335(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_335_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_335_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_216
// Description:	Constant
// Input:
// Output:
//	- name: Constant_216_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_216(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_216_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_216_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_126_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2338
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2338_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2338(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2338_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2338_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_96
// Description:	Constant
// Input:
// Output:
//	- name: Constant_96_0	type: float	shape: Shape{7, 7, 64, 1}
void Constant_float_cuda_Constant_96(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_96_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_96_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12544];
    bin_file.read(tmp_mem, 12544);
    cudaMemcpyAsync(output0, tmp_mem, 12544, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2719
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2719_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2719(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2719_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2719_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2680_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1650_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2683_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1657_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1655<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1649_0, Constant_2680_0, Convolution_1655_0);
// Convolution_float_float_float_cuda_Convolution_1657<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1650_0, Constant_2683_0, Convolution_1657_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1657 : Convolution_float_float_float_cuda_Convolution_1655

// Node name:	Convolution_1655
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1649_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2680_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1655_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[1536];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1655_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1655_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_164<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_627_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2741_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2834_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_629_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_633_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_634_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_9<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_627_0, Constant_2741_0, Slice_582_0, Add_633_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_10<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_629_0, Constant_2834_0, BatchNormInference_561_0, Add_634_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_10 : FusedKernel_float_float_float_float_cuda_Add_Add_9

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_627_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2741_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_582_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_633_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2112<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_627_0, Constant_2741_0, BatchNormInference_631_0);
// Add_float_float_float_cuda_Add_633<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_631_0, Slice_582_0, Add_633_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(temp0, input2[tid]);
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_9_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_17<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_1346_0	type: float	shape: Shape{1, 384, 8, 8}
//	- name: Constant_1935_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: Convolution_1349_0	type: float	shape: Shape{1, 64, 8, 8}
//	- name: Relu_1350_0	type: float	shape: Shape{1, 128, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1349<<<dim3(1, 8, 8), dim3(8, 1, 8), 0, 0>>>(AvgPool_1346_0, Constant_1935_0, Convolution_1349_0);
// Relu_float_float_cuda_Relu_1350<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_1347_0, Relu_1350_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Convolution_1349
// Description:	Convolution
// Input:
//	- name: AvgPool_1346_0	type: float	shape: Shape{1, 384, 8, 8}
//	- name: Constant_1935_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1349_0	type: float	shape: Shape{1, 64, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1349_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 64){
        return;
    }
    const dim3 blockDim(8, 1, 8);
    const dim3 gridDim(1, 8, 8);
    const dim3 threadIdx(thread_id % 8, 0, thread_id / 8);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 8, block_id / 8);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 512);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          #pragma unroll
          for (int rc_outer = 0; rc_outer < 24; ++rc_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[(((((rc_outer * 1024) + (((int)threadIdx.z) * 128)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3) * 64)) + (((int)blockIdx.y) * 8)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7))];
            }
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
              input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + (rc_outer * 16)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
            }
            __syncthreads();
            #pragma unroll
            for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              compute_local[0] = (compute_local[0] + (pad_temp_shared[((rc_inner * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
            }
          }
          compute[((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
// Node name:	Relu_1350
// Description:	Relu
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: Relu_1350_0	type: float	shape: Shape{1, 128, 16, 16}
__device__ __noinline__ void Relu_float_float_cuda_Relu_1350_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Relu_121(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[1024];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1349_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Relu_float_float_cuda_Relu_1350_block_kernel(input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Relu_121_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Relu_121<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2872_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2967_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1668_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1669_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1670_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1672_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2688<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1666_0, Constant_2872_0, BatchNormInference_1669_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_69<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1668_0, Constant_2967_0, Relu_1672_0, BatchNormInference_1670_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2688
// Description:	Add
// Input:
//	- name: Convolution_1666_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2872_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1669_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2688_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1668_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2967_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1672_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1670_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2691<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1668_0, Constant_2967_0, BatchNormInference_1670_0);
// Relu_float_float_cuda_Relu_1672<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1670_0, Relu_1672_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_69_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output1[tid] = temp0;
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        Add_float_float_float_cuda_Add_2688_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_69_block_kernel(input3, input2, output2, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_167<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_573_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2080_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Constant_2083_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_575_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_577<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_573_0, Constant_2080_0, Convolution_577_0);
// Convolution_float_float_float_cuda_Convolution_575<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_573_0, Constant_2083_0, Convolution_575_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_575 : Convolution_float_float_float_cuda_Convolution_577

// Node name:	Convolution_577
// Description:	Convolution
// Input:
//	- name: Relu_573_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2080_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_577_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_577_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(32, 1, 8);
    const dim3 gridDim(1, 32, 2);
    const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 6144);
    {
        float* compute = output0;{
           float compute_local[2];
          
          
          compute_local[0] = 0.000000e+00f;
          compute_local[1] = 0.000000e+00f;
          pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] = input0[((((((int)threadIdx.z) * 6144) + (((((int)threadIdx.x) * 6) / 32) * 1024)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) * 6) & 31))];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 1)] = input0[((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 1) & 31))];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 2)] = input0[((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 2) & 31))];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 3)] = input0[((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 3) & 31))];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 4)] = input0[((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 4) & 31))];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 5)] = input0[((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 5) & 31))];
          input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((int)threadIdx.x) >> 4) * 192)) + ((((int)threadIdx.x) & 15) * 3))];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 1) % 48))];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 2) % 48))];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 96)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[((((int)threadIdx.z) * 96) + 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 49)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 50)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 51)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 52)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 53)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 54)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 55)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 56)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 57)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 58)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 59)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 60)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 61)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 62)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 63)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 64)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 65)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 66)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 67)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 68)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 69)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 70)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 71)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 72)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 73)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 74)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 75)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 76)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 77)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 78)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 79)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 80)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 33)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 81)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 34)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 82)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 35)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 83)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 36)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 84)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 37)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 85)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 38)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 86)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 39)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 87)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 40)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 88)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 41)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 89)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 42)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 90)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 43)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 91)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 44)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 92)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 45)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 93)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 46)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 94)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 47)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 95)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] = input0[(((((((int)threadIdx.z) * 6144) + (((((int)threadIdx.x) * 6) / 32) * 1024)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) * 6) & 31)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 1)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 1) & 31)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 2)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 2) & 31)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 3)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 3) & 31)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 4)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 4) & 31)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 5)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 5) & 31)) + 49152)];
          input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((int)threadIdx.x) >> 4) * 192)) + ((((int)threadIdx.x) & 15) * 3)) + 48)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 1)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 1) % 48)) + 48)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 2)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 2) % 48)) + 48)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 96)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[((((int)threadIdx.z) * 96) + 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 49)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 50)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 51)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 52)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 53)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 54)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 55)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 56)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 57)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 58)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 59)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 60)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 61)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 62)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 63)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 64)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 65)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 66)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 67)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 68)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 69)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 70)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 71)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 72)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 73)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 74)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 75)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 76)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 77)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 78)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 79)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 80)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 33)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 81)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 34)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 82)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 35)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 83)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 36)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 84)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 37)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 85)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 38)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 86)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 39)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 87)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 40)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 88)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 41)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 89)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 42)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 90)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 43)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 91)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 44)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 92)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 45)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 93)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 46)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 94)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 47)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 95)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] = input0[(((((((int)threadIdx.z) * 6144) + (((((int)threadIdx.x) * 6) / 32) * 1024)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) * 6) & 31)) + 98304)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 1)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 1) & 31)) + 98304)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 2)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 2) & 31)) + 98304)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 3)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 3) & 31)) + 98304)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 4)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 4) & 31)) + 98304)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 5)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 5) & 31)) + 98304)];
          input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((int)threadIdx.x) >> 4) * 192)) + ((((int)threadIdx.x) & 15) * 3)) + 96)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 1)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 1) % 48)) + 96)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 2)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 2) % 48)) + 96)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 96)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[((((int)threadIdx.z) * 96) + 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 49)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 50)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 51)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 52)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 53)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 54)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 55)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 56)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 57)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 58)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 59)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 60)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 61)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 62)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 63)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 64)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 65)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 66)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 67)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 68)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 69)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 70)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 71)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 72)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 73)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 74)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 75)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 76)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 77)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 78)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 79)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 80)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 33)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 81)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 34)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 82)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 35)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 83)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 36)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 84)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 37)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 85)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 38)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 86)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 39)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 87)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 40)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 88)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 41)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 89)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 42)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 90)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 43)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 91)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 44)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 92)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 45)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 93)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 46)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 94)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 47)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 95)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6))] = input0[(((((((int)threadIdx.z) * 6144) + (((((int)threadIdx.x) * 6) / 32) * 1024)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) * 6) & 31)) + 147456)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 1)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 1) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 1) & 31)) + 147456)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 2)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 2) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 2) & 31)) + 147456)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 3)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 3) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 3) & 31)) + 147456)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 4)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 4) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 4) & 31)) + 147456)];
          pad_temp_shared[(((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 6)) + 5)] = input0[(((((((int)threadIdx.z) * 6144) + ((((((int)threadIdx.x) * 6) + 5) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 6) + 5) & 31)) + 147456)];
          input1_shared[((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3))] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((int)threadIdx.x) >> 4) * 192)) + ((((int)threadIdx.x) & 15) * 3)) + 144)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 1)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 1) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 1) % 48)) + 144)];
          input1_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 3)) + 2)] = input1[(((((((int)blockIdx.z) * 3072) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + 2) / 48) * 192)) + (((((int)threadIdx.x) * 3) + 2) % 48)) + 144)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 96)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[((((int)threadIdx.z) * 96) + 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 96) + 49)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 96) + 50)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 96) + 51)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 96) + 52)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 96) + 53)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 96) + 54)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 96) + 55)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 96) + 56)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 96) + 57)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 96) + 58)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 96) + 59)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 96) + 60)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 96) + 61)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 96) + 62)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 96) + 63)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 96) + 64)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 96) + 65)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 96) + 66)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 96) + 67)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 96) + 68)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 96) + 69)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 96) + 70)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 96) + 71)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 768)] * input1_shared[((((int)threadIdx.z) * 96) + 72)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 800)] * input1_shared[((((int)threadIdx.z) * 96) + 73)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 832)] * input1_shared[((((int)threadIdx.z) * 96) + 74)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 864)] * input1_shared[((((int)threadIdx.z) * 96) + 75)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 896)] * input1_shared[((((int)threadIdx.z) * 96) + 76)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 928)] * input1_shared[((((int)threadIdx.z) * 96) + 77)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 960)] * input1_shared[((((int)threadIdx.z) * 96) + 78)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 992)] * input1_shared[((((int)threadIdx.z) * 96) + 79)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1024)] * input1_shared[((((int)threadIdx.z) * 96) + 80)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 33)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1056)] * input1_shared[((((int)threadIdx.z) * 96) + 81)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 34)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1088)] * input1_shared[((((int)threadIdx.z) * 96) + 82)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 35)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1120)] * input1_shared[((((int)threadIdx.z) * 96) + 83)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 36)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1152)] * input1_shared[((((int)threadIdx.z) * 96) + 84)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 37)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1184)] * input1_shared[((((int)threadIdx.z) * 96) + 85)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 38)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1216)] * input1_shared[((((int)threadIdx.z) * 96) + 86)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 39)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1248)] * input1_shared[((((int)threadIdx.z) * 96) + 87)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 40)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1280)] * input1_shared[((((int)threadIdx.z) * 96) + 88)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 41)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1312)] * input1_shared[((((int)threadIdx.z) * 96) + 89)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 42)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1344)] * input1_shared[((((int)threadIdx.z) * 96) + 90)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 43)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1376)] * input1_shared[((((int)threadIdx.z) * 96) + 91)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 44)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1408)] * input1_shared[((((int)threadIdx.z) * 96) + 92)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 45)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1440)] * input1_shared[((((int)threadIdx.z) * 96) + 93)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 46)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1472)] * input1_shared[((((int)threadIdx.z) * 96) + 94)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 47)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 1504)] * input1_shared[((((int)threadIdx.z) * 96) + 95)]));
          compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = compute_local[0];
          compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 1024)] = compute_local[1];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[9216];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_577_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_577_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_9<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
