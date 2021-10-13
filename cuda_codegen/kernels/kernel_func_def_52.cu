// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_2749
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2749_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2749(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2749_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2749_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3156
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3156_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3156(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3156_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3156_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2953
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2953_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2953(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2953_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2953_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2902
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2902_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2902(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2902_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2902_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_404
// Description:	Constant
// Input:
// Output:
//	- name: Constant_404_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_404(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_404_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_404_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2449
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2449_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2449(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2449_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2449_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2482
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2482_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2482(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2482_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2482_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2904
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2904_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2904(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2904_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2904_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2976
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2976_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2976(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2976_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2976_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2893
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2893_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2893(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2893_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2893_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2143
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2143_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2143(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2143_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2143_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2879
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2879_0	type: float	shape: Shape{1, 64, 32, 32}
void Constant_float_cuda_Constant_2879(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2879_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2879_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1513_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2602_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1514_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2605_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1512_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2599_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1498_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2593_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3146_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1499_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2596_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3148_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Convolution_1520_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1522_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1518_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1515_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1516_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1520<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1513_0, Constant_2602_0, Convolution_1520_0);
// Convolution_float_float_float_cuda_Convolution_1522<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1514_0, Constant_2605_0, Convolution_1522_0);
// Convolution_float_float_float_cuda_Convolution_1518<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1512_0, Constant_2599_0, Convolution_1518_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3145<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1498_0, Constant_2593_0, Constant_3146_0, Relu_1515_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3147<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1499_0, Constant_2596_0, Constant_3148_0, Relu_1516_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1522 : Convolution_float_float_float_cuda_Convolution_1520
// Convolution_float_float_float_cuda_Convolution_1518 : Convolution_float_float_float_cuda_Convolution_1520
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3147 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3145

// Node name:	Convolution_1520
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1513_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2602_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1520_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1520_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Matched_Pattern_3145
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1498_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2593_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3146_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1515_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3145_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
           float compute1[1];
          
          
          compute1[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1024)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1025)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2048)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2049)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3072)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3073)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 4096)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 4097)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 64)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 5120)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 5121)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 80)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 6144)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 6145)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 96)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 7168)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 7169)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 112)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          compute[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = max((compute1[0] + input2[((((int)blockIdx.z) * 8) + ((int)threadIdx.z))]), 0.000000e+00f);
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_144(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[1536];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1520_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1520_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_1520_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3145_block_kernel(input6, input7, input8, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3145_block_kernel(input9, input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_144_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Convolution_Convolution_Convolution_Matched_Pattern_Matched_Pattern_144<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2846_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_689_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2845_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_643_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_696_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_697_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_13<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_687_0, Constant_2846_0, BatchNormInference_624_0, Add_696_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_14<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_689_0, Constant_2845_0, Slice_643_0, Add_697_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_14 : FusedKernel_float_float_float_float_cuda_Add_Add_13

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_687_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2846_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_696_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2148<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_687_0, Constant_2846_0, BatchNormInference_693_0);
// Add_float_float_float_cuda_Add_696<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_693_0, BatchNormInference_624_0, Add_696_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_13_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_26<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1538_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2614_0	type: float	shape: Shape{128, 768, 1, 1}
//	- name: Constant_2617_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1540<<<dim3(2, 2, 8), dim3(4, 4, 16), 0, 0>>>(Relu_1538_0, Constant_2614_0, Convolution_1540_0);
// Convolution_float_float_float_cuda_Convolution_1542<<<dim3(2, 2, 8), dim3(4, 4, 16), 0, 0>>>(Relu_1538_0, Constant_2617_0, Convolution_1542_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1542 : Convolution_float_float_float_cuda_Convolution_1540

// Node name:	Convolution_1540
// Description:	Convolution
// Input:
//	- name: Relu_1538_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2614_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1540_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(4, 4, 16);
    const dim3 gridDim(2, 2, 8);
    const dim3 threadIdx(thread_id % 4, thread_id / 4 % 4, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 2, block_id / 4);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3))];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 2048)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 2048)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 32)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 33)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 4096)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 64)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 65)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 6144)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 6144)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 96)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 97)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 8192)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 128)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 129)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 10240)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 10240)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 160)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 161)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 12288)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 192)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 193)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 14336)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 14336)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 224)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 225)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 16384)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 16384)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 256)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 257)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 18432)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 18432)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 288)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 289)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 20480)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 20480)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 320)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 321)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 22528)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 22528)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 352)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 353)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 24576)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 24576)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 384)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 385)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 26624)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 26624)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 416)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 417)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 28672)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 28672)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 448)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 449)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 30720)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 30720)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 480)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 481)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 32768)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 32768)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 512)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 513)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 34816)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 34816)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 544)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 545)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 36864)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 36864)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 576)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 577)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 38912)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 38912)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 608)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 609)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 40960)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 40960)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 640)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 641)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 43008)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 43008)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 672)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 673)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 45056)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 45056)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 704)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 705)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 47104)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 47104)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 736)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 737)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[4096];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Convolution_float_float_float_cuda_Convolution_1540_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1540_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_148<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
