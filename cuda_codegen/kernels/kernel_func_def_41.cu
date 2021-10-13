// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_340
// Description:	Constant
// Input:
// Output:
//	- name: Constant_340_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_340(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_340_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_340_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2950
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2950_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2950(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2950_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2950_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3140
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3140_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3140(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3140_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3140_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2978
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2978_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2978(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2978_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2978_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2925
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2925_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2925(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2925_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2925_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2826
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2826_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2826(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2826_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2826_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2125
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2125_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2125(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2125_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2125_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_190
// Description:	Constant
// Input:
// Output:
//	- name: Constant_190_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_190(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_190_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_190_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2707
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2707_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2707(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2707_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2707_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2982
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2982_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2982(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2982_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2982_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3020
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3020_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3020(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3020_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3020_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3162
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3162_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3162(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3162_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3162_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1198_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2431_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1199_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2434_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1204_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1206_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1204<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1198_0, Constant_2431_0, Convolution_1204_0);
// Convolution_float_float_float_cuda_Convolution_1206<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1199_0, Constant_2434_0, Convolution_1206_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1206 : Convolution_float_float_float_cuda_Convolution_1204

// Node name:	Convolution_1204
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1198_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2431_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1204_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1204_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(8, 2, 16);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 1024);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
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
          compute[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_99(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1204_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1204_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_99_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_99<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2899_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1078_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2793_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1080_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1016_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1084_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1085_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_34<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1078_0, Constant_2899_0, Slice_1033_0, Add_1084_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_35<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1080_0, Constant_2793_0, BatchNormInference_1016_0, Add_1085_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_35 : FusedKernel_float_float_float_float_cuda_Add_Add_34

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1078_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2899_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1084_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2361<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1078_0, Constant_2899_0, BatchNormInference_1082_0);
// Add_float_float_float_cuda_Add_1084<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1082_0, Slice_1033_0, Add_1084_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_34_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(temp0, input2[tid]);
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_82(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_34_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_34_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_82_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_82<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Add_903_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_937_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2278_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3056_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_935_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2284_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_936_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2287_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Add_945_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_942_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_944_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_945<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Add_903_0, AvgPool_937_0, Add_945_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_946_0, Constant_2278_0, Constant_3056_0, Relu_954_0);
// Convolution_float_float_float_cuda_Convolution_940<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_934_0, Constant_2281_0, Convolution_940_0);
// Convolution_float_float_float_cuda_Convolution_942<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_935_0, Constant_2284_0, Convolution_942_0);
// Convolution_float_float_float_cuda_Convolution_944<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_936_0, Constant_2287_0, Convolution_944_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_942 : Convolution_float_float_float_cuda_Convolution_940
// Convolution_float_float_float_cuda_Convolution_944 : Convolution_float_float_float_cuda_Convolution_940

// Node name:	Add_945
// Description:	Add
// Input:
//	- name: Add_903_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_937_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_945_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_945_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Matched_Pattern_3055
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_946_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2278_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3056_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(8, 2, 16);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 1024);
    {
        float* compute = output0;{
           float compute1[1];
          
          
          compute1[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
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
          compute[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = max((compute1[0] + input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]), 0.000000e+00f);
        }


    }

}
// Node name:	Convolution_940
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_934_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_940_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(8, 2, 16);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 1024);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
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
          compute[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_945_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3055_block_kernel(input2, input3, input4, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159)
    {
        Convolution_float_float_float_cuda_Convolution_940_block_kernel(input5, input6, output2, threadIdx.x, blockIdx.x - 96 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223)
    {
        Convolution_float_float_float_cuda_Convolution_940_block_kernel(input7, input8, output3, threadIdx.x, blockIdx.x - 160 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 224 && (int)blockIdx.x <= 287)
    {
        Convolution_float_float_float_cuda_Convolution_940_block_kernel(input9, input10, output4, threadIdx.x, blockIdx.x - 224 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Convolution_Convolution_Convolution_63<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4);
}
