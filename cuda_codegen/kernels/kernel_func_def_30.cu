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
// Node name:	Constant_438
// Description:	Constant
// Input:
// Output:
//	- name: Constant_438_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_438(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_438_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_438_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3128
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3128_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3128(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3128_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3128_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2965
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2965_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2965(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2965_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2965_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_164_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3082
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3082_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3082(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3082_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3082_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2083
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2083_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2083(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2083_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2083_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2827
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2827_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2827(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2827_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2827_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_463
// Description:	Constant
// Input:
// Output:
//	- name: Constant_463_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_463(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_463_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_463_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2233
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2233_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2233(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2233_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2233_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_1(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2305
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2305_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2305(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2305_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2305_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1024_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2329_0	type: float	shape: Shape{64, 384, 1, 1}
//	- name: Constant_2332_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1026<<<dim3(1, 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_1024_0, Constant_2329_0, Convolution_1026_0);
// Convolution_float_float_float_cuda_Convolution_1028<<<dim3(1, 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_1024_0, Constant_2332_0, Convolution_1028_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1028 : Convolution_float_float_float_cuda_Convolution_1026

// Node name:	Convolution_1026
// Description:	Convolution
// Input:
//	- name: Relu_1024_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2329_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1026_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(16, 1, 16);
    const dim3 gridDim(1, 16, 4);
    const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 3072);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15))];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15))];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15))];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[(((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3))];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 1)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 2)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 12288)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 12288)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 12288)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 48)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 49)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 50)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 24576)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 24576)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 24576)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 96)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 97)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 98)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 36864)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 36864)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 36864)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 144)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 145)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 146)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 49152)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 49152)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 192)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 193)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 194)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 61440)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 61440)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 61440)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 240)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 241)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 242)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 73728)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 73728)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 73728)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 288)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 289)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 290)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input0[(((((((int)threadIdx.z) * 768) + (((((int)threadIdx.x) * 3) / 16) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) * 3) & 15)) + 86016)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 1) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 1) & 15)) + 86016)];
          pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input0[(((((((int)threadIdx.z) * 768) + ((((((int)threadIdx.x) * 3) + 2) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((((int)threadIdx.x) * 3) + 2) & 15)) + 86016)];
          input1_shared[((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3))] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 336)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 1)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 337)];
          input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + 2)] = input1[((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.x) * 3)) + 338)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 48)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 48) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 48) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 48) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 48) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 48) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 48) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 48) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 48) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 48) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 48) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 48) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 48) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 48) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 48) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 48) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 256)] * input1_shared[((((int)threadIdx.z) * 48) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 272)] * input1_shared[((((int)threadIdx.z) * 48) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 288)] * input1_shared[((((int)threadIdx.z) * 48) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 304)] * input1_shared[((((int)threadIdx.z) * 48) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 320)] * input1_shared[((((int)threadIdx.z) * 48) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 336)] * input1_shared[((((int)threadIdx.z) * 48) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 352)] * input1_shared[((((int)threadIdx.z) * 48) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 368)] * input1_shared[((((int)threadIdx.z) * 48) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 384)] * input1_shared[((((int)threadIdx.z) * 48) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 400)] * input1_shared[((((int)threadIdx.z) * 48) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 416)] * input1_shared[((((int)threadIdx.z) * 48) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 432)] * input1_shared[((((int)threadIdx.z) * 48) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 448)] * input1_shared[((((int)threadIdx.z) * 48) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 464)] * input1_shared[((((int)threadIdx.z) * 48) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 480)] * input1_shared[((((int)threadIdx.z) * 48) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 496)] * input1_shared[((((int)threadIdx.z) * 48) + 31)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 512)] * input1_shared[((((int)threadIdx.z) * 48) + 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 528)] * input1_shared[((((int)threadIdx.z) * 48) + 33)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 544)] * input1_shared[((((int)threadIdx.z) * 48) + 34)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 560)] * input1_shared[((((int)threadIdx.z) * 48) + 35)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 576)] * input1_shared[((((int)threadIdx.z) * 48) + 36)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 592)] * input1_shared[((((int)threadIdx.z) * 48) + 37)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 608)] * input1_shared[((((int)threadIdx.z) * 48) + 38)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 624)] * input1_shared[((((int)threadIdx.z) * 48) + 39)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 640)] * input1_shared[((((int)threadIdx.z) * 48) + 40)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 656)] * input1_shared[((((int)threadIdx.z) * 48) + 41)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 672)] * input1_shared[((((int)threadIdx.z) * 48) + 42)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 688)] * input1_shared[((((int)threadIdx.z) * 48) + 43)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 704)] * input1_shared[((((int)threadIdx.z) * 48) + 44)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 720)] * input1_shared[((((int)threadIdx.z) * 48) + 45)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 736)] * input1_shared[((((int)threadIdx.z) * 48) + 46)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 752)] * input1_shared[((((int)threadIdx.z) * 48) + 47)]));
          compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[6144];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1026_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1026_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_74<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_977_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2308_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3064_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_978_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2311_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3066_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1000_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2314_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1002_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2320_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1001_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2317_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_999_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1010_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1008_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3063<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_977_0, Constant_2308_0, Constant_3064_0, Relu_998_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3065<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_978_0, Constant_2311_0, Constant_3066_0, Relu_999_0);
// Convolution_float_float_float_cuda_Convolution_1006<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1000_0, Constant_2314_0, Convolution_1006_0);
// Convolution_float_float_float_cuda_Convolution_1010<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1002_0, Constant_2320_0, Convolution_1010_0);
// Convolution_float_float_float_cuda_Convolution_1008<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1001_0, Constant_2317_0, Convolution_1008_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3065 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3063
// Convolution_float_float_float_cuda_Convolution_1010 : Convolution_float_float_float_cuda_Convolution_1006
// Convolution_float_float_float_cuda_Convolution_1008 : Convolution_float_float_float_cuda_Convolution_1006

// Node name:	Matched_Pattern_3063
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_977_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2308_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3064_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3063_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Convolution_1006
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1000_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2314_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1006_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_70(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3063_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3063_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_1006_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Convolution_float_float_float_cuda_Convolution_1006_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Convolution_float_float_float_cuda_Convolution_1006_block_kernel(input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_70_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_70<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2938_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1416_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2937_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_53<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1414_0, Constant_2938_0, Relu_1420_0, BatchNormInference_1417_0);
// Add_float_float_float_cuda_Add_2547<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1416_0, Constant_2937_0, BatchNormInference_1418_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1414_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2938_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1420_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1417_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2544<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1414_0, Constant_2938_0, BatchNormInference_1417_0);
// Relu_float_float_cuda_Relu_1420<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1417_0, Relu_1420_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_53_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_2547
// Description:	Add
// Input:
//	- name: Convolution_1416_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2937_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1418_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2547_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_53_block_kernel(input0, input1, output1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_2547_block_kernel(input2, input3, output2, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_131<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2806_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2766_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1330_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1284_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1333_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1334_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_48<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1328_0, Constant_2806_0, BatchNormInference_1269_0, Add_1333_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_49<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1330_0, Constant_2766_0, Slice_1284_0, Add_1334_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_49 : FusedKernel_float_float_float_float_cuda_Add_Add_48

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1328_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2806_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1269_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1333_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2505<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1328_0, Constant_2806_0, BatchNormInference_1331_0);
// Add_float_float_float_cuda_Add_1333<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1331_0, BatchNormInference_1269_0, Add_1333_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_48_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_118<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
