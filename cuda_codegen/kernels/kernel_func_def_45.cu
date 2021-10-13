// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_371
// Description:	Constant
// Input:
// Output:
//	- name: Constant_371_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_371(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_371_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_371_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2958
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2958(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2958_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2958_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2179
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2179_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2179(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2179_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2179_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2077
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2077_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2077(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2077_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2077_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2901
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2901_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2901(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2901_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2901_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2923
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2923_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2923(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2923_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2923_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_353
// Description:	Constant
// Input:
// Output:
//	- name: Constant_353_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_353(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_353_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_353_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2998
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2998_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2998(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2998_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2998_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2914
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2914_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2914(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2914_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2914_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_214
// Description:	Constant
// Input:
// Output:
//	- name: Constant_214_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_214(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_214_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_214_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2221
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2221_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2221(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2221_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2221_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_762_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Constant_2191_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_766_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_764<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_762_0, Constant_2188_0, Convolution_764_0);
// Convolution_float_float_float_cuda_Convolution_766<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_762_0, Constant_2191_0, Convolution_766_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_766 : Convolution_float_float_float_cuda_Convolution_764

// Node name:	Convolution_764
// Description:	Convolution
// Input:
//	- name: Relu_762_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_764_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[9216];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_764_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_764_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_36<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_157_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_10_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_2833_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2748_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2747_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_621_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_622_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_630_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_613_0, Constant_157_0, DepthwiseConv2dNative_621_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_614_0, Constant_10_0, DepthwiseConv2dNative_622_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_616_0, Constant_2833_0, Convolution_620_0, Constant_2748_0, Add_630_0);
// Add_float_float_float_cuda_Add_2106<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_618_0, Constant_2747_0, BatchNormInference_624_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_621
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_157_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_621_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 32;
        const int in_width = 32;
        const int in_depth = 32;
        const int filter_height = 3;
        const int filter_width = 3;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 1;
        const int pad_width = 1;
        const int out_height = 32;
        const int out_width = 32;
        const int out_depth = 32;
        const int num_outputs = 32768;

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
// Node name:	DepthwiseConv2dNative_622
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_10_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_622_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 32;
        const int in_width = 32;
        const int in_depth = 32;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 2;
        const int pad_width = 2;
        const int out_height = 32;
        const int out_width = 32;
        const int out_depth = 32;
        const int num_outputs = 32768;

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
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2833_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2748_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_630_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2103<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_616_0, Constant_2833_0, BatchNormInference_623_0);
// Add_float_float_float_cuda_Add_2109<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_620_0, Constant_2748_0, BatchNormInference_625_0);
// Add_float_float_float_cuda_Add_630<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_625_0, BatchNormInference_623_0, Add_630_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(input2[tid], input3[tid]);
    float temp2 = add(temp1, temp0);
    output0[tid] = temp2;

}
// Node name:	Add_2106
// Description:	Add
// Input:
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2747_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_624_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2106_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_621_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_622_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_8_block_kernel(input5, input4, input7, input6, output2, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 639)
    {
        Add_float_float_float_cuda_Add_2106_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 576 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_15<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2809_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1466_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2946_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1468_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1398_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1472_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1473_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_55<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1466_0, Constant_2809_0, Slice_1421_0, Add_1472_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_56<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1468_0, Constant_2946_0, BatchNormInference_1398_0, Add_1473_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_56 : FusedKernel_float_float_float_float_cuda_Add_Add_55

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1466_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2809_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1421_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1472_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2574<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1466_0, Constant_2809_0, BatchNormInference_1470_0);
// Add_float_float_float_cuda_Add_1472<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1470_0, Slice_1421_0, Add_1472_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(temp0, input2[tid]);
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_55_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_138<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_227_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_540_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_229_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_539_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_12_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2782_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2780_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2781_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_509_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_494_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_354_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_360_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_548_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_547_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_546_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_538_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_551_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_520_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_521_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_522_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_541_0, Constant_227_0, DepthwiseConv2dNative_548_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_540_0, Constant_229_0, DepthwiseConv2dNative_547_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_546<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_539_0, Constant_12_0, DepthwiseConv2dNative_546_0);
// Add_float_float_float_cuda_Add_2064<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_526_0, Constant_2782_0, BatchNormInference_538_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_1<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_535_0, Constant_2780_0, Convolution_537_0, Constant_2781_0, Add_551_0);
// Add_float_float_float_cuda_Add_520<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_509_0, BatchNormInference_494_0, Add_520_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_510_0, Constant_354_0, DepthwiseConv2dNative_521_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_510_0, Constant_360_0, DepthwiseConv2dNative_522_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_546 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547
// Add_float_float_float_cuda_Add_520 : Add_float_float_float_cuda_Add_2064
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_521 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_522 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548

// Node name:	DepthwiseConv2dNative_548
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_227_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_548_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 32;
        const int in_width = 32;
        const int in_depth = 32;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 2;
        const int pad_width = 2;
        const int out_height = 32;
        const int out_width = 32;
        const int out_depth = 32;
        const int num_outputs = 32768;

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
// Node name:	DepthwiseConv2dNative_547
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_540_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_229_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_547_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 32;
        const int in_width = 32;
        const int in_depth = 32;
        const int filter_height = 3;
        const int filter_width = 3;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 1;
        const int pad_width = 1;
        const int out_height = 32;
        const int out_width = 32;
        const int out_depth = 32;
        const int num_outputs = 32768;

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
// Node name:	Add_2064
// Description:	Add
// Input:
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2782_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_538_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2064_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2780_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2781_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_551_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2058<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_535_0, Constant_2780_0, BatchNormInference_544_0);
// Add_float_float_float_cuda_Add_2061<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_537_0, Constant_2781_0, BatchNormInference_545_0);
// Add_float_float_float_cuda_Add_551<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_545_0, BatchNormInference_544_0, Add_551_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_1_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(input2[tid], input3[tid]);
    float temp2 = add(temp1, temp0);
    output0[tid] = temp2;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_4(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 767)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 768 && (int)blockIdx.x <= 831)
    {
        Add_float_float_float_cuda_Add_2064_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 768 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 832 && (int)blockIdx.x <= 895)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_1_block_kernel(input9, input8, input11, input10, output4, threadIdx.x, blockIdx.x - 832 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 896 && (int)blockIdx.x <= 959)
    {
        Add_float_float_float_cuda_Add_2064_block_kernel(input12, input13, output5, threadIdx.x, blockIdx.x - 896 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 960 && (int)blockIdx.x <= 1215)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_547_block_kernel(input14, input15, output6, threadIdx.x, blockIdx.x - 960 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 1216 && (int)blockIdx.x <= 1471)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_548_block_kernel(input14, input16, output7, threadIdx.x, blockIdx.x - 1216 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_4_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_4<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, output0, output1, output2, output3, output4, output5, output6, output7);
}
