// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
// Node name:	Constant_254
// Description:	Constant
// Input:
// Output:
//	- name: Constant_254_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_254(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_254_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_254_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2815
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2815_0	type: float	shape: Shape{1, 96, 32, 32}
void Constant_float_cuda_Constant_2815(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2815_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2815_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[393216];
    bin_file.read(tmp_mem, 393216);
    cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2731
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2731_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2731(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2731_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2731_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2029
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2029_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2029(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2029_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2029_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2503
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2503_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2503(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2503_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2503_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2278
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2278_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2278(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2278_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2278_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2767
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2767_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2767(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2767_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2767_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2758
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2758_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2758(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2758_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2758_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2748
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2748_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2748(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2748_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2748_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2853
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2853_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2853(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2853_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2853_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2776
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2776_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2776(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2776_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2776_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_171
// Description:	Constant
// Input:
// Output:
//	- name: Constant_171_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_171(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_171_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_171_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2473_0	type: float	shape: Shape{128, 384, 1, 1}
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Constant_2476_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Relu_1283_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Convolution_1280_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(Relu_1276_0, Constant_2473_0, Constant_3108_0, Relu_1283_0);
// Convolution_float_float_float_cuda_Convolution_1280<<<dim3(1, 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_1276_0, Constant_2476_0, Convolution_1280_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Matched_Pattern_3107
// Description:	Matched_Pattern
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2473_0	type: float	shape: Shape{128, 384, 1, 1}
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: Relu_1283_0	type: float	shape: Shape{1, 128, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(8, 2, 16);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    {
        float* compute = output0;{
           float compute1[2];
          
          
          compute1[0] = 0.000000e+00f;
          compute1[1] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7))];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4))];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 1)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 2)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 3)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 32)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 33)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 34)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 35)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 16384)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 64)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 65)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 66)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 67)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 24576)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 96)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 97)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 98)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 99)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 32768)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 128)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 129)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 130)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 131)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 40960)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 40960)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 160)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 161)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 162)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 163)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 49152)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 49152)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 192)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 193)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 194)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 195)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 57344)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 57344)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 224)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 225)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 226)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 227)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 65536)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 256)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 257)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 258)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 259)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 73728)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 73728)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 288)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 289)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 290)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 291)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 81920)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 81920)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 320)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 321)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 322)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 323)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 90112)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 90112)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 352)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 353)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 354)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 355)];
          __syncthreads();
          compute1[0] = (compute1[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute1[0] = (compute1[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute1[1] = (compute1[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          compute[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = max((compute1[0] + input2[((((int)blockIdx.z) * 32) + ((int)threadIdx.z))]), 0.000000e+00f);
          compute[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)] = max((compute1[1] + input2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16)]), 0.000000e+00f);
        }


    }

}
// Node name:	Convolution_1280
// Description:	Convolution
// Input:
//	- name: Relu_1276_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2476_0	type: float	shape: Shape{64, 384, 1, 1}
// Output:
//	- name: Convolution_1280_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1280_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[6144];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3107_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_1280_block_kernel(input0, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Matched_Pattern_Convolution_110<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	Constant_2852
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2852_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2852(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2852_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2852_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
