// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
}
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_61
// Description:	Constant
// Input:
// Output:
//	- name: Constant_61_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_61(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_61_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_61_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2739
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2739_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2739(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2739_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2739_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2780
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2780_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2780(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2780_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2780_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2545
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2545_0	type: float	shape: Shape{128, 512, 1, 1}
void Constant_float_cuda_Constant_2545(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2545_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2545_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: Constant_4_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2674
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2674_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2674(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2674_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2674_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3090
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3090_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3090(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3090_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3090_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_170_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2834
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2834_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2834(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2834_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2834_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2934
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2934_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2934(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2934_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2934_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_239
// Description:	Constant
// Input:
// Output:
//	- name: Constant_239_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_239(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_239_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_239_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2837
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2837_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2837(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2837_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2837_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2896
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2896_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2896(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2896_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2896_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2797_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1152_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1154_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2911_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1155_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_40<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1152_0, Constant_2797_0, Relu_1158_0, BatchNormInference_1155_0);
// Add_float_float_float_cuda_Add_2406<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1154_0, Constant_2911_0, BatchNormInference_1156_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1152_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2797_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1155_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2403<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1152_0, Constant_2797_0, BatchNormInference_1155_0);
// Relu_float_float_cuda_Relu_1158<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1155_0, Relu_1158_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_40_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output1[tid] = temp0;
    output0[tid] = temp1;

}
// Node name:	Add_2406
// Description:	Add
// Input:
//	- name: Convolution_1154_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2911_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2406_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_93(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_40_block_kernel(input1, input0, output1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2406_block_kernel(input2, input3, output2, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_93_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_93<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2768_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2853_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_766_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_767_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_768_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_771_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2190<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_764_0, Constant_2768_0, BatchNormInference_767_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_19<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_766_0, Constant_2853_0, Relu_771_0, BatchNormInference_768_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2190
// Description:	Add
// Input:
//	- name: Convolution_764_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2768_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_767_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2190_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_766_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2853_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_771_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_768_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2193<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_766_0, Constant_2853_0, BatchNormInference_768_0);
// Relu_float_float_cuda_Relu_771<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_768_0, Relu_771_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_19_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = relu(temp0);
    output1[tid] = temp0;
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_37(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2190_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_19_block_kernel(input3, input2, output2, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_37_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_37<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
