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
// Node name:	Constant_2725
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2725_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2725(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2725_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2725_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2851
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2851_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2851(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2851_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2851_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_90
// Description:	Constant
// Input:
// Output:
//	- name: Constant_90_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_90(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_90_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_90_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2873
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2873_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2873(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2873_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2873_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2362
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2362_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2362(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2362_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2362_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2907
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2907_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2907(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2907_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2907_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_84
// Description:	Constant
// Input:
// Output:
//	- name: Constant_84_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_84(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_84_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_84_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[25088];
    bin_file.read(tmp_mem, 25088);
    cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2747
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2747_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2747(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2747_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2747_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2161
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2161_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2161_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2206
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2206_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2206(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2206_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2206_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2978_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_474_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2816_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_472_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2984_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_480_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2738_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_478_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2035_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2982_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_477_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2032_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2980_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_469_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2017_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2020_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Convolution_476_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2739_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_485_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_498_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_504_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_494_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_512_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_511_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_484_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_482_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_489_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2016<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_471_0, Constant_2978_0, BatchNormInference_485_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_2<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_474_0, Constant_2816_0, Relu_498_0, BatchNormInference_488_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(DepthwiseConv2dNative_472_0, Constant_2038_0, Constant_2984_0, Relu_504_0);
// Add_float_float_float_cuda_Add_2028<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_480_0, Constant_2738_0, BatchNormInference_494_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2981<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(DepthwiseConv2dNative_478_0, Constant_2035_0, Constant_2982_0, Relu_512_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2979<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(DepthwiseConv2dNative_477_0, Constant_2032_0, Constant_2980_0, Relu_511_0);
// Convolution_float_float_float_cuda_Convolution_484<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(AvgPool_469_0, Constant_2017_0, Convolution_484_0);
// Convolution_float_float_float_cuda_Convolution_482<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(AvgPool_469_0, Constant_2020_0, Convolution_482_0);
// Add_float_float_float_cuda_Add_2025<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_476_0, Constant_2739_0, BatchNormInference_489_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Add_float_float_float_cuda_Add_2028 : Add_float_float_float_cuda_Add_2016
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2981 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2979 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983
// Convolution_float_float_float_cuda_Convolution_482 : Convolution_float_float_float_cuda_Convolution_484
// Add_float_float_float_cuda_Add_2025 : Add_float_float_float_cuda_Add_2016

// Node name:	Add_2016
// Description:	Add
// Input:
//	- name: Convolution_471_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2978_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_485_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2016_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_474_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2816_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_498_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_488_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2031<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_474_0, Constant_2816_0, BatchNormInference_488_0);
// Relu_float_float_cuda_Relu_498<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_488_0, Relu_498_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_2_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Matched_Pattern_2983
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_472_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
//	- name: Constant_2984_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_504_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(32, 1, 8);
    const dim3 gridDim(1, 32, 2);
    const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    {
        float* compute = output0;{
           float compute1[2];
          
          
          compute1[0] = 0.000000e+00f;
          compute1[1] = 0.000000e+00f;
          #pragma unroll
          for (int rc_outer = 0; rc_outer < 6; ++rc_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[(((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 31))];
            }
            input1_shared[((((int)threadIdx.z) * 32) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) + ((((int)threadIdx.x) >> 4) * 96)) + (rc_outer * 16)) + (((int)threadIdx.x) & 15))];
            __syncthreads();
            #pragma unroll
            for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              compute1[0] = (compute1[0] + (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
              compute1[1] = (compute1[1] + (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] * input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 128)]));
            }
          }
          compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = max((compute1[0] + input2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]), 0.000000e+00f);
          compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 8192)] = max((compute1[1] + input2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8)]), 0.000000e+00f);
        }


    }

}
// Node name:	Convolution_484
// Description:	Convolution
// Input:
//	- name: AvgPool_469_0	type: float	shape: Shape{1, 96, 32, 32}
//	- name: Constant_2017_0	type: float	shape: Shape{32, 96, 1, 1}
// Output:
//	- name: Convolution_484_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_484_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(32, 1, 8);
    const dim3 gridDim(1, 32, 2);
    const dim3 threadIdx(thread_id % 32, 0, thread_id / 32);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    {
        float* compute = output0;{
           float compute_local[2];
          
          
          compute_local[0] = 0.000000e+00f;
          compute_local[1] = 0.000000e+00f;
          #pragma unroll
          for (int rc_outer = 0; rc_outer < 6; ++rc_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[(((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 31))];
            }
            input1_shared[((((int)threadIdx.z) * 32) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) + ((((int)threadIdx.x) >> 4) * 96)) + (rc_outer * 16)) + (((int)threadIdx.x) & 15))];
            __syncthreads();
            #pragma unroll
            for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              compute_local[0] = (compute_local[0] + (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
              compute_local[1] = (compute_local[1] + (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] * input1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 128)]));
            }
          }
          compute[((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = compute_local[0];
          compute[(((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 8192)] = compute_local[1];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Add_1(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* input17, float* input18, float* input19, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7, float* output8, float* output9)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2016_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_2_block_kernel(input2, input3, output2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(input4, input5, input6, output3, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Add_float_float_float_cuda_Add_2016_block_kernel(input7, input8, output4, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(input9, input10, input11, output5, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 383)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2983_block_kernel(input12, input13, input14, output6, threadIdx.x, blockIdx.x - 320 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447)
    {
        Convolution_float_float_float_cuda_Convolution_484_block_kernel(input15, input16, output7, threadIdx.x, blockIdx.x - 384 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 448 && (int)blockIdx.x <= 511)
    {
        Convolution_float_float_float_cuda_Convolution_484_block_kernel(input15, input17, output8, threadIdx.x, blockIdx.x - 448 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575)
    {
        Add_float_float_float_cuda_Add_2016_block_kernel(input18, input19, output9, threadIdx.x, blockIdx.x - 512 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Add_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* input17, float* input18, float* input19, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7, float* output8, float* output9) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_Matched_Pattern_Add_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Add_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, output0, output1, output2, output3, output4, output5, output6, output7, output8, output9);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_223_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_999_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_285_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_2790_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2894_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1010_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1008_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2893_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1004_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1020_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1016_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_998_0, Constant_223_0, DepthwiseConv2dNative_1003_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_999_0, Constant_285_0, DepthwiseConv2dNative_1004_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1006_0, Constant_2790_0, Convolution_1010_0, Constant_2894_0, Add_1020_0);
// Add_float_float_float_cuda_Add_2319<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1008_0, Constant_2893_0, BatchNormInference_1016_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_1003
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_998_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_223_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1003_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(128, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 16;
        const int in_width = 16;
        const int in_depth = 64;
        const int filter_height = 3;
        const int filter_width = 3;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 1;
        const int pad_width = 1;
        const int out_height = 16;
        const int out_width = 16;
        const int out_depth = 64;
        const int num_outputs = 16384;

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
// Node name:	DepthwiseConv2dNative_1004
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_999_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_285_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1004_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(128, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 16;
        const int in_width = 16;
        const int in_depth = 64;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 2;
        const int pad_width = 2;
        const int out_height = 16;
        const int out_width = 16;
        const int out_depth = 64;
        const int num_outputs = 16384;

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
//	- name: Convolution_1006_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2790_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1010_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2894_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1020_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2316<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1006_0, Constant_2790_0, BatchNormInference_1015_0);
// Add_float_float_float_cuda_Add_2322<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1010_0, Constant_2894_0, BatchNormInference_1017_0);
// Add_float_float_float_cuda_Add_1020<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1017_0, BatchNormInference_1015_0, Add_1020_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(input2[tid], input3[tid]);
    float temp2 = add(temp1, temp0);
    output0[tid] = temp2;

}
// Node name:	Add_2319
// Description:	Add
// Input:
//	- name: Convolution_1008_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2893_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1016_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2319_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1003_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1004_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 287)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_29_block_kernel(input5, input4, input7, input6, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 288 && (int)blockIdx.x <= 319)
    {
        Add_float_float_float_cuda_Add_2319_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 288 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_fused_kernel_Add_71<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3);
}
// Node name:	Constant_2882
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2882_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2882(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2882_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2882_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_555_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2907_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_553_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2825_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2783_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_557_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_549_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_412_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_550_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_339_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_567_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_558_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_559_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2070<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_555_0, Constant_2907_0, BatchNormInference_561_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_553_0, Constant_2825_0, Convolution_557_0, Constant_2783_0, Add_567_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_549_0, Constant_412_0, DepthwiseConv2dNative_558_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_550_0, Constant_339_0, DepthwiseConv2dNative_559_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2070
// Description:	Add
// Input:
//	- name: Convolution_555_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2907_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_561_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2070_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_553_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2825_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_557_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2783_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_567_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2067<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_553_0, Constant_2825_0, BatchNormInference_560_0);
// Add_float_float_float_cuda_Add_2073<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_557_0, Constant_2783_0, BatchNormInference_562_0);
// Add_float_float_float_cuda_Add_567<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_562_0, BatchNormInference_560_0, Add_567_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_558
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_549_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_412_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_558_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_559
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_550_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_339_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_559_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_6(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2070_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_4_block_kernel(input2, input3, input5, input4, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 383)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_558_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_559_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_6_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_fused_kernel_DepthwiseConv2dNative_DepthwiseConv2dNative_6<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_748_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2185_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_753_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_755_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_753<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_747_0, Constant_2182_0, Convolution_753_0);
// Convolution_float_float_float_cuda_Convolution_755<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_748_0, Constant_2185_0, Convolution_755_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_755 : Convolution_float_float_float_cuda_Convolution_753

// Node name:	Convolution_753
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_747_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2182_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_753_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_753_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(16, 2, 8);
    const dim3 gridDim(2, 16, 2);
    const dim3 threadIdx(thread_id % 16, thread_id / 16 % 2, thread_id / 32);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 16, block_id / 32);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    {
        float* compute = output0;{
           float compute_local[2];
          
          
          for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
            compute_local[ff_c_init] = 0.000000e+00f;
          }
          for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
            __syncthreads();
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[(((((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.y) * 64)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4) * 32)) + (((int)blockIdx.x) * 16)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15))];
            }
            input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + ((int)threadIdx.x))];
            __syncthreads();
            for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              for (int ff_c = 0; ff_c < 2; ++ff_c) {
                compute_local[ff_c] = (compute_local[ff_c] + (pad_temp_shared[(((rc_inner * 32) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] * input1_shared[(((((int)threadIdx.z) * 32) + (ff_c * 16)) + rc_inner)]));
              }
            }
          }
          for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
            compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (ff_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))] = compute_local[ff_inner_inner_inner];
          }
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_753_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_753_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_34<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
