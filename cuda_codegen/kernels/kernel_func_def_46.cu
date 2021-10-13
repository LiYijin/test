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
// Node name:	Constant_2908
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2908_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2908(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2908_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2908_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_292
// Description:	Constant
// Input:
// Output:
//	- name: Constant_292_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_292(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_292_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_292_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2835
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2835_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2835(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2835_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2835_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_73
// Description:	Constant
// Input:
// Output:
//	- name: Constant_73_0	type: float	shape: Shape{3, 3, 96, 1}
void Constant_float_cuda_Constant_73(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_73_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_73_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3456];
    bin_file.read(tmp_mem, 3456);
    cudaMemcpyAsync(output0, tmp_mem, 3456, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2599
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2599_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2599(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2599_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2599_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2801
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2801_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2801(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2801_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2801_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2644
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2644_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2644(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2644_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2644_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2898
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2898_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2898(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2898_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2898_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2996
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2996_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2996(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2996_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2996_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_114
// Description:	Constant
// Input:
// Output:
//	- name: Constant_114_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_114(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_114_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_114_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2257
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2257_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2257(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2257_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2257_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_885_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2260_0	type: float	shape: Shape{64, 192, 1, 1}
//	- name: Constant_889_0	type: float	shape: Shape{}
// Output:
//	- name: Convolution_887_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Pad_890_0	type: float	shape: Shape{1, 192, 33, 33}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_887<<<dim3(1, 32, 2), dim3(16, 1, 16), 0, 0>>>(Relu_885_0, Constant_2260_0, Convolution_887_0);
// Pad_float_float_float_cuda_Pad_890<<<dim3(3267, 1, 1), dim3(64, 1, 1), 0, 0>>>(Relu_885_0, Constant_889_0, Pad_890_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Convolution_887
// Description:	Convolution
// Input:
//	- name: Relu_885_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2260_0	type: float	shape: Shape{64, 192, 1, 1}
// Output:
//	- name: Convolution_887_0	type: float	shape: Shape{1, 64, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_887_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(16, 1, 16);
    const dim3 gridDim(1, 32, 2);
    const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 32, block_id / 32);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 3072);
    {
        float* compute = output0;{
           float compute_local[4];
          
          
          #pragma unroll
          for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
            compute_local[ff_c_init] = 0.000000e+00f;
            compute_local[(ff_c_init + 2)] = 0.000000e+00f;
          }
          #pragma unroll
          for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[((((rc_outer * 24576) + (((((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 5) * 1024)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 31))];
            }
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
              input1_shared[(((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] = input1[(((((((int)blockIdx.z) * 6144) + (((int)threadIdx.z) * 384)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 24) * 192)) + (rc_outer * 24)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 24))];
            }
            __syncthreads();
            #pragma unroll
            for (int rc_inner = 0; rc_inner < 24; ++rc_inner) {
              #pragma unroll
              for (int ff_c = 0; ff_c < 2; ++ff_c) {
                compute_local[ff_c] = (compute_local[ff_c] + (pad_temp_shared[((rc_inner * 32) + ((int)threadIdx.x))] * input1_shared[(((((int)threadIdx.z) * 48) + (ff_c * 24)) + rc_inner)]));
                compute_local[(ff_c + 2)] = (compute_local[(ff_c + 2)] + (pad_temp_shared[(((rc_inner * 32) + ((int)threadIdx.x)) + 16)] * input1_shared[(((((int)threadIdx.z) * 48) + (ff_c * 24)) + rc_inner)]));
              }
            }
          }
          #pragma unroll
          for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
            compute[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (ff_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x))] = compute_local[ff_inner_inner_inner];
            compute[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 2048)) + (ff_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] = compute_local[(ff_inner_inner_inner + 2)];
          }
        }


    }

}
// Node name:	Pad_890
// Description:	Pad
// Input:
//	- name: Relu_885_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_889_0	type: float	shape: Shape{}
// Output:
//	- name: Pad_890_0	type: float	shape: Shape{1, 192, 33, 33}
__device__ __noinline__ void Pad_float_float_float_cuda_Pad_890_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 64){
        return;
    }
    const dim3 blockDim(64, 1, 1);
    const dim3 gridDim(3267, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* in = input0;
    float* pad = input1;
    float* out = output0;
    if (tid < 209088)
    {
        size_t input_shape0 = 1;
        size_t input_shape1 = 192;
        size_t input_shape2 = 32;
        size_t input_shape3 = 32;
        uint32_t input_strides0 = 196608;
        uint32_t input_strides1 = 1024;
        uint32_t input_strides2 = 32;
        uint32_t input_strides3 = 1;
        uint32_t output_strides0 = 209088;
        uint32_t output_strides1 = 1089;
        uint32_t output_strides2 = 33;
        uint32_t output_strides3 = 1;
        uint32_t padding_below0 = 0;
        uint32_t padding_below1 = 0;
        uint32_t padding_below2 = 0;
        uint32_t padding_below3 = 0;
        uint32_t padding_interior0 = 0;
        uint32_t padding_interior1 = 0;
        uint32_t padding_interior2 = 0;
        uint32_t padding_interior3 = 0;
        bool in_bounds = true;
        uint32_t output_pixel = tid;
        uint32_t input_pixel = 0;
        int32_t input, input_dil;
        input_dil = output_pixel / output_strides0 - padding_below0;
        input = input_dil / (padding_interior0 + 1);
        input_dil %= (padding_interior0 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape0) && (input_dil == 0);
        input_pixel += input * input_strides0;
        output_pixel %= output_strides0;
        input_dil = output_pixel / output_strides1 - padding_below1;
        input = input_dil / (padding_interior1 + 1);
        input_dil %= (padding_interior1 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape1) && (input_dil == 0);
        input_pixel += input * input_strides1;
        output_pixel %= output_strides1;
        input_dil = output_pixel / output_strides2 - padding_below2;
        input = input_dil / (padding_interior2 + 1);
        input_dil %= (padding_interior2 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape2) && (input_dil == 0);
        input_pixel += input * input_strides2;
        output_pixel %= output_strides2;
        input_dil = output_pixel / output_strides3 - padding_below3;
        input = input_dil / (padding_interior3 + 1);
        input_dil %= (padding_interior3 + 1);
        in_bounds = in_bounds && (input >= 0) && (input < input_shape3) && (input_dil == 0);
        input_pixel += input * input_strides3;
        out[tid] = (in_bounds) ? in[input_pixel] : *pad;
    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Pad_54(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[6144];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_887_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 3330)
    {
        Pad_float_float_float_cuda_Pad_890_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Pad_54_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Pad_54<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_699_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2152_0	type: float	shape: Shape{32, 192, 1, 1}
//	- name: Constant_2155_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_701_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_703_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_701<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_699_0, Constant_2152_0, Convolution_701_0);
// Convolution_float_float_float_cuda_Convolution_703<<<dim3(1, 32, 2), dim3(32, 1, 8), 0, 0>>>(Relu_699_0, Constant_2155_0, Convolution_703_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_703 : Convolution_float_float_float_cuda_Convolution_701

// Node name:	Convolution_701
// Description:	Convolution
// Input:
//	- name: Relu_699_0	type: float	shape: Shape{1, 192, 32, 32}
//	- name: Constant_2152_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_701_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_701_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_27(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[9216];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_701_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_701_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_27_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_27<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_973_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2299_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3058_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_975_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2305_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3062_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_974_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2302_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3060_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_995_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_997_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_996_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_976_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Relu_float_float_cuda_Relu_971<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_968_0, Relu_971_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_973_0, Constant_2299_0, Constant_3058_0, Relu_995_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3061<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_975_0, Constant_2305_0, Constant_3062_0, Relu_997_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3059<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_974_0, Constant_2302_0, Constant_3060_0, Relu_996_0);
// Add_float_float_float_cuda_Add_976<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_970_0, AvgPool_970_0, Add_976_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3061 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3059 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057

// Node name:	Relu_971
// Description:	Relu
// Input:
//	- name: Slice_968_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_971_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Relu_float_float_cuda_Relu_971_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Matched_Pattern_3057
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_973_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2299_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3058_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_995_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_976
// Description:	Add
// Input:
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_970_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_976_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_976_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Relu_float_float_cuda_Relu_971_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(input1, input2, input3, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(input4, input5, input6, output2, threadIdx.x, blockIdx.x - 96 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3057_block_kernel(input7, input8, input9, output3, threadIdx.x, blockIdx.x - 160 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 224 && (int)blockIdx.x <= 255)
    {
        Add_float_float_float_cuda_Add_976_block_kernel(input10, input10, output4, threadIdx.x, blockIdx.x - 224 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_68<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1681_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2701_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3176_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1682_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2704_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3178_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1705_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2713_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1703_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2707_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: DepthwiseConv2dNative_1704_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2710_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Relu_1701_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1702_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1713_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1709_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1711_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3175<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1681_0, Constant_2701_0, Constant_3176_0, Relu_1701_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3177<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1682_0, Constant_2704_0, Constant_3178_0, Relu_1702_0);
// Convolution_float_float_float_cuda_Convolution_1713<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1705_0, Constant_2713_0, Convolution_1713_0);
// Convolution_float_float_float_cuda_Convolution_1709<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1703_0, Constant_2707_0, Convolution_1709_0);
// Convolution_float_float_float_cuda_Convolution_1711<<<dim3(1, 4, 16), dim3(8, 2, 8), 0, 0>>>(DepthwiseConv2dNative_1704_0, Constant_2710_0, Convolution_1711_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3177 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3175
// Convolution_float_float_float_cuda_Convolution_1709 : Convolution_float_float_float_cuda_Convolution_1713
// Convolution_float_float_float_cuda_Convolution_1711 : Convolution_float_float_float_cuda_Convolution_1713

// Node name:	Matched_Pattern_3175
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1681_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2701_0	type: float	shape: Shape{128, 128, 1, 1}
//	- name: Constant_3176_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1701_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3175_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Convolution_1713
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1705_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2713_0	type: float	shape: Shape{128, 128, 1, 1}
// Output:
//	- name: Convolution_1713_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1713_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_171(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[1536];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3175_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3175_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_1713_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Convolution_float_float_float_cuda_Convolution_1713_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Convolution_float_float_float_cuda_Convolution_1713_block_kernel(input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_171_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_171<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, output0, output1, output2, output3, output4);
}
