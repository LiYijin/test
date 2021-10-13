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
// Node name:	Constant_301
// Description:	Constant
// Input:
// Output:
//	- name: Constant_301_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_301(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_301_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_301_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_285
// Description:	Constant
// Input:
// Output:
//	- name: Constant_285_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_285(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_285_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_285_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: Constant_7_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2781
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2781_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2781(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2781_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2781_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2722
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2722_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2722(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2722_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2722_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[393216];
    bin_file.read(tmp_mem, 393216);
    cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2911
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2911_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2911(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2911_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2911_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2500
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2500_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2500(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2500_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2500_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1338
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1338_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_1338(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1338_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1338_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_458
// Description:	Constant
// Input:
// Output:
//	- name: Constant_458_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_458(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_458_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_458_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_35
// Description:	Constant
// Input:
// Output:
//	- name: Constant_35_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_35(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_35_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_35_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3030
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3030_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3030(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3030_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3030_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2254_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_872_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2257_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_879_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_877<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_871_0, Constant_2254_0, Convolution_877_0);
// Convolution_float_float_float_cuda_Convolution_879<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_872_0, Constant_2257_0, Convolution_879_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_879 : Convolution_float_float_float_cuda_Convolution_877

// Node name:	Convolution_877
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_871_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2254_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_877_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_877_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_877_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_Convolution_Convolution_52<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	Convolution_466
// Description:	Convolution
// Input:
//	- name: Reshape_464_0	type: float	shape: Shape{1, 3, 32, 32}
//	- name: Constant_2011_0	type: float	shape: Shape{96, 3, 3, 3}
// Output:
//	- name: Convolution_466_0	type: float	shape: Shape{1, 96, 32, 32}
extern "C" __global__  void Convolution_float_float_float_cuda_Convolution_466(float* input0, float* input1, float* output0)
{
    __shared__ float pad_temp_shared[408];
    __shared__ float input1_shared[864];
    {
        float* compute = output0;{
           float compute_local[4];
          
          
          #pragma unroll
          for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
            compute_local[yy_c_init] = 0.000000e+00f;
            compute_local[(yy_c_init + 2)] = 0.000000e+00f;
          }
          if (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) < 408) {
            if (((int)threadIdx.x) < 26) {
              pad_temp_shared[((((int)threadIdx.z) * 26) + ((int)threadIdx.x))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 136) / 34))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 136) / 34)) < 33)) && (1 <= (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 34))) && ((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 34) < 33)) ? input0[((((((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) / 136) * 1024) + (((int)blockIdx.y) * 64)) + (((((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 136) / 34) * 32)) + (((((int)threadIdx.z) * 26) + ((int)threadIdx.x)) % 34)) - 33)] : 0.000000e+00f);
            }
          }
          #pragma unroll
          for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
            if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 27)) < 32) {
              if (((((int)threadIdx.z) * 6) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 9)) < 96) {
                if (((((int)threadIdx.z) * 18) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 3)) < 288) {
                  if ((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 864) {
                    if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 54) {
                      input1_shared[(((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input1[((((((int)blockIdx.z) * 864) + (((int)threadIdx.z) * 54)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
                    }
                  }
                }
              }
            }
          }
          __syncthreads();
          #pragma unroll
          for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
            #pragma unroll
            for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
              #pragma unroll
              for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
                #pragma unroll
                for (int yy_c = 0; yy_c < 2; ++yy_c) {
                  compute_local[yy_c] = (compute_local[yy_c] + (pad_temp_shared[(((((rc_inner * 136) + (yy_c * 34)) + (ry_inner * 34)) + ((int)threadIdx.x)) + rx_inner)] * input1_shared[((((((int)threadIdx.z) * 27) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
                  compute_local[(yy_c + 2)] = (compute_local[(yy_c + 2)] + (pad_temp_shared[(((((rc_inner * 136) + (yy_c * 34)) + (ry_inner * 34)) + ((int)threadIdx.x)) + rx_inner)] * input1_shared[(((((((int)threadIdx.z) * 27) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
                }
              }
            }
          }
          #pragma unroll
          for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
            compute[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 64)) + (yy_inner_inner_inner * 32)) + ((int)threadIdx.x))] = compute_local[yy_inner_inner_inner];
            compute[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 64)) + (yy_inner_inner_inner * 32)) + ((int)threadIdx.x)) + 16384)] = compute_local[(yy_inner_inner_inner + 2)];
          }
        }


    }

}
extern void Convolution_float_float_float_cuda_Convolution_466_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Convolution_float_float_float_cuda_Convolution_466<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Constant_2741
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2741_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2741(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2741_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2741_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_1096_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1101_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2377_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3082_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1099_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2371_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3078_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1100_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2374_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3080_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1102_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1123_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1121_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1122_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Relu_float_float_cuda_Relu_1097<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_1094_0, Relu_1097_0);
// Add_float_float_float_cuda_Add_1102<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1096_0, AvgPool_1096_0, Add_1102_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1101_0, Constant_2377_0, Constant_3082_0, Relu_1123_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3077<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1099_0, Constant_2371_0, Constant_3078_0, Relu_1121_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3079<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1100_0, Constant_2374_0, Constant_3080_0, Relu_1122_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3077 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3079 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081

// Node name:	Relu_1097
// Description:	Relu
// Input:
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Relu_float_float_cuda_Relu_1097_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Add_1102
// Description:	Add
// Input:
//	- name: AvgPool_1096_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_1096_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1102_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1102_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Matched_Pattern_3081
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1101_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2377_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3082_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1123_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_86(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Relu_float_float_cuda_Relu_1097_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_1102_block_kernel(input1, input1, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081_block_kernel(input2, input3, input4, output2, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081_block_kernel(input5, input6, input7, output3, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3081_block_kernel(input8, input9, input10, output4, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_86_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_86<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4);
}
