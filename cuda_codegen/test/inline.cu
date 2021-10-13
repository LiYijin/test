// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_426
// Description:	Constant
// Input:
// Output:
//	- name: Constant_426_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_426(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_426_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_426_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_116
// Description:	Constant
// Input:
// Output:
//	- name: Constant_116_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2521
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2521_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2521(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2521_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2521_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2071
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2071_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2071(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2071_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2071_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_57
// Description:	Constant
// Input:
// Output:
//	- name: Constant_57_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_57(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_57_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_57_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_250
// Description:	Constant
// Input:
// Output:
//	- name: Constant_250_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_250(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_250_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_250_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2910
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2910_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2910(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2910_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2910_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2803
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2803_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2803(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2803_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2803_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3174
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3174_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3174(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3174_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3174_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2704
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2704_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2704(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2704_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2704_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2137
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2137_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2530
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2530_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2530(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2530_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2530_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2876_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1715_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1671_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1717_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2973_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1651_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1724_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1725_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_71<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1715_0, Constant_2876_0, Slice_1671_0, Add_1724_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_72<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1717_0, Constant_2973_0, BatchNormInference_1651_0, Add_1725_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_72 : FusedKernel_float_float_float_float_cuda_Add_Add_71

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1715_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2876_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1671_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1724_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2718<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1715_0, Constant_2876_0, BatchNormInference_1721_0);
// Add_float_float_float_cuda_Add_1724<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1721_0, Slice_1671_0, Add_1724_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_71_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_174(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_71_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_71_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_174_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_174<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	Constant_290
// Description:	Constant
// Input:
// Output:
//	- name: Constant_290_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_290(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_290_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_290_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_1338_0	type: float	shape: Shape{}
//	- name: Constant_2509_0	type: float	shape: Shape{128, 384, 1, 1}
// Output:
//	- name: Pad_1339_0	type: float	shape: Shape{1, 384, 17, 17}
//	- name: Convolution_1341_0	type: float	shape: Shape{1, 128, 16, 16}
// Fused functions:
// Pad_float_float_float_cuda_Pad_1339<<<dim3(1734, 1, 1), dim3(64, 1, 1), 0, 0>>>(Relu_1336_0, Constant_1338_0, Pad_1339_0);
// Convolution_float_float_float_cuda_Convolution_1341<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(Relu_1336_0, Constant_2509_0, Convolution_1341_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Pad_1339
// Description:	Pad
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_1338_0	type: float	shape: Shape{}
// Output:
//	- name: Pad_1339_0	type: float	shape: Shape{1, 384, 17, 17}
__device__ __noinline__ void Pad_float_float_float_cuda_Pad_1339_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 64){
        return;
    }
    const dim3 blockDim(64, 1, 1);
    const dim3 gridDim(1734, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* in = input0;
    float* pad = input1;
    float* out = output0;
    if (tid < 110976)
    {
        size_t input_shape0 = 1;
        size_t input_shape1 = 384;
        size_t input_shape2 = 16;
        size_t input_shape3 = 16;
        uint32_t input_strides0 = 98304;
        uint32_t input_strides1 = 256;
        uint32_t input_strides2 = 16;
        uint32_t input_strides3 = 1;
        uint32_t output_strides0 = 110976;
        uint32_t output_strides1 = 289;
        uint32_t output_strides2 = 17;
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
// Node name:	Convolution_1341
// Description:	Convolution
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
//	- name: Constant_2509_0	type: float	shape: Shape{128, 384, 1, 1}
// Output:
//	- name: Convolution_1341_0	type: float	shape: Shape{1, 128, 16, 16}
__device__ __forceinline__ void Convolution_float_float_float_cuda_Convolution_1341_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    
    const dim3 blockDim(8, 2, 16);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    /*
    __shared__ float pad_temp_shared[512];
    __shared__ float input1_shared[1024];
    */
    //__shared__ char shared_buffer1[6144];
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
    
    {
        float* compute = output0;{
           float compute_local[2];
          
          
          compute_local[0] = 0.000000e+00f;
          compute_local[1] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7))];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4))];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 1)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 2)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 3)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 32)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 33)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 34)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 35)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 16384)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 64)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 65)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 66)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 67)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 24576)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 96)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 97)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 98)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 99)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 32768)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 128)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 129)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 130)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 131)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 40960)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 40960)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 160)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 161)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 162)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 163)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 49152)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 49152)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 192)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 193)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 194)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 195)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 57344)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 57344)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 224)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 225)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 226)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 227)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 65536)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 256)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 257)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 258)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 259)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 73728)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 73728)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 288)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 289)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 290)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 291)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 81920)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 81920)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 320)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 321)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 322)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 323)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 90112)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 90112)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 352)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 353)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 354)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 355)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          compute[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
          compute[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)] = compute_local[1];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Pad_Convolution_119(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[6144];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 1733)
    {
        Pad_float_float_float_cuda_Pad_1339_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 1734 && (int)blockIdx.x <= 1797)
    {
        Convolution_float_float_float_cuda_Convolution_1341_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 1734 + 0, shared_buffer);
    }

}
extern "C"  __global__  void pad_test(float* input0, float* input1, float* output0)
{
    /*
    __shared__ char shared_buffer[6144];
    Pad_float_float_float_cuda_Pad_1339_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    */
   uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* in = input0;
    float* pad = input1;
    float* out = output0;
    if (tid < 110976)
    {
        size_t input_shape0 = 1;
        size_t input_shape1 = 384;
        size_t input_shape2 = 16;
        size_t input_shape3 = 16;
        uint32_t input_strides0 = 98304;
        uint32_t input_strides1 = 256;
        uint32_t input_strides2 = 16;
        uint32_t input_strides3 = 1;
        uint32_t output_strides0 = 110976;
        uint32_t output_strides1 = 289;
        uint32_t output_strides2 = 17;
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
extern "C" __global__  void conv_test(float* input0, float* input1, float* output0)
{
    
    __shared__ char shared_buffer[6144];
    Convolution_float_float_float_cuda_Convolution_1341_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x, shared_buffer);
    
   /*
   __shared__ float pad_temp_shared[512];
   __shared__ float input1_shared[1024];
   {
        float* compute = output0;{
           float compute_local[2];
          
          
          compute_local[0] = 0.000000e+00f;
          compute_local[1] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7))];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4))];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 1)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 2)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 3)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 32)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 33)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 34)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 35)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 16384)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 64)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 65)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 66)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 67)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 24576)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 96)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 97)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 98)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 99)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 32768)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 128)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 129)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 130)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 131)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 40960)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 40960)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 160)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 161)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 162)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 163)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 49152)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 49152)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 192)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 193)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 194)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 195)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 57344)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 57344)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 224)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 225)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 226)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 227)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 65536)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 65536)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 256)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 257)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 258)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 259)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 73728)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 73728)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 288)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 289)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 290)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 291)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 81920)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 81920)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 320)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 321)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 322)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 323)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 2) * 16)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 90112)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.x) * 2) + 1) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 2) + 1) & 7)) + 90112)];
          input1_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 352)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 353)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 2)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 354)];
          input1_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + 3)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 4)) + 355)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[((((int)threadIdx.y) * 8) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 32) + 512)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 513)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 514)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 515)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 516)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 517)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 518)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 519)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 520)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 521)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 522)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 523)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 524)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 525)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 526)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 527)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 528)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 529)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 530)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 531)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 532)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 533)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 534)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 535)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 536)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 537)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 538)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 539)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 540)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 541)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 542)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute_local[1] = (compute_local[1] + (pad_temp_shared[(((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 543)]));
          compute[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
          compute[(((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)] = compute_local[1];
        }
        

    }
    */
}



extern void BlockFusionKernel_float_float_float_float_float_cuda_Pad_Convolution_119_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Pad_Convolution_119<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
