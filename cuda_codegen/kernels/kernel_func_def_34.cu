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
// Node name:	Constant_324
// Description:	Constant
// Input:
// Output:
//	- name: Constant_324_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_324(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_324_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_324_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2992
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2992_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2992(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2992_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2992_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_339
// Description:	Constant
// Input:
// Output:
//	- name: Constant_339_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_339(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_339_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_339_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2575
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2575_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2575(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2575_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2575_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2446
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2446_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2446(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2446_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2446_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2335
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2335_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2335(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2335_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2335_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_370
// Description:	Constant
// Input:
// Output:
//	- name: Constant_370_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_370(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_370_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_370_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2671
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2671_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2671(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2671_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2671_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_401
// Description:	Constant
// Input:
// Output:
//	- name: Constant_401_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_401(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_401_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_401_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2845
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2845_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2845(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2845_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2845_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3054
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3054_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3054(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3054_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3054_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_403
// Description:	Constant
// Input:
// Output:
//	- name: Constant_403_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_403(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_403_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_403_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	AvgPool_1352
// Description:	AvgPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: AvgPool_1352_0	type: float	shape: Shape{1, 128, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1352(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 16, 16));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 8, 8));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1089_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2902_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1091_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2904_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1092_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1093_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1095_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2367<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1089_0, Constant_2902_0, BatchNormInference_1092_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_36<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1091_0, Constant_2904_0, Relu_1095_0, BatchNormInference_1093_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2367
// Description:	Add
// Input:
//	- name: Convolution_1089_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2902_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1092_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2367_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1091_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2904_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1095_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1093_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2370<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1091_0, Constant_2904_0, BatchNormInference_1093_0);
// Relu_float_float_cuda_Relu_1095<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1093_0, Relu_1095_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_36_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_84(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_2367_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_36_block_kernel(input2, input3, output2, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_84_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_84<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	AvgPool_469
// Description:	AvgPool
// Input:
//	- name: BatchNormInference_467_0	type: float	shape: Shape{1, 96, 32, 32}
// Output:
//	- name: AvgPool_469_0	type: float	shape: Shape{1, 96, 32, 32}
void AvgPool_float_float_cuda_lib_AvgPool_469(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 96, 32, 32));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 96, 32, 32));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,3, 3, 1, 1, 1, 1));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
