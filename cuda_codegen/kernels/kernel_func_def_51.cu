// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_28(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1853
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_1853(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1853_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1853_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_197
// Description:	Constant
// Input:
// Output:
//	- name: Constant_197_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_197(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_197_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_197_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2632
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2632_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2632(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2632_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2632_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2611
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2611_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2611(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2611_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2611_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2596
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2596_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2596(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2596_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2596_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2485
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2485_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2485(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2485_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2485_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2368
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2368_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2368(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2368_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2368_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[98304];
    bin_file.read(tmp_mem, 98304);
    cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_329_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Convolution_942_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2762_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_944_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2883_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2882_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_955_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_948_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_952_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_955<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_954_0, Constant_329_0, DepthwiseConv2dNative_955_0);
// Add_float_float_float_cuda_Add_2286<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_942_0, Constant_2762_0, BatchNormInference_948_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_25<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_944_0, Constant_2883_0, Convolution_940_0, Constant_2882_0, Add_952_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_955
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_954_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_329_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_955_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_955_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_2286
// Description:	Add
// Input:
//	- name: Convolution_942_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2762_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_948_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2286_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_944_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2883_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_940_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2882_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_952_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2289<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_944_0, Constant_2883_0, BatchNormInference_949_0);
// Add_float_float_float_cuda_Add_2283<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_940_0, Constant_2882_0, BatchNormInference_947_0);
// Add_float_float_float_cuda_Add_952<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_947_0, BatchNormInference_949_0, Add_952_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_25_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_Add_fused_kernel_64(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_955_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 159)
    {
        Add_float_float_float_cuda_Add_2286_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 191)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_25_block_kernel(input4, input5, input7, input6, output2, threadIdx.x, blockIdx.x - 160 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_Add_fused_kernel_64_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_Add_fused_kernel_64<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, output0, output1, output2);
}
// Node name:	Constant_43
// Description:	Constant
// Input:
// Output:
//	- name: Constant_43_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_43(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_43_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_43_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2623
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2623_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2623(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2623_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2623_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	MaxPool_1351
// Description:	MaxPool
// Input:
//	- name: Slice_1347_0	type: float	shape: Shape{1, 128, 16, 16}
// Output:
//	- name: MaxPool_1351_0	type: float	shape: Shape{1, 128, 8, 8}
void MaxPool_float_float_cuda_lib_MaxPool_1351(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 16, 16));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 8, 8));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,3, 3, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_274_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_134_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_191_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1163_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1162_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_274_0, DepthwiseConv2dNative_1161_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1163<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_134_0, DepthwiseConv2dNative_1163_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1158_0, Constant_191_0, DepthwiseConv2dNative_1162_0);
// Slice_float_float_cuda_Slice_1159<<<dim3(256, 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1156_0, Slice_1159_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1163 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161

// Node name:	DepthwiseConv2dNative_1161
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_274_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1161_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1162
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1158_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_191_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1162_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Slice_1159
// Description:	Slice
// Input:
//	- name: BatchNormInference_1156_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Slice_1159_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Slice_float_float_cuda_Slice_1159_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 64){
        return;
    }
    const dim3 blockDim(64, 1, 1);
    const dim3 gridDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 16384)
    {
        uint32_t input_strides[] = {16384, 256, 16, 1};
        uint32_t output_strides[] = {16384, 256, 16, 1};
        uint32_t lower_bounds[] = {0, 0, 0, 0};
        uint32_t slice_strides[] = {1, 1, 1, 1};
        uint32_t input_idx = 0;
        uint32_t output_idx = tid;
        input_idx += (((output_idx / output_strides[0]) * slice_strides[0]) + lower_bounds[0]) * input_strides[0];
        output_idx %= output_strides[0];
        input_idx += (((output_idx / output_strides[1]) * slice_strides[1]) + lower_bounds[1]) * input_strides[1];
        output_idx %= output_strides[1];
        input_idx += (((output_idx / output_strides[2]) * slice_strides[2]) + lower_bounds[2]) * input_strides[2];
        output_idx %= output_strides[2];
        input_idx += (((output_idx / output_strides[3]) * slice_strides[3]) + lower_bounds[3]) * input_strides[3];
        output0[tid] = input0[input_idx];
    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1161_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1162_block_kernel(input0, input3, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 639)
    {
        Slice_float_float_cuda_Slice_1159_block_kernel(input4, output3, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Slice_94<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2095_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3002_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_597_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2098_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3004_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_612_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_610_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2101_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_611_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2104_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_614_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_616_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_618_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_596_0, Constant_2095_0, Constant_3002_0, Relu_613_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3003<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_597_0, Constant_2098_0, Constant_3004_0, Relu_614_0);
// Convolution_float_float_float_cuda_Convolution_620<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_612_0, Constant_2107_0, Convolution_620_0);
// Convolution_float_float_float_cuda_Convolution_616<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_610_0, Constant_2101_0, Convolution_616_0);
// Convolution_float_float_float_cuda_Convolution_618<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_611_0, Constant_2104_0, Convolution_618_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3003 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001
// Convolution_float_float_float_cuda_Convolution_616 : Convolution_float_float_float_cuda_Convolution_620
// Convolution_float_float_float_cuda_Convolution_618 : Convolution_float_float_float_cuda_Convolution_620

// Node name:	Matched_Pattern_3001
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_596_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2095_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3002_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_613_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
           float compute1[2];
          
          
          for (int ff_init = 0; ff_init < 2; ++ff_init) {
            compute1[ff_init] = 0.000000e+00f;
          }
          for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
            __syncthreads();
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[(((((((rc_outer * 16384) + (((int)threadIdx.z) * 2048)) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.y) * 64)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4) * 32)) + (((int)blockIdx.x) * 16)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15))];
            }
            input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 512) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + (rc_outer * 16)) + ((int)threadIdx.x))];
            __syncthreads();
            for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              for (int ff = 0; ff < 2; ++ff) {
                compute1[ff] = (compute1[ff] + (pad_temp_shared[(((rc_inner * 32) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] * input1_shared[(((((int)threadIdx.z) * 32) + (ff * 16)) + rc_inner)]));
              }
            }
          }
          for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
            compute[(((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + (i1_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))] = max((compute1[i1_inner_inner_inner] + input2[(((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + i1_inner_inner_inner)]), 0.000000e+00f);
          }
        }


    }

}
// Node name:	Convolution_620
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_612_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2107_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_620_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_620_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3001_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_620_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Convolution_float_float_float_cuda_Convolution_620_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Convolution_float_float_float_cuda_Convolution_620_block_kernel(input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_14<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	Convolution_1729
// Description:	Convolution
// Input:
//	- name: Relu_1727_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2722_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1729_0	type: float	shape: Shape{1, 128, 8, 8}
extern "C" __global__  void Convolution_float_float_float_cuda_Convolution_1729(float* input0, float* input1, float* output0)
{
    __shared__ float pad_temp_shared[512];
    __shared__ float input1_shared[512];
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2))];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3))];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 2048)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 2048)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 32)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 33)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 4096)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 64)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 65)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 6144)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 6144)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 96)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 97)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 8192)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 128)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 129)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 10240)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 10240)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 160)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 161)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 12288)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 192)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 193)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 14336)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 14336)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 224)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 225)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 16384)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 16384)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 256)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 257)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 18432)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 18432)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 288)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 289)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 20480)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 20480)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 320)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 321)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 22528)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 22528)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 352)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 353)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 24576)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 24576)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 384)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 385)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 26624)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 26624)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 416)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 417)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 28672)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 28672)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 448)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 449)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 30720)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 30720)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 480)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 481)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 32768)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 32768)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 512)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 513)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 34816)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 34816)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 544)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 545)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 36864)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 36864)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 576)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 577)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 38912)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 38912)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 608)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 609)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 40960)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 40960)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 640)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 641)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 43008)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 43008)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 672)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 673)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 45056)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 45056)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 704)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 705)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 1)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 47104)];
          pad_temp_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input0[(((((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) >> 2) * 64)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + 1) >> 2)) & 3) * 8)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 47104)];
          input1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2))] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 736)];
          input1_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 1)] = input1[(((((((int)blockIdx.z) * 12288) + (((int)threadIdx.z) * 768)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + 737)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] * input1_shared[(((int)threadIdx.z) * 32)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16)] * input1_shared[((((int)threadIdx.z) * 32) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32)] * input1_shared[((((int)threadIdx.z) * 32) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48)] * input1_shared[((((int)threadIdx.z) * 32) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64)] * input1_shared[((((int)threadIdx.z) * 32) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80)] * input1_shared[((((int)threadIdx.z) * 32) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96)] * input1_shared[((((int)threadIdx.z) * 32) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112)] * input1_shared[((((int)threadIdx.z) * 32) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128)] * input1_shared[((((int)threadIdx.z) * 32) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144)] * input1_shared[((((int)threadIdx.z) * 32) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160)] * input1_shared[((((int)threadIdx.z) * 32) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176)] * input1_shared[((((int)threadIdx.z) * 32) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192)] * input1_shared[((((int)threadIdx.z) * 32) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208)] * input1_shared[((((int)threadIdx.z) * 32) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224)] * input1_shared[((((int)threadIdx.z) * 32) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240)] * input1_shared[((((int)threadIdx.z) * 32) + 15)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256)] * input1_shared[((((int)threadIdx.z) * 32) + 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272)] * input1_shared[((((int)threadIdx.z) * 32) + 17)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288)] * input1_shared[((((int)threadIdx.z) * 32) + 18)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304)] * input1_shared[((((int)threadIdx.z) * 32) + 19)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320)] * input1_shared[((((int)threadIdx.z) * 32) + 20)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336)] * input1_shared[((((int)threadIdx.z) * 32) + 21)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352)] * input1_shared[((((int)threadIdx.z) * 32) + 22)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368)] * input1_shared[((((int)threadIdx.z) * 32) + 23)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384)] * input1_shared[((((int)threadIdx.z) * 32) + 24)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400)] * input1_shared[((((int)threadIdx.z) * 32) + 25)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416)] * input1_shared[((((int)threadIdx.z) * 32) + 26)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432)] * input1_shared[((((int)threadIdx.z) * 32) + 27)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448)] * input1_shared[((((int)threadIdx.z) * 32) + 28)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464)] * input1_shared[((((int)threadIdx.z) * 32) + 29)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480)] * input1_shared[((((int)threadIdx.z) * 32) + 30)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496)] * input1_shared[((((int)threadIdx.z) * 32) + 31)]));
          compute[((((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 8)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern void Convolution_float_float_float_cuda_Convolution_1729_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Convolution_float_float_float_cuda_Convolution_1729<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
