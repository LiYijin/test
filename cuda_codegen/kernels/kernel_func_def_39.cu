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
// Node name:	Constant_2365
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2365_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2365(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2365_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2365_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[98304];
    bin_file.read(tmp_mem, 98304);
    cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_268
// Description:	Constant
// Input:
// Output:
//	- name: Constant_268_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_268(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_268_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_268_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2014
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2014_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2014(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2014_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2014_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_227
// Description:	Constant
// Input:
// Output:
//	- name: Constant_227_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_227(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_227_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_227_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2563
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2563_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2563(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2563_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2563_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_39
// Description:	Constant
// Input:
// Output:
//	- name: Constant_39_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_39(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_39_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_39_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2356
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2356_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2356(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2356_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2356_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2659
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2659_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2659(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2659_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2659_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_386
// Description:	Constant
// Input:
// Output:
//	- name: Constant_386_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_386(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_386_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_386_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2847
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2847_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2847(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2847_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2847_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2871
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2871_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2871(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2871_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2871_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	AvgPool_1361
// Description:	AvgPool
// Input:
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: AvgPool_1361_0	type: float	shape: Shape{1, 128, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1361(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 8, 8));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 8, 8));
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
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2952_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2954_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1543_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1546_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2616<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1540_0, Constant_2952_0, BatchNormInference_1543_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_61<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1542_0, Constant_2954_0, Relu_1546_0, BatchNormInference_1544_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2616
// Description:	Add
// Input:
//	- name: Convolution_1540_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2952_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1543_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2616_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1542_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2954_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1546_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2619<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1542_0, Constant_2954_0, BatchNormInference_1544_0);
// Relu_float_float_cuda_Relu_1546<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1544_0, Relu_1546_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_61_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        Add_float_float_float_cuda_Add_2616_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_61_block_kernel(input3, input2, output2, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_149<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	Constant_3104
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3104_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3104(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3104_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3104_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_673_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_105_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_674_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_254_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Convolution_685_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2784_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2842_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_681_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2843_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_683_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_678_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_679_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_692_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_695_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_678<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_673_0, Constant_105_0, DepthwiseConv2dNative_678_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_679<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_674_0, Constant_254_0, DepthwiseConv2dNative_679_0);
// Add_float_float_float_cuda_Add_2145<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_685_0, Constant_2784_0, BatchNormInference_692_0);
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_12<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_681_0, Constant_2842_0, Convolution_683_0, Constant_2843_0, Add_695_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	DepthwiseConv2dNative_678
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_673_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_105_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_678_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_678_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_679
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_674_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_254_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_679_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_679_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_2145
// Description:	Add
// Input:
//	- name: Convolution_685_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2784_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_692_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2145_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_681_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2842_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_683_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2843_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_695_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2139<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_681_0, Constant_2842_0, BatchNormInference_690_0);
// Add_float_float_float_cuda_Add_2142<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_683_0, Constant_2843_0, BatchNormInference_691_0);
// Add_float_float_float_cuda_Add_695<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_690_0, BatchNormInference_691_0, Add_695_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_12_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
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
    float temp2 = add(temp0, temp1);
    output0[tid] = temp2;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_24(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_678_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_679_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575)
    {
        Add_float_float_float_cuda_Add_2145_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 639)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_12_block_kernel(input7, input6, input9, input8, output3, threadIdx.x, blockIdx.x - 576 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_24_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_fused_kernel_24<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3);
}
