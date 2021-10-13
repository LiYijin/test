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
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_646_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_332_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Constant_123_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_671_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_452_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_672_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_88_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_670_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_56_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: Add_652_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_654_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_676_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_677_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_675_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_652<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_646_0, BatchNormInference_578_0, Add_652_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_647_0, Constant_332_0, DepthwiseConv2dNative_653_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_647_0, Constant_123_0, DepthwiseConv2dNative_654_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_676<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_671_0, Constant_452_0, DepthwiseConv2dNative_676_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_677<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_672_0, Constant_88_0, DepthwiseConv2dNative_677_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_675<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_670_0, Constant_56_0, DepthwiseConv2dNative_675_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_676 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_677 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_675 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653

// Node name:	Add_652
// Description:	Add
// Input:
//	- name: AvgPool_646_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_578_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_652_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_652_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	DepthwiseConv2dNative_653
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_332_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_653_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_654
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_123_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_654_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_652_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 319)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 575)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(input2, input4, output2, threadIdx.x, blockIdx.x - 320 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 831)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(input5, input6, output3, threadIdx.x, blockIdx.x - 576 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 832 && (int)blockIdx.x <= 1087)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_654_block_kernel(input7, input8, output4, threadIdx.x, blockIdx.x - 832 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 1088 && (int)blockIdx.x <= 1343)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_653_block_kernel(input9, input10, output5, threadIdx.x, blockIdx.x - 1088 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_22<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	Constant_2407
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2407_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2407(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2407_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2407_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2404
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2404_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2404(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2404_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2404_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[98304];
    bin_file.read(tmp_mem, 98304);
    cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_169
// Description:	Constant
// Input:
// Output:
//	- name: Constant_169_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_169(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_169_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_169_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3180
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3180_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3180(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3180_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3180_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2437
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2437_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_2437(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2437_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2437_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[98304];
    bin_file.read(tmp_mem, 98304);
    cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2855
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2855_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2855(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2855_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2855_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2296
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2296_0	type: float	shape: Shape{64, 256, 1, 1}
void Constant_float_cuda_Constant_2296(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2296_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2296_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3048
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3048_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3048(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3048_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3048_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3148
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3148_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3148(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3148_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3148_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2158
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2158_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2158(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2158_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2158_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_961_0	type: float	shape: Shape{1, 256, 16, 16}
//	- name: Constant_2293_0	type: float	shape: Shape{64, 256, 1, 1}
//	- name: Constant_2296_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_965_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_963<<<dim3(1, 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_961_0, Constant_2293_0, Convolution_963_0);
// Convolution_float_float_float_cuda_Convolution_965<<<dim3(1, 16, 4), dim3(16, 1, 16), 0, 0>>>(Relu_961_0, Constant_2296_0, Convolution_965_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_965 : Convolution_float_float_float_cuda_Convolution_963

// Node name:	Convolution_963
// Description:	Convolution
// Input:
//	- name: Relu_961_0	type: float	shape: Shape{1, 256, 16, 16}
//	- name: Constant_2293_0	type: float	shape: Shape{64, 256, 1, 1}
// Output:
//	- name: Convolution_963_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_963_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(16, 1, 16);
    const dim3 gridDim(1, 16, 4);
    const dim3 threadIdx(thread_id % 16, 0, thread_id / 16);
    const dim3 blockIdx(block_id % 1, block_id / 1 % 16, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 1024);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[(((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x))];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[(((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x))];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 4096)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 16)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 8192)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 32)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 12288)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 48)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 16384)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 64)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 20480)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 80)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 24576)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 96)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 28672)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 112)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 32768)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 128)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 36864)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 144)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 40960)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 160)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 45056)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 176)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 49152)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 192)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 53248)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 208)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 57344)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 224)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          __syncthreads();
          pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input0[((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x)) + 61440)];
          input1_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((int)threadIdx.x)) + 240)];
          __syncthreads();
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * input1_shared[(((int)threadIdx.z) * 16)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 16)] * input1_shared[((((int)threadIdx.z) * 16) + 1)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 32)] * input1_shared[((((int)threadIdx.z) * 16) + 2)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 48)] * input1_shared[((((int)threadIdx.z) * 16) + 3)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 64)] * input1_shared[((((int)threadIdx.z) * 16) + 4)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 80)] * input1_shared[((((int)threadIdx.z) * 16) + 5)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 96)] * input1_shared[((((int)threadIdx.z) * 16) + 6)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 112)] * input1_shared[((((int)threadIdx.z) * 16) + 7)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 128)] * input1_shared[((((int)threadIdx.z) * 16) + 8)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 144)] * input1_shared[((((int)threadIdx.z) * 16) + 9)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 160)] * input1_shared[((((int)threadIdx.z) * 16) + 10)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 176)] * input1_shared[((((int)threadIdx.z) * 16) + 11)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 192)] * input1_shared[((((int)threadIdx.z) * 16) + 12)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 208)] * input1_shared[((((int)threadIdx.z) * 16) + 13)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 224)] * input1_shared[((((int)threadIdx.z) * 16) + 14)]));
          compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 240)] * input1_shared[((((int)threadIdx.z) * 16) + 15)]));
          compute[((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_963_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Convolution_float_float_float_cuda_Convolution_963_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_65<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1475_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2578_0	type: float	shape: Shape{128, 768, 1, 1}
//	- name: Constant_2581_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1477_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1479_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Convolution_float_float_float_cuda_Convolution_1477<<<dim3(2, 2, 8), dim3(4, 4, 16), 0, 0>>>(Relu_1475_0, Constant_2578_0, Convolution_1477_0);
// Convolution_float_float_float_cuda_Convolution_1479<<<dim3(2, 2, 8), dim3(4, 4, 16), 0, 0>>>(Relu_1475_0, Constant_2581_0, Convolution_1479_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Convolution_float_float_float_cuda_Convolution_1479 : Convolution_float_float_float_cuda_Convolution_1477

// Node name:	Convolution_1477
// Description:	Convolution
// Input:
//	- name: Relu_1475_0	type: float	shape: Shape{1, 768, 8, 8}
//	- name: Constant_2578_0	type: float	shape: Shape{128, 768, 1, 1}
// Output:
//	- name: Convolution_1477_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1477_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(4, 4, 16);
    const dim3 gridDim(2, 2, 8);
    const dim3 threadIdx(thread_id % 4, thread_id / 4 % 4, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 2, block_id / 4);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 2048);
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_139(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[4096];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Convolution_float_float_float_cuda_Convolution_1477_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Convolution_float_float_float_cuda_Convolution_1477_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_139_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Convolution_Convolution_139<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_895_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: AvgPool_896_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16, 16}
// Fused functions:
// Relu_float_float_cuda_Relu_899<<<dim3(128, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_895_0, Relu_899_0);
// Convolution_float_float_float_cuda_Convolution_901<<<dim3(2, 8, 4), dim3(8, 2, 8), 0, 0>>>(AvgPool_896_0, Constant_1853_0, Convolution_901_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Relu_899
// Description:	Relu
// Input:
//	- name: Slice_895_0	type: float	shape: Shape{1, 64, 32, 32}
// Output:
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
__device__ __noinline__ void Relu_float_float_cuda_Relu_899_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(128, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Convolution_901
// Description:	Convolution
// Input:
//	- name: AvgPool_896_0	type: float	shape: Shape{1, 192, 16, 16}
//	- name: Constant_1853_0	type: float	shape: Shape{32, 192, 1, 1}
// Output:
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_901_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(8, 2, 8);
    const dim3 gridDim(2, 8, 4);
    const dim3 threadIdx(thread_id % 8, thread_id / 8 % 2, thread_id / 16);
    const dim3 blockIdx(block_id % 2, block_id / 2 % 8, block_id / 16);
    float* pad_temp_shared = (float*)(shared_buffer + 0);
    float* input1_shared = (float*)(shared_buffer + 3072);
    {
        float* compute = output0;{
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          #pragma unroll
          for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = input0[((((((((rc_outer * 12288) + (((int)threadIdx.z) * 1536)) + (((int)threadIdx.y) * 768)) + ((((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4) * 256)) + (((int)blockIdx.y) * 32)) + (((((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 15) >> 3) * 16)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7))];
            }
            #pragma unroll
            for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
              input1_shared[((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] = input1[((((((((int)blockIdx.z) * 1536) + (((int)threadIdx.z) * 192)) + (rc_outer * 48)) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
            }
            __syncthreads();
            #pragma unroll
            for (int rc_inner = 0; rc_inner < 48; ++rc_inner) {
              compute_local[0] = (compute_local[0] + (pad_temp_shared[(((rc_inner * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] * input1_shared[((((int)threadIdx.z) * 48) + rc_inner)]));
            }
          }
          compute[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56(float* input0, float* input1, float* input2, float* output0, float* output1)
{
    __shared__ char shared_buffer[4608];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        Relu_float_float_cuda_Relu_899_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_901_block_kernel(input1, input2, output1, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Relu_Convolution_56<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2788_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2863_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_879_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_882_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_883_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_23<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_877_0, Constant_2788_0, BatchNormInference_817_0, Add_882_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_24<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_879_0, Constant_2863_0, Slice_833_0, Add_883_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_24 : FusedKernel_float_float_float_float_cuda_Add_Add_23

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_877_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2788_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_817_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_882_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2256<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_877_0, Constant_2788_0, BatchNormInference_880_0);
// Add_float_float_float_cuda_Add_882<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_880_0, BatchNormInference_817_0, Add_882_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(temp0, input2[tid]);
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_23_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_53<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
