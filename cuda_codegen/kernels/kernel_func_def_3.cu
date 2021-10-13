// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_329
// Description:	Constant
// Input:
// Output:
//	- name: Constant_329_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_329(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_329_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_329_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2488
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2488_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2488(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2488_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2488_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2802
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2802_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2802(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2802_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2802_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2350
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2350_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2350(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2350_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2350_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2149
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2149_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2149(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2149_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2149_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_167
// Description:	Constant
// Input:
// Output:
//	- name: Constant_167_0	type: float	shape: Shape{5, 5, 64, 1}
void Constant_float_cuda_Constant_167(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_167_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_167_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[6400];
    bin_file.read(tmp_mem, 6400);
    cudaMemcpyAsync(output0, tmp_mem, 6400, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2266
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2266_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2266(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2266_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2266_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2977
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2977_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2977(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2977_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2977_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2281
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2281_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2281(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2281_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2281_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2236
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2236_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2236(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2236_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2236_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2560
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2560_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2560(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2560_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2560_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2053
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2053_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2053(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2053_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2053_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_1092_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1095_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_425_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_291_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_282_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1101_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1099_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1100_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Slice_float_float_cuda_Slice_1094<<<dim3(256, 1, 1), dim3(64, 1, 1), 0, 0>>>(BatchNormInference_1092_0, Slice_1094_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1101<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1095_0, Constant_425_0, DepthwiseConv2dNative_1101_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1099<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1095_0, Constant_291_0, DepthwiseConv2dNative_1099_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1100<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1095_0, Constant_282_0, DepthwiseConv2dNative_1100_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1100 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1101

// Node name:	Slice_1094
// Description:	Slice
// Input:
//	- name: BatchNormInference_1092_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Slice_float_float_cuda_Slice_1094_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1101
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1095_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_425_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1101_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1101_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1099
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1095_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_291_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1099_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1099_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_85(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255)
    {
        Slice_float_float_cuda_Slice_1094_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1101_block_kernel(input1, input2, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1099_block_kernel(input1, input3, output2, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 639)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1101_block_kernel(input1, input4, output3, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_85_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_Slice_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_85<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1589_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2959_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1591_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1525_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1598_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1599_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_63<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1589_0, Constant_2958_0, Slice_1545_0, Add_1598_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_64<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1591_0, Constant_2959_0, BatchNormInference_1525_0, Add_1599_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_64 : FusedKernel_float_float_float_float_cuda_Add_Add_63

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1589_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2958_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1545_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1598_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2646<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1589_0, Constant_2958_0, BatchNormInference_1595_0);
// Add_float_float_float_cuda_Add_1598<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1595_0, Slice_1545_0, Add_1598_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_63_block_kernel(input4, input3, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_156<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2743_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1520_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1522_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2745_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1518_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2951_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1515_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_40_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1516_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_146_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: Add_1532_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1525_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1523_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1524_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_58<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1520_0, Constant_2743_0, Convolution_1522_0, Constant_2745_0, Add_1532_0);
// Add_float_float_float_cuda_Add_2601<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1518_0, Constant_2951_0, BatchNormInference_1525_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1523<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1515_0, Constant_40_0, DepthwiseConv2dNative_1523_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1524<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1516_0, Constant_146_0, DepthwiseConv2dNative_1524_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1520_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2743_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1522_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2745_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1532_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2604<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1520_0, Constant_2743_0, BatchNormInference_1526_0);
// Add_float_float_float_cuda_Add_2607<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1522_0, Constant_2745_0, BatchNormInference_1527_0);
// Add_float_float_float_cuda_Add_1532<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1526_0, BatchNormInference_1527_0, Add_1532_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_58_block_kernel(float* input0, float* input1, float* input2, float* input3, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(input2[tid], input3[tid]);
    float temp2 = add(temp0, temp1);
    output0[tid] = temp2;

}
// Node name:	Add_2601
// Description:	Add
// Input:
//	- name: Convolution_1518_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2951_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1525_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2601_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	DepthwiseConv2dNative_1523
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1515_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_40_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1523_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1523_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 8;
        const int in_width = 8;
        const int in_depth = 128;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 2;
        const int pad_width = 2;
        const int out_height = 8;
        const int out_width = 8;
        const int out_depth = 128;
        const int num_outputs = 8192;

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
// Node name:	DepthwiseConv2dNative_1524
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1516_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_146_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1524_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1524_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

        typedef float S;
        float *input = input0;
        float *filter = input1;
        float *output = output0;

        const int in_height = 8;
        const int in_width = 8;
        const int in_depth = 128;
        const int filter_height = 3;
        const int filter_width = 3;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 1;
        const int pad_width = 1;
        const int out_height = 8;
        const int out_width = 8;
        const int out_depth = 128;
        const int num_outputs = 8192;

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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_145(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_float_cuda_Add_Add_Add_58_block_kernel(input1, input0, input2, input3, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_2601_block_kernel(input4, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1523_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1524_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 96 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_145_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_fused_kernel_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_145<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3);
}
