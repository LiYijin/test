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
// Node name:	Constant_274
// Description:	Constant
// Input:
// Output:
//	- name: Constant_274_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_274(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_274_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_274_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2344
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2344_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2344(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2344_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2344_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2464
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2464_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2464(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2464_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2464_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2782
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2782_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2782(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2782_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2782_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_241
// Description:	Constant
// Input:
// Output:
//	- name: Constant_241_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_241(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_241_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_241_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2650
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2650_0	type: float	shape: Shape{128, 768, 1, 1}
void Constant_float_cuda_Constant_2650(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2650_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2650_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[393216];
    bin_file.read(tmp_mem, 393216);
    cudaMemcpyAsync(output0, tmp_mem, 393216, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_857_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_96_0	type: float	shape: Shape{7, 7, 64, 1}
//	- name: Relu_855_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_153_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_856_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_316_0	type: float	shape: Shape{7, 7, 64, 1}
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: DepthwiseConv2dNative_862_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_860_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_861_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_838_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_862<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_857_0, Constant_96_0, DepthwiseConv2dNative_862_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_860<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_855_0, Constant_153_0, DepthwiseConv2dNative_860_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_861<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_856_0, Constant_316_0, DepthwiseConv2dNative_861_0);
// Relu_float_float_cuda_Relu_838<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_833_0, Relu_838_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_861 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_862

// Node name:	DepthwiseConv2dNative_862
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_857_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_96_0	type: float	shape: Shape{7, 7, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_862_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_862_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
        const int filter_height = 7;
        const int filter_width = 7;
        const int depth_multiplier = 1;
        const int stride = 1;
        const int pad_height = 3;
        const int pad_width = 3;
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
// Node name:	DepthwiseConv2dNative_860
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_855_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_153_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_860_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_860_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Relu_838
// Description:	Relu
// Input:
//	- name: Slice_833_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_838_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Relu_float_float_cuda_Relu_838_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_48(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_862_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_860_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_862_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447)
    {
        Relu_float_float_cuda_Relu_838_block_kernel(input6, output3, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_48_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Relu_48<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, output0, output1, output2, output3);
}
// Node name:	Constant_3146
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3146_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2218
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2218_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2218(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2218_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2218_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2146
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2146_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2146(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2146_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2146_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3108
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3108_0	type: float	shape: Shape{1, 128, 16, 16}
void Constant_float_cuda_Constant_3108(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3108_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3108_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_224
// Description:	Constant
// Input:
// Output:
//	- name: Constant_224_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_224(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_224_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_224_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	AvgPool_1337
// Description:	AvgPool
// Input:
//	- name: Relu_1336_0	type: float	shape: Shape{1, 384, 16, 16}
// Output:
//	- name: AvgPool_1337_0	type: float	shape: Shape{1, 384, 8, 8}
void AvgPool_float_float_cuda_lib_AvgPool_1337(cudnnHandle_t cudnn_handle, float* input0, float* output0)
{
    cudnnTensorDescriptor_t input_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 384, 16, 16));
    cudnnTensorDescriptor_t output_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 384, 8, 8));
    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,1, 1, 0, 0, 2, 2));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle, desc, &alpha, input_desc, input0, &beta, output_desc, output0));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));

}
// Node name:	 BlockFusion
// Input:
//	- name: Slice_643_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_645_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_649_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2125_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3008_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_650_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2128_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3010_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_648_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2122_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3006_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_651_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_671_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_672_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_670_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Relu_float_float_cuda_Relu_647<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_643_0, Relu_647_0);
// Add_float_float_float_cuda_Add_651<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_645_0, AvgPool_645_0, Add_651_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_649_0, Constant_2125_0, Constant_3008_0, Relu_671_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3009<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_650_0, Constant_2128_0, Constant_3010_0, Relu_672_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3005<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_648_0, Constant_2122_0, Constant_3006_0, Relu_670_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3009 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3005 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007

// Node name:	Relu_647
// Description:	Relu
// Input:
//	- name: Slice_643_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_647_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Relu_float_float_cuda_Relu_647_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Add_651
// Description:	Add
// Input:
//	- name: AvgPool_645_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_645_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_651_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_651_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Matched_Pattern_3007
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_649_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2125_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_3008_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_671_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_21(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Relu_float_float_cuda_Relu_647_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Add_float_float_float_cuda_Add_651_block_kernel(input1, input1, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007_block_kernel(input2, input3, input4, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007_block_kernel(input5, input6, input7, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3007_block_kernel(input8, input9, input10, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_21_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Relu_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_21<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1343_0	type: float	shape: Shape{1, 64, 8, 8}
//	- name: Convolution_1349_0	type: float	shape: Shape{1, 64, 8, 8}
//	- name: Relu_1350_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Constant_183_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: MaxPool_1351_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1325_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: AvgPool_1352_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1326_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Concat_1353_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1354_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1355_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1356_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Concat_float_float_float_cuda_Concat_1353<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1343_0, Convolution_1349_0, Concat_1353_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1354<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1350_0, Constant_183_0, DepthwiseConv2dNative_1354_0);
// Add_float_float_float_cuda_Add_1355<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(MaxPool_1351_0, BatchNormInference_1325_0, Add_1355_0);
// Add_float_float_float_cuda_Add_1356<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1352_0, BatchNormInference_1326_0, Add_1356_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Add_float_float_float_cuda_Add_1356 : Add_float_float_float_cuda_Add_1355

// Node name:	Concat_1353
// Description:	Concat
// Input:
//	- name: Convolution_1343_0	type: float	shape: Shape{1, 64, 8, 8}
//	- name: Convolution_1349_0	type: float	shape: Shape{1, 64, 8, 8}
// Output:
//	- name: Concat_1353_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Concat_float_float_float_cuda_Concat_1353_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t inputs_strides[] = {4096, 4096};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 8192)
    {
        uint32_t block_id = tid / 8192;
        uint32_t block_idx = tid % 8192;
        uint32_t output_idx = block_id * 8192 + block_idx;
        if(block_idx < inputs_strides[0])
        {
            output0[output_idx] = input0[block_id * inputs_strides[0] + block_idx];
            return;
        }
        block_idx -= inputs_strides[0];
        if(block_idx < inputs_strides[1])
        {
            output0[output_idx] = input1[block_id * inputs_strides[1] + block_idx];
            return;
        }
        block_idx -= inputs_strides[1];
    }

}
// Node name:	DepthwiseConv2dNative_1354
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1350_0	type: float	shape: Shape{1, 128, 16, 16}
//	- name: Constant_183_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1354_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1354_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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

        const int in_height = 16;
        const int in_width = 16;
        const int in_depth = 128;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 2;
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
// Node name:	Add_1355
// Description:	Add
// Input:
//	- name: MaxPool_1351_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1325_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1355_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1355_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Concat_DepthwiseConv2dNative_Add_Add_122(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        Concat_float_float_float_cuda_Concat_1353_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1354_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 80 && (int)blockIdx.x <= 95)
    {
        Add_float_float_float_cuda_Add_1355_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 80 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 111)
    {
        Add_float_float_float_cuda_Add_1355_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 96 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Concat_DepthwiseConv2dNative_Add_Add_122_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Concat_DepthwiseConv2dNative_Add_Add_122<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, output0, output1, output2, output3);
}
