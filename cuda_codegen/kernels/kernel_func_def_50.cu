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
// Node name:	Constant_3112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3112_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_452
// Description:	Constant
// Input:
// Output:
//	- name: Constant_452_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_452(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_452_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_452_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_374
// Description:	Constant
// Input:
// Output:
//	- name: Constant_374_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_374(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_374_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_374_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2629
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2629_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2629(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2629_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2629_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2954
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2954_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2954(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2954_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2954_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2320
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2320_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2320(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2320_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2320_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_217
// Description:	Constant
// Input:
// Output:
//	- name: Constant_217_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_217(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_217_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_217_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[25088];
    bin_file.read(tmp_mem, 25088);
    cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2863
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2863_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2863(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2863_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2863_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_112
// Description:	Constant
// Input:
// Output:
//	- name: Constant_112_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_112(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_112_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_112_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1057_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_58_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1058_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_93_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1056_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_188_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_1038_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_61_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Constant_253_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: AvgPool_1039_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_967_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: DepthwiseConv2dNative_1062_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1063_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1061_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1046_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1047_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1048_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1057_0, Constant_58_0, DepthwiseConv2dNative_1062_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1063<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1058_0, Constant_93_0, DepthwiseConv2dNative_1063_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1061<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1056_0, Constant_188_0, DepthwiseConv2dNative_1061_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1046<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1038_0, Constant_61_0, DepthwiseConv2dNative_1046_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1047<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1038_0, Constant_253_0, DepthwiseConv2dNative_1047_0);
// Add_float_float_float_cuda_Add_1048<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1039_0, BatchNormInference_967_0, Add_1048_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1063 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1046 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1047 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1061

// Node name:	DepthwiseConv2dNative_1062
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1057_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_58_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1062_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1061
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1056_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_188_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1061_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1061_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_1048
// Description:	Add
// Input:
//	- name: AvgPool_1039_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_967_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1048_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1048_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_78(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 383)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1061_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1062_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 639)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1061_block_kernel(input6, input8, output4, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 640 && (int)blockIdx.x <= 671)
    {
        Add_float_float_float_cuda_Add_1048_block_kernel(input9, input10, output5, threadIdx.x, blockIdx.x - 640 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_78_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_78<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_1032_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1035_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2338_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3070_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1036_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2341_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3072_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1034_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2335_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3068_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1037_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1057_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1058_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1056_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1038_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_1037<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1032_0, AvgPool_1032_0, Add_1037_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1035_0, Constant_2338_0, Constant_3070_0, Relu_1057_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3071<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1036_0, Constant_2341_0, Constant_3072_0, Relu_1058_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3067<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1034_0, Constant_2335_0, Constant_3068_0, Relu_1056_0);
// Relu_float_float_cuda_Relu_1038<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_1033_0, Relu_1038_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3071 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3067 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069

// Node name:	Add_1037
// Description:	Add
// Input:
//	- name: AvgPool_1032_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: AvgPool_1032_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1037_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1037_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Matched_Pattern_3069
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1035_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2338_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3070_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1057_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Relu_1038
// Description:	Relu
// Input:
//	- name: Slice_1033_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1038_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Relu_float_float_cuda_Relu_1038_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_Relu_77(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_1037_block_kernel(input0, input0, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 95)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069_block_kernel(input1, input2, input3, output1, threadIdx.x, blockIdx.x - 32 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 96 && (int)blockIdx.x <= 159)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069_block_kernel(input4, input5, input6, output2, threadIdx.x, blockIdx.x - 96 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 160 && (int)blockIdx.x <= 223)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3069_block_kernel(input7, input8, input9, output3, threadIdx.x, blockIdx.x - 160 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 224 && (int)blockIdx.x <= 255)
    {
        Relu_float_float_cuda_Relu_1038_block_kernel(input10, output4, threadIdx.x, blockIdx.x - 224 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_Relu_77_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Matched_Pattern_Matched_Pattern_Matched_Pattern_Relu_77<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_507_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2047_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2990_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_506_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2044_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2988_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_505_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2041_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2986_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_513_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2062_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_524_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2059_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: DepthwiseConv2dNative_523_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2056_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Slice_500_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_540_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_539_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_508_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_537_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_535_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_507_0, Constant_2047_0, Constant_2990_0, Relu_541_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2987<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_506_0, Constant_2044_0, Constant_2988_0, Relu_540_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2985<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_505_0, Constant_2041_0, Constant_2986_0, Relu_539_0);
// Add_float_float_float_cuda_Add_508<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_499_0, AvgPool_499_0, Add_508_0);
// Convolution_float_float_float_cuda_Convolution_526<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_513_0, Constant_2062_0, Convolution_526_0);
// Convolution_float_float_float_cuda_Convolution_537<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_524_0, Constant_2059_0, Convolution_537_0);
// Convolution_float_float_float_cuda_Convolution_535<<<dim3(2, 16, 2), dim3(16, 2, 8), 0, 0>>>(DepthwiseConv2dNative_523_0, Constant_2056_0, Convolution_535_0);
// Relu_float_float_cuda_Relu_510<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Slice_500_0, Relu_510_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2987 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2985 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989
// Convolution_float_float_float_cuda_Convolution_537 : Convolution_float_float_float_cuda_Convolution_526
// Convolution_float_float_float_cuda_Convolution_535 : Convolution_float_float_float_cuda_Convolution_526

// Node name:	Matched_Pattern_2989
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_507_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2047_0	type: float	shape: Shape{32, 32, 1, 1}
//	- name: Constant_2990_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_541_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_508
// Description:	Add
// Input:
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: AvgPool_499_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_508_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_508_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	Convolution_526
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_513_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2062_0	type: float	shape: Shape{32, 32, 1, 1}
// Output:
//	- name: Convolution_526_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_526_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Relu_510
// Description:	Relu
// Input:
//	- name: Slice_500_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_510_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Relu_float_float_cuda_Relu_510_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Convolution_Convolution_Convolution_Relu_3(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7)
{
    __shared__ char shared_buffer[3072];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_2989_block_kernel(input6, input7, input8, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Add_float_float_float_cuda_Add_508_block_kernel(input9, input9, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Convolution_float_float_float_cuda_Convolution_526_block_kernel(input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 383)
    {
        Convolution_float_float_float_cuda_Convolution_526_block_kernel(input12, input13, output5, threadIdx.x, blockIdx.x - 320 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447)
    {
        Convolution_float_float_float_cuda_Convolution_526_block_kernel(input14, input15, output6, threadIdx.x, blockIdx.x - 384 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 448 && (int)blockIdx.x <= 511)
    {
        Relu_float_float_cuda_Relu_510_block_kernel(input16, output7, threadIdx.x, blockIdx.x - 448 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Convolution_Convolution_Convolution_Relu_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Matched_Pattern_Add_Convolution_Convolution_Convolution_Relu_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, output0, output1, output2, output3, output4, output5, output6, output7);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1635_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_290_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1633_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_92_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1634_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_272_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: AvgPool_1615_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1616_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_386_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_345_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1640_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1638_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1639_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1623_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1624_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1625_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1635_0, Constant_290_0, DepthwiseConv2dNative_1640_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1633_0, Constant_92_0, DepthwiseConv2dNative_1638_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1639<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1634_0, Constant_272_0, DepthwiseConv2dNative_1639_0);
// Add_float_float_float_cuda_Add_1623<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1615_0, BatchNormInference_1544_0, Add_1623_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1624<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1616_0, Constant_386_0, DepthwiseConv2dNative_1624_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1625<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1616_0, Constant_345_0, DepthwiseConv2dNative_1625_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1639 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1624 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1625 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640

// Node name:	DepthwiseConv2dNative_1640
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1635_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_290_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1640_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1638
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1633_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_92_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1638_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_1623
// Description:	Add
// Input:
//	- name: AvgPool_1615_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1544_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1623_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1623_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 207)
    {
        Add_float_float_float_cuda_Add_1623_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 192 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 208 && (int)blockIdx.x <= 271)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1638_block_kernel(input8, input9, output4, threadIdx.x, blockIdx.x - 208 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 272 && (int)blockIdx.x <= 335)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1640_block_kernel(input8, input10, output5, threadIdx.x, blockIdx.x - 272 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_161<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: DepthwiseConv2dNative_1229_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2452_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3104_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1230_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2455_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3106_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1254_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2464_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1252_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2458_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: DepthwiseConv2dNative_1253_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2461_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1251_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1258_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1260_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1229_0, Constant_2452_0, Constant_3104_0, Relu_1250_0);
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3105<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1230_0, Constant_2455_0, Constant_3106_0, Relu_1251_0);
// Convolution_float_float_float_cuda_Convolution_1262<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1254_0, Constant_2464_0, Convolution_1262_0);
// Convolution_float_float_float_cuda_Convolution_1258<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1252_0, Constant_2458_0, Convolution_1258_0);
// Convolution_float_float_float_cuda_Convolution_1260<<<dim3(2, 8, 4), dim3(8, 2, 16), 0, 0>>>(DepthwiseConv2dNative_1253_0, Constant_2461_0, Convolution_1260_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3105 : Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103
// Convolution_float_float_float_cuda_Convolution_1258 : Convolution_float_float_float_cuda_Convolution_1262
// Convolution_float_float_float_cuda_Convolution_1260 : Convolution_float_float_float_cuda_Convolution_1262

// Node name:	Matched_Pattern_3103
// Description:	Matched_Pattern
// Input:
//	- name: DepthwiseConv2dNative_1229_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2452_0	type: float	shape: Shape{64, 64, 1, 1}
//	- name: Constant_3104_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1250_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Convolution_1262
// Description:	Convolution
// Input:
//	- name: DepthwiseConv2dNative_1254_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2464_0	type: float	shape: Shape{64, 64, 1, 1}
// Output:
//	- name: Convolution_1262_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Convolution_float_float_float_cuda_Convolution_1262_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
           float compute_local[1];
          
          
          compute_local[0] = 0.000000e+00f;
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[(((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 4096)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 16)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 8192)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 32)];
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
          pad_temp_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input0[((((((((int)threadIdx.z) * 256) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 12288)];
          input1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = input1[(((((((int)blockIdx.z) * 1024) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) + 48)];
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
          compute[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = compute_local[0];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4)
{
    __shared__ char shared_buffer[2048];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        Matched_Pattern_float_float_float_float_cuda_Matched_Pattern_3103_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 64 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        Convolution_float_float_float_cuda_Convolution_1262_block_kernel(input6, input7, output2, threadIdx.x, blockIdx.x - 128 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        Convolution_float_float_float_cuda_Convolution_1262_block_kernel(input8, input9, output3, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        Convolution_float_float_float_cuda_Convolution_1262_block_kernel(input10, input11, output4, threadIdx.x, blockIdx.x - 256 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* output0, float* output1, float* output2, float* output3, float* output4) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Matched_Pattern_Matched_Pattern_Convolution_Convolution_Convolution_106<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, output0, output1, output2, output3, output4);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_638_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2835_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2836_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Convolution_640_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_641_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_642_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_644_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2118<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_638_0, Constant_2835_0, BatchNormInference_641_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_11<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_640_0, Constant_2836_0, Relu_644_0, BatchNormInference_642_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2118
// Description:	Add
// Input:
//	- name: Convolution_638_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2835_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: BatchNormInference_641_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2118_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_640_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_2836_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Relu_644_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_642_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// Add_float_float_float_cuda_Add_2121<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_640_0, Constant_2836_0, BatchNormInference_642_0);
// Relu_float_float_cuda_Relu_644<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_642_0, Relu_644_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_11_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2118_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_11_block_kernel(input3, input2, output2, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_19<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
