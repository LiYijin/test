// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_414
// Description:	Constant
// Input:
// Output:
//	- name: Constant_414_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_414(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_414_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_414_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2536
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2536_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2536(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2536_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2536_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2946
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2946_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2946(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2946_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2946_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2308
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2308_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2308(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2308_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2308_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2026
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2026_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2026(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2026_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2026_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2212
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2212_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2212(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2212_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2212_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2035
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2035_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2035(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2035_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2035_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2422
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2422_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2422(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2422_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2422_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3126
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3126_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3126(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3126_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3126_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3170
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3170_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3170(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3170_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3170_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3036
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3036_0	type: float	shape: Shape{1, 64, 32, 32}
void Constant_float_cuda_Constant_3036(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3036_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3036_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[262144];
    bin_file.read(tmp_mem, 262144);
    cudaMemcpyAsync(output0, tmp_mem, 262144, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2047
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2047_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2047(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2047_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2047_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_437
// Description:	Constant
// Input:
// Output:
//	- name: Constant_437_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_437(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_437_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_437_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_187_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Constant_367_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: AvgPool_1098_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1123_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_363_0	type: float	shape: Shape{3, 3, 64, 1}
//	- name: Relu_1121_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_168_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Relu_1122_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_107_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1103_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1104_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1105_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1128_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1126_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_1127_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1097_0, Constant_187_0, DepthwiseConv2dNative_1103_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1097_0, Constant_367_0, DepthwiseConv2dNative_1104_0);
// Add_float_float_float_cuda_Add_1105<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1098_0, BatchNormInference_1029_0, Add_1105_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1128<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1123_0, Constant_363_0, DepthwiseConv2dNative_1128_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1126<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1121_0, Constant_168_0, DepthwiseConv2dNative_1126_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1127<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1122_0, Constant_107_0, DepthwiseConv2dNative_1127_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1128 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1126 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1127 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104

// Node name:	DepthwiseConv2dNative_1103
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_187_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1103_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1104
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1097_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_367_0	type: float	shape: Shape{3, 3, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_1104_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_1105
// Description:	Add
// Input:
//	- name: AvgPool_1098_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1105_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1105_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 287)
    {
        Add_float_float_float_cuda_Add_1105_block_kernel(input3, input4, output2, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 288 && (int)blockIdx.x <= 415)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(input5, input6, output3, threadIdx.x, blockIdx.x - 288 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 416 && (int)blockIdx.x <= 543)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1103_block_kernel(input7, input8, output4, threadIdx.x, blockIdx.x - 416 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 544 && (int)blockIdx.x <= 671)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1104_block_kernel(input9, input10, output5, threadIdx.x, blockIdx.x - 544 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_87<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	Add_2724
// Description:	Add
// Input:
//	- name: Convolution_1729_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2877_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: BatchNormInference_1730_0	type: float	shape: Shape{1, 128, 8, 8}
extern "C" __launch_bounds__(512) __global__ void Add_float_float_float_cuda_Add_2724(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern void Add_float_float_float_cuda_Add_2724_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Add_float_float_float_cuda_Add_2724<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
