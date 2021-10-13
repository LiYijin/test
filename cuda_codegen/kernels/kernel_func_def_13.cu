// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_119
// Description:	Constant
// Input:
// Output:
//	- name: Constant_119_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_119(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_119_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_119_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_76
// Description:	Constant
// Input:
// Output:
//	- name: Constant_76_0	type: float	shape: Shape{64}
void Constant_float_cuda_Constant_76(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_76_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_76_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[256];
    bin_file.read(tmp_mem, 256);
    cudaMemcpyAsync(output0, tmp_mem, 256, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_264
// Description:	Constant
// Input:
// Output:
//	- name: Constant_264_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_264(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_264_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_264_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2584
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2584_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2584(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2584_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2584_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2602
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2602_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2602(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2602_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2602_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2919
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2919_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2919(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2919_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2919_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3120
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3120_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3120(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3120_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3120_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2116
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2116_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2116(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2116_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2116_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3014
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3014_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3014(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3014_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3014_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2194
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2194_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2194(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2194_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2194_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_182
// Description:	Constant
// Input:
// Output:
//	- name: Constant_182_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_182(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_182_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_182_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_425
// Description:	Constant
// Input:
// Output:
//	- name: Constant_425_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_425(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_425_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_425_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2970
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2970_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2970(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2970_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2970_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: AvgPool_897_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_873_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: MaxPool_898_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_874_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_239_0	type: float	shape: Shape{5, 5, 64, 1}
//	- name: Convolution_893_0	type: float	shape: Shape{1, 32, 16, 16}
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16, 16}
// Output:
//	- name: Add_902_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_903_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: DepthwiseConv2dNative_904_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Concat_905_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_902<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_897_0, BatchNormInference_873_0, Add_902_0);
// Add_float_float_float_cuda_Add_903<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(MaxPool_898_0, BatchNormInference_874_0, Add_903_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_904<<<dim3(128, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_899_0, Constant_239_0, DepthwiseConv2dNative_904_0);
// Concat_float_float_float_cuda_Concat_905<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_893_0, Convolution_901_0, Concat_905_0);
// Deduped function map: <src_function_name : deduped_function_name>
// Add_float_float_float_cuda_Add_903 : Add_float_float_float_cuda_Add_902

// Node name:	Add_902
// Description:	Add
// Input:
//	- name: AvgPool_897_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_873_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_902_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_902_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	DepthwiseConv2dNative_904
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_899_0	type: float	shape: Shape{1, 64, 32, 32}
//	- name: Constant_239_0	type: float	shape: Shape{5, 5, 64, 1}
// Output:
//	- name: DepthwiseConv2dNative_904_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_904_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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

        const int in_height = 32;
        const int in_width = 32;
        const int in_depth = 64;
        const int filter_height = 5;
        const int filter_width = 5;
        const int depth_multiplier = 1;
        const int stride = 2;
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
// Node name:	Concat_905
// Description:	Concat
// Input:
//	- name: Convolution_893_0	type: float	shape: Shape{1, 32, 16, 16}
//	- name: Convolution_901_0	type: float	shape: Shape{1, 32, 16, 16}
// Output:
//	- name: Concat_905_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Concat_float_float_float_cuda_Concat_905_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t inputs_strides[] = {8192, 8192};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 16384)
    {
        uint32_t block_id = tid / 16384;
        uint32_t block_idx = tid % 16384;
        uint32_t output_idx = block_id * 16384 + block_idx;
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_DepthwiseConv2dNative_Concat_57(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2, float* output3)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_902_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_902_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 191)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_904_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 223)
    {
        Concat_float_float_float_cuda_Concat_905_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 192 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_DepthwiseConv2dNative_Concat_57_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Add_Add_DepthwiseConv2dNative_Concat_57<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, output0, output1, output2, output3);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_772_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_172_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Constant_114_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: AvgPool_773_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_704_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Relu_796_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_335_0	type: float	shape: Shape{5, 5, 32, 1}
//	- name: Relu_798_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_394_0	type: float	shape: Shape{3, 3, 32, 1}
//	- name: Relu_797_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_439_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_778_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_779_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Add_780_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_801_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_803_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: DepthwiseConv2dNative_802_0	type: float	shape: Shape{1, 32, 32, 32}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_772_0, Constant_172_0, DepthwiseConv2dNative_778_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_779<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_772_0, Constant_114_0, DepthwiseConv2dNative_779_0);
// Add_float_float_float_cuda_Add_780<<<dim3(64, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_773_0, BatchNormInference_704_0, Add_780_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_801<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_796_0, Constant_335_0, DepthwiseConv2dNative_801_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_803<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_798_0, Constant_394_0, DepthwiseConv2dNative_803_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_802<<<dim3(256, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_797_0, Constant_439_0, DepthwiseConv2dNative_802_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_801 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_779
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_803 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_802 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778

// Node name:	DepthwiseConv2dNative_778
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_772_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_172_0	type: float	shape: Shape{3, 3, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_778_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_779
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_772_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: Constant_114_0	type: float	shape: Shape{5, 5, 32, 1}
// Output:
//	- name: DepthwiseConv2dNative_779_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_779_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_780
// Description:	Add
// Input:
//	- name: AvgPool_773_0	type: float	shape: Shape{1, 32, 32, 32}
//	- name: BatchNormInference_704_0	type: float	shape: Shape{1, 32, 32, 32}
// Output:
//	- name: Add_780_0	type: float	shape: Shape{1, 32, 32, 32}
__device__ __noinline__ void Add_float_float_float_cuda_Add_780_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_40(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 511)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_779_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 512 && (int)blockIdx.x <= 575)
    {
        Add_float_float_float_cuda_Add_780_block_kernel(input3, input4, output2, threadIdx.x, blockIdx.x - 512 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 831)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_779_block_kernel(input5, input6, output3, threadIdx.x, blockIdx.x - 576 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 832 && (int)blockIdx.x <= 1087)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778_block_kernel(input7, input8, output4, threadIdx.x, blockIdx.x - 832 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 1088 && (int)blockIdx.x <= 1343)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_778_block_kernel(input9, input10, output5, threadIdx.x, blockIdx.x - 1088 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_40_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_40<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
