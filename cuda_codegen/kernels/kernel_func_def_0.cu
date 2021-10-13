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
// Node name:	Constant_245
// Description:	Constant
// Input:
// Output:
//	- name: Constant_245_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_245(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_245_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_245_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_161
// Description:	Constant
// Input:
// Output:
//	- name: Constant_161_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_161(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_161_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_161_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2010
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2010_0	type: float	shape: Shape{1, 10}
void Constant_float_cuda_Constant_2010(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2010_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2010_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[40];
    bin_file.read(tmp_mem, 40);
    cudaMemcpyAsync(output0, tmp_mem, 40, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2608
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2608_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2608(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2608_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2608_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2626
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2626_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2626(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2626_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2626_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: float	shape: Shape{7, 7, 128, 1}
void Constant_float_cuda_Constant_46(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[25088];
    bin_file.read(tmp_mem, 25088);
    cudaMemcpyAsync(output0, tmp_mem, 25088, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2410
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2410_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2410(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2410_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2410_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2038
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2038_0	type: float	shape: Shape{32, 96, 1, 1}
void Constant_float_cuda_Constant_2038(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2038_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2038_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12288];
    bin_file.read(tmp_mem, 12288);
    cudaMemcpyAsync(output0, tmp_mem, 12288, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2605
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2605_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2605(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2605_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2605_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2302
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2302_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2302(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2302_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2302_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3066
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3066_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_3066(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3066_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3066_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2287
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2287_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2287(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2287_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2287_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2820_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2822_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Relu_32<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1026_0, Constant_2820_0, Relu_1031_0, BatchNormInference_1029_0);
// Add_float_float_float_cuda_Add_2334<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1028_0, Constant_2822_0, BatchNormInference_1030_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1026_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2820_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1031_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1029_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2331<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1026_0, Constant_2820_0, BatchNormInference_1029_0);
// Relu_float_float_cuda_Relu_1031<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1029_0, Relu_1031_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_32_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_2334
// Description:	Add
// Input:
//	- name: Convolution_1028_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2822_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1030_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2334_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_32_block_kernel(input1, input0, output1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        Add_float_float_float_cuda_Add_2334_block_kernel(input2, input3, output2, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_fused_kernel_Add_75<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
// Node name:	 BlockFusion
// Input:
//	- name: Relu_1446_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_112_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: Relu_1445_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_201_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1444_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_324_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Relu_1426_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_109_0	type: float	shape: Shape{3, 3, 128, 1}
//	- name: Constant_36_0	type: float	shape: Shape{5, 5, 128, 1}
//	- name: AvgPool_1427_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: DepthwiseConv2dNative_1451_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1450_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1449_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1434_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1435_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1436_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1446_0, Constant_112_0, DepthwiseConv2dNative_1451_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1445_0, Constant_201_0, DepthwiseConv2dNative_1450_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1444_0, Constant_324_0, DepthwiseConv2dNative_1449_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1434<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1426_0, Constant_109_0, DepthwiseConv2dNative_1434_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1435<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1426_0, Constant_36_0, DepthwiseConv2dNative_1435_0);
// Add_float_float_float_cuda_Add_1436<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(AvgPool_1427_0, BatchNormInference_1357_0, Add_1436_0);
// Deduped function map: <src_function_name : deduped_function_name>
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1449 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1434 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1435 : DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451

// Node name:	DepthwiseConv2dNative_1451
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1446_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_112_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1451_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	DepthwiseConv2dNative_1450
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1445_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_201_0	type: float	shape: Shape{3, 3, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1450_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
// Node name:	Add_1436
// Description:	Add
// Input:
//	- name: AvgPool_1427_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1436_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Add_float_float_float_cuda_Add_1436_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = add(input0[blockIdx.x * 512 + threadIdx.x], input1[blockIdx.x * 512 + threadIdx.x]);

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450_block_kernel(input4, input5, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1450_block_kernel(input6, input7, output3, threadIdx.x, blockIdx.x - 192 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1451_block_kernel(input6, input8, output4, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 335)
    {
        Add_float_float_float_cuda_Add_1436_block_kernel(input9, input10, output5, threadIdx.x, blockIdx.x - 320 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_DepthwiseConv2dNative_Add_134<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, output0, output1, output2, output3, output4, output5);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2966_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1610_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Convolution_1657_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2873_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: BatchNormInference_1593_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1661_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Add_1662_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_67<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1655_0, Constant_2966_0, Slice_1610_0, Add_1661_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_68<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1657_0, Constant_2873_0, BatchNormInference_1593_0, Add_1662_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_68 : FusedKernel_float_float_float_float_cuda_Add_Add_67

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1655_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_2966_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Slice_1610_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Add_1661_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Add_float_float_float_cuda_Add_2682<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1655_0, Constant_2966_0, BatchNormInference_1659_0);
// Add_float_float_float_cuda_Add_1661<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1659_0, Slice_1610_0, Add_1661_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_67_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_165(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_67_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_67_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_165_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_165<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_2796_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1138_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1076_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1140_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2910_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Slice_1094_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1147_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Add_1148_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// FusedKernel_float_float_float_float_cuda_Add_Add_38<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1138_0, Constant_2796_0, BatchNormInference_1076_0, Add_1147_0);
// FusedKernel_float_float_float_float_cuda_Add_Add_39<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1140_0, Constant_2910_0, Slice_1094_0, Add_1148_0);
// Deduped function map: <src_function_name : deduped_function_name>
// FusedKernel_float_float_float_float_cuda_Add_Add_39 : FusedKernel_float_float_float_float_cuda_Add_Add_38

// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Convolution_1138_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2796_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1076_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Add_1147_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2397<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1138_0, Constant_2796_0, BatchNormInference_1144_0);
// Add_float_float_float_cuda_Add_1147<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1144_0, BatchNormInference_1076_0, Add_1147_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(float* input0, float* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(32, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    int tid = blockIdx.x * 512 + threadIdx.x;
    float temp0 = add(input0[tid], input1[tid]);
    float temp1 = add(temp0, input2[tid]);
    output0[tid] = temp1;

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(input1, input0, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Add_38_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_cuda_fused_kernel_fused_kernel_91<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
