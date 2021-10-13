// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
__device__ __forceinline__ float relu(float x0)
{
    return fmaxf(0,x0);
}
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
// Node name:	Constant_266
// Description:	Constant
// Input:
// Output:
//	- name: Constant_266_0	type: float	shape: Shape{3, 3, 128, 1}
void Constant_float_cuda_Constant_266(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_266_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_266_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4608];
    bin_file.read(tmp_mem, 4608);
    cudaMemcpyAsync(output0, tmp_mem, 4608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_185
// Description:	Constant
// Input:
// Output:
//	- name: Constant_185_0	type: float	shape: Shape{128}
void Constant_float_cuda_Constant_185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_185_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[512];
    bin_file.read(tmp_mem, 512);
    cudaMemcpyAsync(output0, tmp_mem, 512, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_144
// Description:	Constant
// Input:
// Output:
//	- name: Constant_144_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_144(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_144_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_144_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_177
// Description:	Constant
// Input:
// Output:
//	- name: Constant_177_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_177(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_177_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_177_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_345
// Description:	Constant
// Input:
// Output:
//	- name: Constant_345_0	type: float	shape: Shape{5, 5, 128, 1}
void Constant_float_cuda_Constant_345(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_345_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_345_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[12800];
    bin_file.read(tmp_mem, 12800);
    cudaMemcpyAsync(output0, tmp_mem, 12800, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3164
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3164_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_3164(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3164_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3164_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_137
// Description:	Constant
// Input:
// Output:
//	- name: Constant_137_0	type: float	shape: Shape{3, 3, 64, 1}
void Constant_float_cuda_Constant_137(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_137_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_137_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3000
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3000_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3000(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3000_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3000_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1934
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1934_0	type: float	shape: Shape{64, 384, 1, 1}
void Constant_float_cuda_Constant_1934(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1934_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1934_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[98304];
    bin_file.read(tmp_mem, 98304);
    cudaMemcpyAsync(output0, tmp_mem, 98304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2425
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2425_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2425(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2425_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2425_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3018
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3018_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3018(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3018_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3018_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2870
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2870_0	type: float	shape: Shape{1, 64, 16, 16}
void Constant_float_cuda_Constant_2870(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2870_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2870_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2347
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2347_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2347(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2347_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2347_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Relu_1367_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_273_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: Relu_1360_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: DepthwiseConv2dNative_1374_0	type: float	shape: Shape{1, 128, 8, 8}
// Fused functions:
// Relu_float_float_cuda_Relu_1360<<<dim3(16, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1357_0, Relu_1360_0);
// DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1374<<<dim3(64, 1, 1), dim3(128, 1, 1), 0, 0>>>(Relu_1367_0, Constant_273_0, DepthwiseConv2dNative_1374_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Relu_1360
// Description:	Relu
// Input:
//	- name: BatchNormInference_1357_0	type: float	shape: Shape{1, 128, 8, 8}
// Output:
//	- name: Relu_1360_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void Relu_float_float_cuda_Relu_1360_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 512){
        return;
    }
    const dim3 blockDim(512, 1, 1);
    const dim3 gridDim(16, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 512 + threadIdx.x] = relu(input0[blockIdx.x * 512 + threadIdx.x]);

}
// Node name:	DepthwiseConv2dNative_1374
// Description:	DepthwiseConv2dNative
// Input:
//	- name: Relu_1367_0	type: float	shape: Shape{1, 128, 8, 8}
//	- name: Constant_273_0	type: float	shape: Shape{5, 5, 128, 1}
// Output:
//	- name: DepthwiseConv2dNative_1374_0	type: float	shape: Shape{1, 128, 8, 8}
__device__ __noinline__ void DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1374_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_cuda_Relu_DepthwiseConv2dNative_124(float* input0, float* input1, float* input2, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 15)
    {
        Relu_float_float_cuda_Relu_1360_block_kernel(input0, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 16 && (int)blockIdx.x <= 79)
    {
        DepthwiseConv2dNative_float_float_float_cuda_DepthwiseConv2dNative_1374_block_kernel(input1, input2, output1, threadIdx.x, blockIdx.x - 16 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_cuda_Relu_DepthwiseConv2dNative_124_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_cuda_Relu_DepthwiseConv2dNative_124<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2920_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Convolution_1217_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2922_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2439<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1215_0, Constant_2920_0, BatchNormInference_1218_0);
// FusedKernel_float_float_float_float_cuda_Add_Relu_44<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1217_0, Constant_2922_0, Relu_1221_0, BatchNormInference_1219_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	Add_2439
// Description:	Add
// Input:
//	- name: Convolution_1215_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2920_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: BatchNormInference_1218_0	type: float	shape: Shape{1, 64, 16, 16}
__device__ __noinline__ void Add_float_float_float_cuda_Add_2439_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
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
//	- name: Convolution_1217_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: Constant_2922_0	type: float	shape: Shape{1, 64, 16, 16}
// Output:
//	- name: Relu_1221_0	type: float	shape: Shape{1, 64, 16, 16}
//	- name: BatchNormInference_1219_0	type: float	shape: Shape{1, 64, 16, 16}
// Fused functions:
// Add_float_float_float_cuda_Add_2442<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(Convolution_1217_0, Constant_2922_0, BatchNormInference_1219_0);
// Relu_float_float_cuda_Relu_1221<<<dim3(32, 1, 1), dim3(512, 1, 1), 0, 0>>>(BatchNormInference_1219_0, Relu_1221_0);
__device__ __noinline__ void FusedKernel_float_float_float_float_cuda_Add_Relu_44_block_kernel(float* input0, float* input1, float* output0, float* output1, int thread_id, int block_id, char *shared_buffer)
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
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 31)
    {
        Add_float_float_float_cuda_Add_2439_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 32 && (int)blockIdx.x <= 63)
    {
        FusedKernel_float_float_float_float_cuda_Add_Relu_44_block_kernel(input2, input3, output2, output1, threadIdx.x, blockIdx.x - 32 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1, float* output2) {
    BlockFusionKernel_float_float_float_float_float_float_float_cuda_Add_fused_kernel_102<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1, output2);
}
