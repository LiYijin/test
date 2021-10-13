// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "shared.h"
// Node name:	Constant_2539
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2539_0	type: float	shape: Shape{128, 128, 1, 1}
void Constant_float_cuda_Constant_2539(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2539_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2539_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[65536];
    bin_file.read(tmp_mem, 65536);
    cudaMemcpyAsync(output0, tmp_mem, 65536, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_360
// Description:	Constant
// Input:
// Output:
//	- name: Constant_360_0	type: float	shape: Shape{5, 5, 32, 1}
void Constant_float_cuda_Constant_360(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_360_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_360_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3200];
    bin_file.read(tmp_mem, 3200);
    cudaMemcpyAsync(output0, tmp_mem, 3200, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2877
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2877_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2877(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2877_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2877_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2944
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2944_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2944(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2944_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2944_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2467
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2467_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2467(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2467_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2467_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_72
// Description:	Constant
// Input:
// Output:
//	- name: Constant_72_0	type: float	shape: Shape{3, 3, 32, 1}
void Constant_float_cuda_Constant_72(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_72_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_72_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1152];
    bin_file.read(tmp_mem, 1152);
    cudaMemcpyAsync(output0, tmp_mem, 1152, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2359
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2359_0	type: float	shape: Shape{64, 64, 1, 1}
void Constant_float_cuda_Constant_2359(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2359_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2359_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[16384];
    bin_file.read(tmp_mem, 16384);
    cudaMemcpyAsync(output0, tmp_mem, 16384, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2188
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2188_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2188(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2188_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2188_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3022
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3022_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_3022(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3022_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3022_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2176
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2176_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2176(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2176_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2176_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2185
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2185_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2185(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2185_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2185_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2191
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2191_0	type: float	shape: Shape{32, 192, 1, 1}
void Constant_float_cuda_Constant_2191(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2191_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2191_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[24576];
    bin_file.read(tmp_mem, 24576);
    cudaMemcpyAsync(output0, tmp_mem, 24576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2858
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2858_0	type: float	shape: Shape{1, 32, 32, 32}
void Constant_float_cuda_Constant_2858(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2858_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2858_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[131072];
    bin_file.read(tmp_mem, 131072);
    cudaMemcpyAsync(output0, tmp_mem, 131072, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2059
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2059_0	type: float	shape: Shape{32, 32, 1, 1}
void Constant_float_cuda_Constant_2059(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2059_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2059_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4096];
    bin_file.read(tmp_mem, 4096);
    cudaMemcpyAsync(output0, tmp_mem, 4096, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2812
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2812_0	type: float	shape: Shape{1, 128, 8, 8}
void Constant_float_cuda_Constant_2812(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2812_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2812_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[32768];
    bin_file.read(tmp_mem, 32768);
    cudaMemcpyAsync(output0, tmp_mem, 32768, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
