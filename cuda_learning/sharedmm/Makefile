
INCLUDES=-I../include/
test:
	nvcc  $(INCLUDES) -gencode arch=compute_75,code=sm_75 sharedmm.cu test1.cu -O2 -o noinline
	nvcc  $(INCLUDES) -gencode arch=compute_75,code=sm_75 sharedmm.cu test2.cu -O2 -o forceinline