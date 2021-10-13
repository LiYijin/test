#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
__device__ __noinline__ void myprint()
{
  printf("hello\n");
}