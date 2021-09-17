#pragma once
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUBLAS(call)\
{\
  if (call != CUBLAS_STATUS_SUCCESS){\
    printf("CUBLAS initialization failed\n");\
    exit(1);\
  }\
}

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
} 
#define CUDA_SAFE_CALL(call) CHECK(call)

#include <time.h>
void initialData(float* ip,int size,float val=1)
{
  for(int i=0;i<size;i++)
    ip[i]=rand()*1.0/RAND_MAX;
}

void printMatrix(float * C,const int nx,const int ny)
{
  printf("Matrix<%d,%d>:\n",nx,ny);
  for(int i=0;i<nx;i++){
    for(int j=0;j<ny;j++)
      printf("%6f ",C[i*ny+j]);
    printf("\n");
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));
}

void checkResult(float * test,float * ref,const int N)
{
  double epsilon=1.0E-4;
  for(int i=0;i<N;i++){
    if(abs(test[i]-ref[i])/ref[i]>epsilon){
      printf("Results don\'t match!\n");
      printf("%f(test[%d] )!= %f(ref[%d])\n",test[i],i,ref[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}