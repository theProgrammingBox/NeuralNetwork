#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cuda_runtime.h>

int main(int argc, char **argv) {
    struct timeval start, end;
    
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float elapsedTime;
    
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event, 0);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d has compute capability %d.%d.\n", dev, deviceProp.major, deviceProp.minor);
    }
  
    gettimeofday(&end, NULL);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
    
    printf("Cpu Time: %f ms\n", (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);
    printf("Gpu Time: %f ms\n", elapsedTime);
    
  return 0;
}