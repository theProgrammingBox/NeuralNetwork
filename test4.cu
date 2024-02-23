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
    
    const uint32_t samples = 1000;
    
    float cpuTimes[samples];
    float gpuTimes[samples];
    
    for (uint32_t i = 0; i < samples; i++) {
        gettimeofday(&start, NULL);
        cudaEventRecord(start_event, 0);
        
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        // printf("Number of CUDA devices: %d\n", deviceCount);
        for (int dev = 0; dev < deviceCount; dev++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            // printf("Device %d has compute capability %d.%d.\n", dev, deviceProp.major, deviceProp.minor);
        }
    
        gettimeofday(&end, NULL);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        
        cpuTimes[i] = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        gpuTimes[i] = elapsedTime;
    }
    // printf("Cpu Time: %f ms\n", (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);
    // printf("Gpu Time: %f ms\n", elapsedTime);
    
    // compute median, mean, and standard deviation
    float cpuMedian = 0.0;
    float gpuMedian = 0.0;
    float cpuMean = 0.0;
    float gpuMean = 0.0;
    float cpuStdDev = 0.0;
    float gpuStdDev = 0.0;
    
    for (uint32_t i = 0; i < samples; i++) {
        cpuMean += cpuTimes[i];
        gpuMean += gpuTimes[i];
    }
    cpuMean /= samples;
    gpuMean /= samples;
    
    for (uint32_t i = 0; i < samples; i++) {
        cpuStdDev += (cpuTimes[i] - cpuMean) * (cpuTimes[i] - cpuMean);
        gpuStdDev += (gpuTimes[i] - gpuMean) * (gpuTimes[i] - gpuMean);
    }
    cpuStdDev = sqrt(cpuStdDev / samples);
    gpuStdDev = sqrt(gpuStdDev / samples);
    
    for (uint32_t i = 0; i < samples; i++) {
        for (uint32_t j = i + 1; j < samples; j++) {
            if (cpuTimes[i] > cpuTimes[j]) {
                float temp = cpuTimes[i];
                cpuTimes[i] = cpuTimes[j];
                cpuTimes[j] = temp;
            }
            if (gpuTimes[i] > gpuTimes[j]) {
                float temp = gpuTimes[i];
                gpuTimes[i] = gpuTimes[j];
                gpuTimes[j] = temp;
            }
        }
    }
    cpuMedian = (samples % 2 == 0) ? (cpuTimes[samples / 2 - 1] + cpuTimes[samples / 2]) / 2 : cpuTimes[samples / 2];
    gpuMedian = (samples % 2 == 0) ? (gpuTimes[samples / 2 - 1] + gpuTimes[samples / 2]) / 2 : gpuTimes[samples / 2];
    
    printf("CPU Median: %f ms\n", cpuMedian);
    printf("GPU Median: %f ms\n\n", gpuMedian);
    printf("CPU Mean: %f ms\n", cpuMean);
    printf("GPU Mean: %f ms\n\n", gpuMean);
    printf("CPU Standard Deviation: %f ms\n", cpuStdDev);
    printf("GPU Standard Deviation: %f ms\n\n", gpuStdDev);
    
    return 0;
}