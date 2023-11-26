#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <cublas_v2.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(1);
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CuBLAS error: %d\n", status);
    exit(1);
  }
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) {
        *seed2 *= 0xbf324c81;
        *seed1 ^= *seed2;
        *seed1 *= 0x9c7493ad;
        *seed2 ^= *seed1;
    }
}

__global__ void _fillDTensor(float *dTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
	seed1 ^= idx;
    seed1 *= 0x4ba1bb47;
    seed1 ^= seed2;
    seed1 *= 0xb7ebcb79;
    dTensor[idx] = (int32_t)seed1 * 0.0000000004656612875245797f;
}

void fillDTensor(float *dTensor, uint32_t size, uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1;
    _fillDTensor<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size, *seed1, *seed2);
}

void printDTensor(float *dTensor, uint32_t width, uint32_t height, const char *label) {
    float *tensor = (float *)malloc(width * height * sizeof(float));
    checkCudaStatus(cudaMemcpy(tensor, dTensor, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s:\n", label);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            printf("%f ", tensor[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(tensor);
}