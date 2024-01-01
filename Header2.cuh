#include <stdio.h>

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

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2 ^ 0x4ba1bb47;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1 ^ 0xb7ebcb79;
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) mixSeed(seed1, seed2);
}

float generateRandomF32(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return (int32_t)*seed1 * 0.0000000004656612875245797f;
}

uint32_t generateRandomUI32(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return *seed1;
}

__global__ void fillDeviceTensor(float *deviceTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
	seed1 ^= idx ^ 0x9c7493ad;
    seed1 *= 0x4ba1bb47;
    seed1 ^= seed2 ^ 0xbf324c81;
    seed1 *= 0xb7ebcb79;
    deviceTensor[idx] = (int32_t)seed1 * 0.0000000004656612875245797f;
}