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

float generateRandomFloat(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return (int32_t)*seed1 * 0.0000000004656612875245797f;
}

uint32_t generateRandomUint32(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return *seed1;
}

__global__ void _fillDTensor(float *dTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
	seed1 ^= idx ^ 0x9c7493ad;
    seed1 *= 0x4ba1bb47;
    seed1 ^= seed2 ^ 0xbf324c81;
    seed1 *= 0xb7ebcb79;
    dTensor[idx] = (int32_t)seed1 * 0.0000000004656612875245797f;
}

void fillDTensor(float *dTensor, uint32_t size, uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    _fillDTensor<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size, *seed1, *seed2);
}

__global__ void _customFillDTensor(float *dTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx <= 0) {
        dTensor[idx] = 1.0f;
        return;
    }
	seed1 ^= idx ^ 0x9c7493ad;
    seed1 *= 0x4ba1bb47;
    seed1 ^= seed2 ^ 0xbf324c81;
    seed1 *= 0xb7ebcb79;
    dTensor[idx] = (int32_t)seed1 * 0.0000000004656612875245797f;
}

void customFillDTensor(float *dTensor, uint32_t size, uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    _customFillDTensor<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size, *seed1, *seed2);
}

__global__ void _customFillDTensorConstant(float *dTensor, uint32_t size, float constant) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] = constant;
}

void customFillDTensorConstant(float *dTensor, uint32_t size, float constant) {
    _customFillDTensorConstant<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size, constant);
}

__global__ void _integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, float betaMean, float betaVar, float betaMeanCor, float betaVarCor, float learningRate, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float grad = dTensorGrad[idx];
    float mean = betaMean * dTensorMean[idx] + (1.0f - betaMean) * grad;
    float var = betaVar * dTensorVar[idx] + (1.0f - betaVar) * grad * grad;
    float meanCor = mean / (1.0f - betaMeanCor);
    float varCor = var / (1.0f - betaVarCor);
    dTensorMean[idx] = mean;
    dTensorVar[idx] = var;
    dTensor[idx] += learningRate * meanCor / (sqrtf(varCor) + 1e-8f);
}

void integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, float betaMean, float betaVar, float betaMeanCor, float betaVarCor, float learningRate, uint32_t size) {
    _integratedAdamUpdate<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, dTensorGrad, dTensorMean, dTensorVar, betaMean, betaVar, betaMeanCor, betaVarCor, learningRate, size);
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

__global__ void _reluForward(float *dTensor, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] = dTensor[idx] > 0 ? dTensor[idx] : 0;
}

void reluForward(float *dTensor, uint32_t size) {
    _reluForward<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size);
}

__global__ void _reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensorGrad[idx] = dTensor[idx] > 0 ? dTensorGrad[idx] : 0;
}

void reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    _reluBackward<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, dTensorGrad, size);
}