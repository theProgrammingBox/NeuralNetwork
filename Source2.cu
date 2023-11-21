#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <cublasLt.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        exit(-1);
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

__global__ void fillDTensor(float *dTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
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
    fillDTensor<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, size, *seed1, *seed2);
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

void printBits(void *mem, size_t size) {
    unsigned char *bytes = (unsigned char *)mem;
    for (size_t i = 0; i < size; i++) {
        for (int j = 7; j >= 0; j--) {
            printf("%c", (bytes[i] & (1 << j)) ? '1' : '0');
        }
    }
    printf("\n");
}

int main() {
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    uint16_t aWidth = 4, aHeight = 3, dWidth = 2;
    
    float *dTensorA, *dTensorB, *dTensorC, *dTensorD;
    checkCudaStatus(cudaMalloc((void **)&dTensorA, dWidth * aWidth * sizeof(float)));
    checkCudaStatus(cudaMalloc((void **)&dTensorB, aWidth * aHeight * sizeof(float)));
    checkCudaStatus(cudaMalloc((void **)&dTensorC, dWidth * 1 * sizeof(float)));
    checkCudaStatus(cudaMalloc((void **)&dTensorD, dWidth * aHeight * sizeof(float)));
    
    float *dTensorDG;
    checkCudaStatus(cudaMalloc((void **)&dTensorDG, dWidth * aWidth * sizeof(float)));
    
    fillDTensor(dTensorA, dWidth * aWidth, &seed1, &seed2);
    fillDTensor(dTensorB, aWidth * aHeight, &seed1, &seed2);
    fillDTensor(dTensorC, dWidth * 1, &seed1, &seed2);
    
    printDTensor(dTensorA, dWidth, aWidth, "Weight");
    printDTensor(dTensorB, aWidth, aHeight, "Input");
    // printDTensor(dTensorC, dWidth, 1, "Bias");
    
    cublasLtHandle_t ltHandle;
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX;
    
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t descA, descB, descD;
    
    int returnedResults;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulAlgo_t algo;
    
    checkCublasStatus(cublasLtCreate(&ltHandle));
    
    size_t size = 128;
    void* idk;
    checkCudaStatus(cudaMalloc(&idk, size));
    void *hostMemory = malloc(size);
    
    checkCublasStatus(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &idk, sizeof(idk)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &size, sizeof(size)));
    
    checkCublasStatus(cublasLtMatrixLayoutCreate(&descA, CUDA_R_32F, dWidth, aWidth, dWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&descB, CUDA_R_32F, aWidth, aHeight, aWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&descD, CUDA_R_32F, dWidth, aHeight, dWidth));
    
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, descA, descB, descD, descD, preference, 1, &heuristicResult, &returnedResults));
    
    algo = heuristicResult.algo;
    float alpha = 1.0f, beta = 0.0f;
    
    checkCublasStatus(cublasLtMatmul(
        ltHandle, opDesc,
        &alpha,
        dTensorA, descA,
        dTensorB, descB,
        &beta,
        dTensorD, descD,
        dTensorD, descD,
        &algo, NULL, 0, 0));
        
    printDTensor(dTensorD, dWidth, aHeight, "Output");
    
    
    checkCudaStatus(cudaMemcpy(hostMemory, idk, size, cudaMemcpyDeviceToHost));
    printBits(hostMemory, size);
    
    float* hTensorDGrad = (float *)malloc(dWidth * aHeight * sizeof(float));
    for (uint32_t i = 0; i < dWidth * aHeight; i++) {
        hTensorDGrad[i] = 1.0f;
    }
    
    float *dTensorAGrad, *dTensorBGrad, *dTensorDGrad;
    checkCudaStatus(cudaMalloc((void **)&dTensorAGrad, dWidth * aWidth * sizeof(float)));
    checkCudaStatus(cudaMalloc((void **)&dTensorBGrad, aWidth * aHeight * sizeof(float)));
    checkCudaStatus(cudaMalloc((void **)&dTensorDGrad, dWidth * aHeight * sizeof(float)));
    
    checkCudaStatus(cudaMemcpy(dTensorDGrad, hTensorDGrad, dWidth * aHeight * sizeof(float), cudaMemcpyHostToDevice));
    
    printDTensor(dTensorDGrad, dWidth, aHeight, "Output Gradient");
    cublasLtMatmulDesc_t opInputGradDesc;
    cublasLtEpilogue_t drelu = cublasLtEpilogue_t(8 | 128);// CUBLASLT_EPILOGUE_DRELU;
    cublasOperation_t trans = CUBLAS_OP_T;
    size_t ww = 8;
    checkCublasStatus(cublasLtMatmulDescCreate(&opInputGradDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opInputGradDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opInputGradDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &drelu, sizeof(drelu)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opInputGradDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &idk, sizeof(idk)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opInputGradDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &size, sizeof(size)));
    // checkCublasStatus(cublasLtMatmulDescSetAttribute(opInputGradDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &ww, sizeof(ww)));
    
    
    printf("idk: %p\n", idk);
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, opInputGradDesc, descA, descD, descB, descB, preference, 1, &heuristicResult, &returnedResults));
    printf("idk: %p\n", idk);
    algo = heuristicResult.algo;
    
    checkCublasStatus(cublasLtMatmul(
        ltHandle, opInputGradDesc,
        &alpha,
        dTensorA, descA,
        dTensorDGrad, descD,
        &beta,
        dTensorBGrad, descB,
        dTensorBGrad, descB,
        &algo, NULL, 0, 0));
    
    
    
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(descA));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(descB));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(descD));
    checkCublasStatus(cublasLtMatmulDescDestroy(opDesc));
    checkCublasStatus(cublasLtDestroy(ltHandle));
    
    return 0;
}