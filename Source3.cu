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

int main() {
    const uint8_t networkBatches = 16;
    const uint8_t networkLayers = 2;
    uint8_t networkParameter[networkLayers + 1] = {2, 8, 1};
    float* weightTensors[networkLayers], * outputTensors[networkLayers + 1];
    uint32_t seed1, seed2;
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t OutputOpDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayouts[networkLayers], outputLayouts[networkLayers + 1];
    cublasLtMatmulAlgo_t algos[networkLayers];
    cublasLtEpilogue_t reluEp = CUBLASLT_EPILOGUE_RELU;
    // cublasOperation_t transOp = CUBLAS_OP_T;
    
    initializeSeeds(&seed1, &seed2);
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCublasStatus(cublasLtMatmulDescCreate(&OutputOpDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    
    checkCublasStatus(cublasLtMatmulDescSetAttribute(OutputOpDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &reluEp, sizeof(reluEp)));
    
    checkCudaStatus(cudaMalloc(&outputTensors[0], networkParameter[0] * networkBatches * sizeof(float)));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputLayouts[0], CUDA_R_32F, networkParameter[0], networkBatches, networkParameter[0]));
    for (uint8_t i = 0; i < networkLayers; i++) {
        cublasLtMatmulHeuristicResult_t heuristicResult;
        int returnedResults;
        checkCudaStatus(cudaMalloc(&weightTensors[i], networkParameter[i + 1] * networkParameter[i] * sizeof(float)));
        checkCudaStatus(cudaMalloc(&outputTensors[i + 1], networkParameter[i + 1] * networkBatches * sizeof(float)));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&weightLayouts[i], CUDA_R_32F, networkParameter[i + 1], networkParameter[i], networkParameter[i + 1]));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&outputLayouts[i + 1], CUDA_R_32F, networkParameter[i + 1], networkBatches, networkParameter[i + 1]));
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, OutputOpDesc, weightLayouts[i], outputLayouts[i], outputLayouts[i + 1], outputLayouts[i + 1], preference, 1, &heuristicResult, &returnedResults));
        algos[i] = heuristicResult.algo;
        
        fillDTensor(weightTensors[i], networkParameter[i + 1] * networkParameter[i], &seed1, &seed2);
    }
    
    fillDTensor(outputTensors[0], networkParameter[0] * networkBatches, &seed1, &seed2);
    
    float alpha = 1.0f, beta = 0.0f;
    for (uint8_t i = 0; i < networkLayers; i++) {
        checkCublasStatus(cublasLtMatmul(
            ltHandle, OutputOpDesc,
            &alpha,
            weightTensors[i], weightLayouts[i],
            outputTensors[i], outputLayouts[i],
            &beta,
            outputTensors[i + 1], outputLayouts[i + 1],
            outputTensors[i + 1], outputLayouts[i + 1],
            &algos[i], NULL, 0, 0));
    }
    
    printDTensor(outputTensors[0], networkParameter[0], networkBatches, "Input");
    for (uint8_t i = 0; i < networkLayers; i++) {
        printDTensor(weightTensors[i], networkParameter[i + 1], networkParameter[i], "Weights");
        printDTensor(outputTensors[i + 1], networkParameter[i + 1], networkBatches, "Output");
    }
    
    return 0;
}