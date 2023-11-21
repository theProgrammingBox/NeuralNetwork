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
        *seed1 ^= *seed1 >> 13;
        *seed1 *= 0x9c7493ad;
        *seed2 ^= *seed1;
        *seed2 ^= *seed2 >> 17;
    }
}

__global__ void _fillDTensor(float *dTensor, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
	seed1 ^= idx;
    seed1 *= 0x4ba1bb47;
    seed1 ^= seed1 >> 17;
    seed1 ^= seed2;
    seed1 *= 0xb7ebcb79;
    seed1 ^= seed1 >> 13;
    dTensor[idx] = (int32_t)seed1 * 0.0000000004656612875245797f;
}

void fillDTensor(float *dTensor, uint32_t size, uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2;
    *seed1 ^= *seed1 >> 13;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1;
    *seed2 ^= *seed2 >> 17;
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

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2;
    *seed1 ^= *seed1 >> 17;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1;
    *seed2 ^= *seed2 >> 13;
}

__global__ void _addDTensors(float alpha, float *dTensor1, float *dTensor2, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor1[idx] += alpha * dTensor2[idx];
}

void addDTensors(float alpha, float *dTensor1, float *dTensor2, uint32_t size) {
    _addDTensors<<<(size >> 10) + (size & 0x3ff), 0x400>>>(alpha, dTensor1, dTensor2, size);
}

int main() {
    const uint16_t networkBatches = 1024;
    const uint8_t networkLayers = 3;
    uint8_t networkParameters[networkLayers + 1] = {3, 4, 4, 1};
    float* weightTensors[networkLayers], * outputTensors[networkLayers + 1];
    float* outputGradTensors[networkLayers + 1], * weightGradTensors[networkLayers];
    uint32_t seed1, seed2;
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t OutputOpDesc, InputGradOpDesc, WeightGradOpDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayouts[networkLayers], outputLayouts[networkLayers + 1];
    cublasLtMatmulAlgo_t outputAlgos[networkLayers], inputGradAlgos[networkLayers], weightGradAlgos[networkLayers];
    cublasLtEpilogue_t reluEp = CUBLASLT_EPILOGUE_RELU;
    cublasOperation_t transOp = CUBLAS_OP_T;
    
    initializeSeeds(&seed1, &seed2);
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCublasStatus(cublasLtMatmulDescCreate(&OutputOpDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescCreate(&InputGradOpDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescCreate(&WeightGradOpDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    
    checkCublasStatus(cublasLtMatmulDescSetAttribute(OutputOpDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &reluEp, sizeof(reluEp)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(InputGradOpDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transOp, sizeof(transOp)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(WeightGradOpDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transOp, sizeof(transOp)));
    
    checkCudaStatus(cudaMalloc(&outputTensors[0], networkParameters[0] * networkBatches * sizeof(float)));
    checkCudaStatus(cudaMalloc(&outputGradTensors[0], networkParameters[0] * networkBatches * sizeof(float)));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputLayouts[0], CUDA_R_32F, networkParameters[0], networkBatches, networkParameters[0]));
    for (uint8_t i = 0; i < networkLayers; i++) {
        checkCudaStatus(cudaMalloc(&weightTensors[i], networkParameters[i + 1] * networkParameters[i] * sizeof(float)));
        checkCudaStatus(cudaMalloc(&outputTensors[i + 1], networkParameters[i + 1] * networkBatches * sizeof(float)));
        
        checkCudaStatus(cudaMalloc(&outputGradTensors[i + 1], networkParameters[i + 1] * networkBatches * sizeof(float)));
        checkCudaStatus(cudaMalloc(&weightGradTensors[i], networkParameters[i + 1] * networkParameters[i] * sizeof(float)));
        
        checkCublasStatus(cublasLtMatrixLayoutCreate(&weightLayouts[i], CUDA_R_32F, networkParameters[i + 1], networkParameters[i], networkParameters[i + 1]));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&outputLayouts[i + 1], CUDA_R_32F, networkParameters[i + 1], networkBatches, networkParameters[i + 1]));
        
        cublasLtMatmulHeuristicResult_t heuristicResult;
        int returnedResults;
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, OutputOpDesc, weightLayouts[i], outputLayouts[i], outputLayouts[i + 1], outputLayouts[i + 1], preference, 1, &heuristicResult, &returnedResults));
        outputAlgos[i] = heuristicResult.algo;
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, InputGradOpDesc, weightLayouts[i], outputLayouts[i + 1], outputLayouts[i], outputLayouts[i], preference, 1, &heuristicResult, &returnedResults));
        inputGradAlgos[i] = heuristicResult.algo;
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, WeightGradOpDesc, outputLayouts[i + 1], outputLayouts[i], weightLayouts[i], weightLayouts[i], preference, 1, &heuristicResult, &returnedResults));
        weightGradAlgos[i] = heuristicResult.algo;
        
        fillDTensor(weightTensors[i], networkParameters[i + 1] * networkParameters[i], &seed1, &seed2);
    }
    
    float inputTensor[networkParameters[0] * networkBatches];
    float outputGradTensor[networkParameters[networkLayers] * networkBatches];
    float expectedTensor[networkParameters[networkLayers] * networkBatches];
    float one = 1.0f, zero = 0.0f, negOne = -1.0f, learningRate = 0.1f / networkBatches;
    for (uint32_t epoch = 0; epoch < 1000; epoch++) {
        for (uint16_t i = 0; i < networkBatches; i++) {
            mixSeed(&seed1, &seed2);
            uint8_t a = seed1 & 1, b = seed2 & 1;
            inputTensor[i * networkParameters[0]] = a;
            inputTensor[i * networkParameters[0] + 1] = b;
            inputTensor[i * networkParameters[0] + 2] = 1;
            expectedTensor[i * networkParameters[networkLayers]] = a ^ b;
        }
        
        checkCudaStatus(cudaMemcpy(outputTensors[0], inputTensor, networkParameters[0] * networkBatches * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(outputGradTensors[networkLayers], expectedTensor, networkParameters[networkLayers] * networkBatches * sizeof(float), cudaMemcpyHostToDevice));
        
        for (uint8_t i = 0; i < networkLayers; i++) {
            checkCublasStatus(cublasLtMatmul(
                ltHandle, OutputOpDesc,
                &one,
                weightTensors[i], weightLayouts[i],
                outputTensors[i], outputLayouts[i],
                &zero,
                outputTensors[i + 1], outputLayouts[i + 1],
                outputTensors[i + 1], outputLayouts[i + 1],
                &outputAlgos[i], NULL, 0, 0));
        }
        
        addDTensors(negOne, outputGradTensors[networkLayers], outputTensors[networkLayers], networkParameters[networkLayers] * networkBatches);
        checkCudaStatus(cudaMemcpy(outputGradTensor, outputGradTensors[networkLayers], networkParameters[networkLayers] * networkBatches * sizeof(float), cudaMemcpyDeviceToHost));
        float error = 0;
        for (uint16_t i = networkParameters[networkLayers] * networkBatches; i--;) {
            error += fabs(outputGradTensor[i]);
        }
        printf("Error: %f\n", error / networkBatches);
        
        for (uint8_t i = networkLayers; i--;) {
            checkCublasStatus(cublasLtMatmul(
                ltHandle, InputGradOpDesc,
                &one,
                weightTensors[i], weightLayouts[i],
                outputGradTensors[i + 1], outputLayouts[i + 1],
                &zero,
                outputGradTensors[i], outputLayouts[i],
                outputGradTensors[i], outputLayouts[i],
                &inputGradAlgos[i], NULL, 0, 0));
                
            checkCublasStatus(cublasLtMatmul(
                ltHandle, WeightGradOpDesc,
                &one,
                outputGradTensors[i + 1], outputLayouts[i + 1],
                outputTensors[i], outputLayouts[i],
                &zero,
                weightGradTensors[i], weightLayouts[i],
                weightGradTensors[i], weightLayouts[i],
                &weightGradAlgos[i], NULL, 0, 0));
        }
        
        for (uint8_t i = 0; i < networkLayers; i++) {
            addDTensors(learningRate, weightTensors[i], weightGradTensors[i], networkParameters[i + 1] * networkParameters[i]);
        }
    }
        
    // printDTensor(outputTensors[0], networkParameters[0], networkBatches, "Input");
    // for (uint8_t i = 0; i < networkLayers; i++) {
    //     printDTensor(weightTensors[i], networkParameters[i + 1], networkParameters[i], "Weights");
    //     printDTensor(outputTensors[i + 1], networkParameters[i + 1], networkBatches, "Output");
    // }
        
    // printDTensor(outputGradTensors[networkLayers], networkParameters[networkLayers], networkBatches, "Output Grad");
    // for (uint8_t i = networkLayers; i--;) {
    //     printDTensor(weightGradTensors[i], networkParameters[i + 1], networkParameters[i], "Weight Grad");
    //     printDTensor(outputGradTensors[i], networkParameters[i], networkBatches, "Input Grad");
    // }
    
    return 0;
}