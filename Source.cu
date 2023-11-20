#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>
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

inline void printTensorDev(float* TensorDev, int width, int height, const char* name)
{
    float* TensorHost = (float*)malloc(width * height * sizeof(float));
    checkCudaStatus(cudaMemcpy(TensorHost, TensorDev, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("%s:\n", name);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            printf("%f, ", TensorHost[i * width + j]);
        printf("\n");
    }
    printf("\n");
    free(TensorHost);
}

__global__ void gpuRandFunc(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint32_t Hash = idx;

        Hash ^= seed1;
        Hash *= 0xBAC57D37;
        Hash ^= seed2;
        Hash *= 0x24F66AC9;

        arr[idx] = int32_t(Hash) * 0.0000000004656612875245796f;
    }
}

struct GpuRand {
    uint32_t seed1, seed2;

    GpuRand() {
        seed1 = time(NULL) ^ 0xE621B963;
        seed2 = 0x6053653F ^ (time(NULL) >> 32);

        printf("Seed1: %u\n", seed1);
        printf("Seed2: %u\n\n", seed2);
    }

    void Rand(float* arr, uint32_t size) {
        seed1 ^= seed2;
        seed1 *= 0xBAC57D37;
        seed2 ^= seed1;
        seed2 *= 0x24F66AC9;

        gpuRandFunc << <ceil(0.0009765625f * size), 1024 >> > (arr, size, seed1, seed2);
    }
};

int main()
{
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));

    const int aWidth = 1024;
    const int aHeight = 1024;
    const int dWidth = 1024;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Create descriptors
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc;

    checkCublasStatus(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    
    checkCublasStatus(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, dWidth, aWidth, dWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, aWidth, aHeight, aWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, dWidth, aHeight, dWidth));

    // heuristics
    // void *workspace;
    // size_t workspaceSize = 0;//1024 * 1024 * 4;
    // checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    
    int returnedResults = 0;
    const int requestedAlgoCount = 32;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    // checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, aDesc, bDesc, cDesc, cDesc, preference, requestedAlgoCount, heuristicResult, &returnedResults));
    
    if (returnedResults == 0)
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    
    // print heuristics
    for (int i = 0; i < returnedResults; i++)
        printf("heuristic %d: workspace %zu\n", i, heuristicResult[i].workspaceSize);
    printf("\n");
    
    // allocate memory
    float *aDev, *bDev, *cDev;
    checkCudaStatus(cudaMalloc((void**)&aDev, dWidth * aWidth * sizeof(float)));
    checkCudaStatus(cudaMalloc((void**)&bDev, aWidth * aHeight * sizeof(float)));
    checkCudaStatus(cudaMalloc((void**)&cDev, dWidth * aHeight * sizeof(float)));
    
    // initialize tensors
    GpuRand rand;
    rand.Rand(aDev, dWidth * aWidth);
    rand.Rand(bDev, aWidth * aHeight);
    // rand.Rand(cDev, dWidth * aHeight);
    
    // benchmarking
    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaStatus(cudaStreamCreate(&stream));
    checkCudaStatus(cudaEventCreate(&start));
    checkCudaStatus(cudaEventCreate(&stop));
    
    const int repeat = 4096;
    float times[repeat];
    float bestTime = 0;
    int bestAlgo = 0;
    float meanTime, medianTime;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int algo = returnedResults; algo--;)
    {
        meanTime = 0;
        medianTime = 0;
        for (int rep = repeat; rep--;)
        {
            checkCudaStatus(cudaEventRecord(start, stream));
            
            checkCublasStatus(cublasLtMatmul(
                ltHandle, opDesc, 
                &alpha,
                aDev, aDesc,
                bDev, bDesc, 
                &beta,
                cDev, cDesc,
                cDev, cDesc, 
                &heuristicResult[algo].algo, NULL, 0, stream));
                
            checkCudaStatus(cudaEventRecord(stop, stream));
            checkCudaStatus(cudaEventSynchronize(stop));
            checkCudaStatus(cudaEventElapsedTime(&times[rep], start, stop));
            meanTime += times[rep];
        }
        qsort(times, repeat, sizeof(float), [](const void* a, const void* b) -> int
        {
            return *(float*)a > *(float*)b;
        });
        
        medianTime = times[repeat / 2];
        printf("algo: %d, medianTime: %f, meanTime: %f, workspace: %zu\n", algo, medianTime, meanTime / repeat, heuristicResult[algo].workspaceSize);
        if (bestTime == 0 || medianTime < bestTime)
        {
            bestTime = medianTime;
            bestAlgo = algo;
        }
    }
    
    printf("best algo: %d, medianTime: %f, workspace: %zu\n", bestAlgo, bestTime, heuristicResult[bestAlgo].workspaceSize);
    
    // print tensors
    // printTensorDev(bDev, aWidth, aHeight, "b");
    // printTensorDev(aDev, dWidth, aWidth, "a");
    // printTensorDev(cDev, dWidth, aHeight, "c");
    
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(cDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(opDesc));
    checkCublasStatus(cublasLtDestroy(ltHandle));

    return 0;
}