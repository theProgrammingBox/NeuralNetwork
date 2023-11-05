#include <stdio.h>

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

int main()
{
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));

    const int aWidth = 4;
    const int aHeight = 2;
    const int dWidth = 3;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Create descriptors
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc;

    checkCublasStatus(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    
    checkCublasStatus(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, dWidth, aWidth, dWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, aWidth, aHeight, aWidth));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, dWidth, aHeight, dWidth));

    // heuristics
    int returnedResults = 0;
    const int requestedAlgoCount = 32;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
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
    
    // print tensors
    printTensorDev(aDev, aWidth, dWidth, "a");
    printTensorDev(bDev, aWidth, aHeight, "b");
    printTensorDev(cDev, aHeight, dWidth, "c");
    
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(cDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(opDesc));
    checkCublasStatus(cublasLtDestroy(ltHandle));

    return 0;
}