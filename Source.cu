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

    checkCublasStatus(cublasLtMatrixLayoutDestroy(cDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(opDesc));
    checkCublasStatus(cublasLtDestroy(ltHandle));

    return 0;
}