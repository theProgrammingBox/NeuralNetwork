#include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    checkCudaStatus(cudaMalloc((void **)&d_A, sizeof(float) * 3 * 2));
    checkCudaStatus(cudaMalloc((void **)&d_B, sizeof(float) * 2 * 3));
    checkCudaStatus(cudaMalloc((void **)&d_C, sizeof(float) * 3 * 3));

    float h_A[6] = {1, 2, 3, 4, 5, 6};
    float h_B[6] = {1, 2, 3, 4, 5, 6};
    float h_C[9] = {0};

    checkCudaStatus(cudaMemcpy(d_A, h_A, sizeof(float) * 3 * 2, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_B, h_B, sizeof(float) * 2 * 3, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_C, h_C, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    checkCublasStatus(cublasSgemm(
        handle, CUBLAS_OP_N,CUBLAS_OP_N,
        3, 3, 2,
        &alpha,
        d_A, 3,
        d_B, 2,
        &beta,
        d_C, 3));

    checkCudaStatus(cudaMemcpy(h_C, d_C, sizeof(float) * 3 * 3, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 9; i++)
        printf("%f ", h_C[i]);

    checkCudaStatus(cudaFree(d_A));
    checkCudaStatus(cudaFree(d_B));
    checkCudaStatus(cudaFree(d_C));

    return 0;
}