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

int compareFloats(const void* a, const void* b) {
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

__global__ void compareTensors(const float *tensorA, const float *tensorB, uint32_t size, bool *result) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (fabsf(tensorA[idx] - tensorB[idx]) > 0.001f) {
        *result = false;
    }
}

bool areTensorsEqual(float *dTensorA, float *dTensorB, uint32_t size) {
    bool *dResult, hResult = true;
    cudaMalloc(&dResult, sizeof(bool));
    cudaMemcpy(dResult, &hResult, sizeof(bool), cudaMemcpyHostToDevice);

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    compareTensors<<<blocks, threadsPerBlock>>>(dTensorA, dTensorB, size, dResult);

    cudaMemcpy(&hResult, dResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(dResult);
    
    return hResult;
}

__global__ void matrixMul(const float *A, const float *B, float *C, 
                          int M, int N, int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(ty < M && tx < N) {
        float c = 0;
        for(int i = 0; i < K; ++i){
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = a + by * BLOCK * k;
  float *begin_b = b + bx * BLOCK;
  float *end_a = begin_a + k;

  float sum = 0.f;
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * n) {
    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    ashare[ty][tx] = a_ptr[ty * k + tx];
    bshare[ty][tx] = b_ptr[ty * n + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }

  c[(BLOCK * by + ty) * n + BLOCK * bx + tx] = sum;
}


int main() {
  uint32_t seed1, seed2;
  initializeSeeds(&seed1, &seed2);

  const uint32_t widthA = 1024;
  const uint32_t heightA = 2048; 
  const uint32_t widthC = 512;
  
  float *dTensorA, *dTensorB, *dTensorC, *dTensorCCublas;
  
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));

  cudaMalloc(&dTensorA, widthA * heightA * sizeof(float));
  cudaMalloc(&dTensorB, widthC * widthA * sizeof(float)); 
  cudaMalloc(&dTensorC, heightA * widthC * sizeof(float));
  cudaMalloc(&dTensorCCublas, heightA * widthC * sizeof(float));
  
  fillDTensor(dTensorA, widthA * heightA, &seed1, &seed2);
  fillDTensor(dTensorB, widthC * widthA, &seed1, &seed2);
  
  // printDTensor(dTensorA, widthA, heightA, "A");
  // printDTensor(dTensorB, widthC, widthA, "B");
  
  const uint32_t samples = 1024 * 4;
  
  float times[samples];
  
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  struct timeval t_start, t_end;
  
  float mean, median;
  
  
  
  for (uint32_t i = samples; i--;) {

    gettimeofday(&t_start, NULL);
  
    cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      widthC, heightA, widthA,
      &alpha,
      dTensorB, widthC,
      dTensorA, widthA,
      &beta,
      dTensorCCublas, widthC);
      
    gettimeofday(&t_end, NULL);

    times[i] = (t_end.tv_sec - t_start.tv_sec) * 1000.0f + (t_end.tv_usec - t_start.tv_usec) / 1000.0f;
  }
    
  // printDTensor(dTensorCCublas, widthC, heightA, "C");
  
  mean = 0.0f;
  for (uint32_t i = samples; i--;) mean += times[i];
  mean /= samples;
  
  qsort(times, samples, sizeof(float), compareFloats);
  median = times[samples >> 1];
  
  printf("Mean: %f ms\n", mean);
  printf("Median: %f ms\n", median);
  
  
  
  for (uint32_t i = samples; i--;) {

    gettimeofday(&t_start, NULL);
    
    dim3 threads(32, 32);
    dim3 blocks((widthC + threads.x - 1) / threads.x, (heightA + threads.y - 1) / threads.y);
    matrixMul<<<blocks, threads>>>(dTensorA, dTensorB, dTensorC, heightA, widthC, widthA);
      
    gettimeofday(&t_end, NULL);

    times[i] = (t_end.tv_sec - t_start.tv_sec) * 1000.0f + (t_end.tv_usec - t_start.tv_usec) / 1000.0f;
  }
    
  areTensorsEqual(dTensorC, dTensorCCublas, heightA * widthC) ? printf("\nTensors are equal\n") : printf("\nTensors are not equal\n");
  
  mean = 0.0f;
  for (uint32_t i = samples; i--;) mean += times[i];
  mean /= samples;
  
  qsort(times, samples, sizeof(float), compareFloats);
  median = times[samples >> 1];
  
  printf("Mean: %f ms\n", mean);
  printf("Median: %f ms\n", median);
  
  
  
  for (uint32_t i = samples; i--;) {

    gettimeofday(&t_start, NULL);
    
    dim3 threads(32, 32);
    dim3 blocks((widthC + threads.x - 1) / threads.x, (heightA + threads.y - 1) / threads.y);
    sgemm<32><<<blocks, threads>>>(heightA, widthC, widthA, dTensorA, widthA, dTensorB, widthC, dTensorC, widthC);
      
    gettimeofday(&t_end, NULL);

    times[i] = (t_end.tv_sec - t_start.tv_sec) * 1000.0f + (t_end.tv_usec - t_start.tv_usec) / 1000.0f;
  }
  
  areTensorsEqual(dTensorC, dTensorCCublas, heightA * widthC) ? printf("\nTensors are equal\n") : printf("\nTensors are not equal\n");
  
  mean = 0.0f;
  for (uint32_t i = samples; i--;) mean += times[i];
  mean /= samples;
  
  qsort(times, samples, sizeof(float), compareFloats);
  median = times[samples >> 1];
  
  printf("Mean: %f ms\n", mean);
  printf("Median: %f ms\n", median);
  
  
  
  cudaFree(dTensorA);
  cudaFree(dTensorB);
  cudaFree(dTensorC);
  cudaFree(dTensorCCublas);
  
  checkCublasStatus(cublasDestroy(handle));

  return 0;
}