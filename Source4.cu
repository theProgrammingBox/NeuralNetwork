#include <stdio.h>
#include <cublas_v2.h>

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CuBLAS error: %d\n", status);
    exit(1); 
  }
}

int main() {

  const int widthA = 32;
  const int heightA = 32;
  const int widthC = 64;
  
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));

  // Rest of code

  checkCublasStatus(cublasDestroy(handle));

  return 0;
}