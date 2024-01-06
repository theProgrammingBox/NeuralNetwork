#include <stdio.h>
#include <stdint.h>

#include <cudnn.h>
#include <cublas_v2.h>

int main() {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
  
  cudnnTensorDescriptor_t inputDesc;
  cudnnTensorDescriptor_t outputDesc;
  cudnnFilterDescriptor_t kernelDesc;
  
  cudnnCreateTensorDescriptor(&inputDesc);
  cudnnCreateTensorDescriptor(&outputDesc);
  cudnnCreateFilterDescriptor(&kernelDesc);
  
  cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
  cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 2, 2);
  cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 1, 3, 3);
  
  cudnnConvolutionDescriptor_t convDesc;
  cudnnCreateConvolutionDescriptor(&convDesc);
  cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  
  int maxConvAlgos = 1;
  cudnnConvolutionFwdAlgoPerf_t convFwdAlgos[maxConvAlgos];
  cudnnConvolutionBwdDataAlgoPerf_t convBwdDataAlgos[maxConvAlgos];
  cudnnConvolutionBwdFilterAlgoPerf_t convBwdFilterAlgos[maxConvAlgos];
  cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, maxConvAlgos, &maxConvAlgos, convFwdAlgos);
  cudnnFindConvolutionBackwardDataAlgorithm(cudnn, kernelDesc, outputDesc, convDesc, inputDesc, maxConvAlgos, &maxConvAlgos, convBwdDataAlgos);
  cudnnFindConvolutionBackwardFilterAlgorithm(cudnn, inputDesc, outputDesc, convDesc, kernelDesc, maxConvAlgos, &maxConvAlgos, convBwdFilterAlgos);
  cudnnConvolutionFwdAlgo_t convFwdAlgo = convFwdAlgos[0].algo;
  cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo = convBwdDataAlgos[0].algo;
  cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo = convBwdFilterAlgos[0].algo;
  printf("ConvFwdAlgo: %d\n\n", convFwdAlgo);
  printf("ConvBwdDataAlgo: %d\n\n", convBwdDataAlgo);
  printf("ConvBwdFilterAlgo: %d\n\n", convBwdFilterAlgo);
  
  
  float* dInputTensor;
  float* dOutputTensor;
  float* dKernelTensor;
  
  float* dInputGradTensor;
  float* dOutputGradTensor;
  float* dKernelGradTensor;
  
  cudaMalloc(&dInputTensor, 4 * 4 * 1 * 1 * sizeof(float));
  cudaMalloc(&dOutputTensor, 2 * 2 * 2 * 1 * sizeof(float));
  cudaMalloc(&dKernelTensor, 3 * 3 * 1 * 2 * sizeof(float));
  
  cudaMalloc(&dInputGradTensor, 4 * 4 * 1 * 1 * sizeof(float));
  cudaMalloc(&dOutputGradTensor, 2 * 2 * 2 * 1 * sizeof(float));
  cudaMalloc(&dKernelGradTensor, 3 * 3 * 1 * 2 * sizeof(float));
  
  float hInputTensor[4 * 4 * 1 * 1] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9,10,11,12,
    13,14,15,16
  };
  
  float hKernelTensor[3 * 3 * 1 * 2] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    -1, -2, -3,
    -4, -5, -6,
    -7, -8, -9
  };
  
  float hOutputTensor[2 * 2 * 2 * 1];
  
  float hOutputGradTensor[2 * 2 * 2 * 1] = {
    -1, -2,
    -3, -4,
    1, 2,
    3, 4
  };
    
  float hInputGradTensor[4 * 4 * 1 * 1];
  float hKernelGradTensor[3 * 3 * 1 * 2];
  
  cudaMemcpy(dInputTensor, hInputTensor, 4 * 4 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dKernelTensor, hKernelTensor, 3 * 3 * 1 * 2 * sizeof(float), cudaMemcpyHostToDevice);
  
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnConvolutionForward(
    cudnn,
    &alpha,
    inputDesc, dInputTensor,
    kernelDesc, dKernelTensor,
    convDesc, convFwdAlgo,
    0, 0,
    &beta,
    outputDesc, dOutputTensor);
  
  cudaMemcpy(hOutputTensor, dOutputTensor, 2 * 2 * 2 * 1 * sizeof(float), cudaMemcpyDeviceToHost);
  
  printf("hOutputTensor:\n");
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 2; ++l) {
          printf("%f ", hOutputTensor[i * 2 * 2 * 2 + j * 2 * 2 + k * 2 + l]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  
  cudaMemcpy(dOutputGradTensor, hOutputGradTensor, 2 * 2 * 2 * 1 * sizeof(float), cudaMemcpyHostToDevice);
  
  cudnnConvolutionBackwardData(
    cudnn,
    &alpha,
    kernelDesc, dKernelTensor,
    outputDesc, dOutputGradTensor,
    convDesc, convBwdDataAlgo,
    0, 0,
    &beta,
    inputDesc, dInputGradTensor);
    
    cudnnConvolutionBackwardFilter(
    cudnn,
    &alpha,
    inputDesc, dInputTensor,
    outputDesc, dOutputGradTensor,
    convDesc, convBwdFilterAlgo,
    0, 0,
    &beta,
    kernelDesc, dKernelGradTensor);
  
  cudaMemcpy(hInputGradTensor, dInputGradTensor, 4 * 4 * 1 * 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hKernelGradTensor, dKernelGradTensor, 3 * 3 * 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost);
  
  printf("hInputGradTensor:\n");
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
          printf("%f ", hInputGradTensor[i * 4 * 4 * 1 + j * 4 * 4 + k * 4 + l]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  
  printf("hKernelGradTensor:\n");
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          printf("%f ", hKernelGradTensor[i * 3 * 3 * 1 + j * 3 * 3 + k * 3 + l]);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  
  return 0;
}