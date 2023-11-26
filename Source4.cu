#include "Header.cuh"

struct network {
  uint32_t batchSize;
  uint32_t layers;
  uint32_t* parameters;
  float** outputs;
  float** weights;
};

void initializeNetwork(network* network0, uint32_t batchSize, uint32_t layers, uint32_t* parameters, uint32_t *seed1, uint32_t *seed2) {
  network0->batchSize = batchSize;
  network0->layers = layers;
  network0->parameters = (uint32_t*)malloc((network0->layers + 1) * sizeof(uint32_t));
  memcpy(network0->parameters, parameters, (network0->layers + 1) * sizeof(uint32_t));
  
  network0->outputs = (float**)malloc((network0->layers + 1) * sizeof(float*));
  
  printf("Initialize network\n");
  for (uint32_t i = 0; i < network0->layers + 1; i++) {
    checkCudaStatus(cudaMalloc(&network0->outputs[i], network0->parameters[i] * network0->batchSize * sizeof(float)));
    printDTensor(network0->outputs[i], network0->parameters[i], network0->batchSize, "outputs");
  }
  
  network0->weights = (float**)malloc(network0->layers * sizeof(float*));
  
  for (uint32_t i = 0; i < network0->layers; i++) {
    checkCudaStatus(cudaMalloc(&network0->weights[i], network0->parameters[i + 1] * network0->parameters[i] * sizeof(float)));
    fillDTensor(network0->weights[i], network0->parameters[i + 1] * network0->parameters[i], seed1, seed2);
    printDTensor(network0->weights[i], network0->parameters[i + 1], network0->parameters[i], "weights");
  }
  printf("\n");
}

void feedNetwork(cublasHandle_t *cublasHandle, network* network0, float* inputs) {
  checkCudaStatus(cudaMemcpy(network0->outputs[0], inputs, network0->parameters[0] * network0->batchSize * sizeof(float), cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  printf("Feed forward\n");
  printDTensor(network0->outputs[0], network0->parameters[0], network0->batchSize, "outputs");
  for (uint32_t i = 0; i < network0->layers; i++) {
    checkCublasStatus(cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, network0->parameters[i + 1], network0->batchSize, network0->parameters[i], &alpha, network0->weights[i], network0->parameters[i + 1], network0->outputs[i], network0->parameters[i], &beta, network0->outputs[i + 1], network0->parameters[i + 1]));
    printDTensor(network0->outputs[i + 1], network0->parameters[i + 1], network0->batchSize, "outputs");
  }
  printf("\n");
}

void freeNetwork(network* network0) {
  printf("Free network\n");
  for (uint32_t i = 0; i < network0->layers; i++)
    checkCudaStatus(cudaFree(network0->weights[i]));
  
  free(network0->weights);
  
  for (uint32_t i = 0; i < network0->layers + 1; i++)
    checkCudaStatus(cudaFree(network0->outputs[i]));
  
  free(network0->outputs);
  
  free(network0->parameters);
}

int main() {
  uint32_t seed1, seed2;
  initializeSeeds(&seed1, &seed2);
  
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));
  
  
  network network0;
  uint32_t batchSize = 2;
  uint32_t layers = 2;
  uint32_t parameters[layers + 1] = {2, 3, 1};
  initializeNetwork(&network0, batchSize, layers, parameters, &seed1, &seed2);
  
  float inputs[batchSize * parameters[0]] = {1.0f, 2.0f, 3.0f, 4.0f};
  feedNetwork(&handle, &network0, inputs);
  
  freeNetwork(&network0);
  
  
  checkCublasStatus(cublasDestroy(handle));

  return 0;
}