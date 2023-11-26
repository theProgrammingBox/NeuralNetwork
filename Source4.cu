#include "Header.cuh"

struct network {
  uint32_t batchSize;
  uint32_t layers;
  uint32_t* parameters;
  float** outputs;
  float** weights;
  float** outputGradients;
  float** weightGradients;
};

void initializeNetwork(network* network0, uint32_t batchSize, uint32_t layers, uint32_t* parameters, uint32_t *seed1, uint32_t *seed2, bool debug = false) {
  network0->batchSize = batchSize;
  network0->layers = layers;
  network0->parameters = (uint32_t*)malloc((network0->layers + 2) * sizeof(uint32_t));
  memcpy(network0->parameters, parameters, (network0->layers + 2) * sizeof(uint32_t));
  
  network0->outputs = (float**)malloc((network0->layers + 2) * sizeof(float*));
  network0->outputGradients = (float**)malloc((network0->layers + 2) * sizeof(float*));
  
  if (debug) printf("Initialize network:\n");
  for (uint32_t i = 0; i < network0->layers + 2; i++) {
    checkCudaStatus(cudaMalloc(&network0->outputs[i], network0->parameters[i] * network0->batchSize * sizeof(float)));
    if (debug) printDTensor(network0->outputs[i], network0->parameters[i], network0->batchSize, "output");
    
    checkCudaStatus(cudaMalloc(&network0->outputGradients[i], network0->parameters[i] * network0->batchSize * sizeof(float)));
    if (debug) printDTensor(network0->outputGradients[i], network0->parameters[i], network0->batchSize, "output gradient");
  }
  
  network0->weights = (float**)malloc((network0->layers + 1) * sizeof(float*));
  network0->weightGradients = (float**)malloc((network0->layers + 1) * sizeof(float*));
  
  for (uint32_t i = 0; i < network0->layers + 1; i++) {
    checkCudaStatus(cudaMalloc(&network0->weights[i], network0->parameters[i + 1] * network0->parameters[i] * sizeof(float)));
    fillDTensor(network0->weights[i], network0->parameters[i + 1] * network0->parameters[i], seed1, seed2);
    if (debug) printDTensor(network0->weights[i], network0->parameters[i + 1], network0->parameters[i], "weight");
    
    checkCudaStatus(cudaMalloc(&network0->weightGradients[i], network0->parameters[i + 1] * network0->parameters[i] * sizeof(float)));
    if (debug) printDTensor(network0->weightGradients[i], network0->parameters[i + 1], network0->parameters[i], "weight gradient");
  }
  if (debug) printf("\n");
}

void forwardPropagate(cublasHandle_t *cublasHandle, network* network0, float* input, bool debug = false) {
  checkCudaStatus(cudaMemcpy(network0->outputs[0], input, network0->parameters[0] * network0->batchSize * sizeof(float), cudaMemcpyHostToDevice));

  const float one = 1.0f;
  const float zero = 0.0f;
  if (debug) printf("Forward propagation:\n");
  if (debug) printDTensor(network0->outputs[0], network0->parameters[0], network0->batchSize, "input");
  
  for (uint32_t i = 0; i < network0->layers; i++) {
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      network0->parameters[i + 1], network0->batchSize, network0->parameters[i],
      &one,
      network0->weights[i], network0->parameters[i + 1],
      network0->outputs[i], network0->parameters[i],
      &zero,
      network0->outputs[i + 1], network0->parameters[i + 1]));
    if (debug) printDTensor(network0->outputs[i + 1], network0->parameters[i + 1], network0->batchSize, "sum");
    
    reluForward(network0->outputs[i + 1], network0->parameters[i + 1] * network0->batchSize);
    if (debug) printDTensor(network0->outputs[i + 1], network0->parameters[i + 1], network0->batchSize, "relu");
  }
  
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    network0->parameters[network0->layers + 1], network0->batchSize, network0->parameters[network0->layers],
    &one,
    network0->weights[network0->layers], network0->parameters[network0->layers + 1],
    network0->outputs[network0->layers], network0->parameters[network0->layers],
    &zero,
    network0->outputs[network0->layers + 1], network0->parameters[network0->layers + 1]));
  if (debug) printDTensor(network0->outputs[network0->layers + 1], network0->parameters[network0->layers + 1], network0->batchSize, "output");
  
  if (debug) printf("\n");
}

void backPropagate(cublasHandle_t *cublasHandle, network* network0, float* target, bool errorPrint = false, bool debug = false) {
  checkCudaStatus(cudaMemcpy(network0->outputGradients[network0->layers + 1], target, network0->parameters[network0->layers + 1] * network0->batchSize * sizeof(float), cudaMemcpyHostToDevice));
  const float negativeOne = -1.0f;
  checkCublasStatus(cublasSaxpy(
    *cublasHandle,
    network0->parameters[network0->layers + 1] * network0->batchSize,
    &negativeOne,
    network0->outputs[network0->layers + 1], 1,
    network0->outputGradients[network0->layers + 1], 1));
  if (debug) printDTensor(network0->outputGradients[network0->layers + 1], network0->parameters[network0->layers + 1], network0->batchSize, "output gradient");
  
  if (errorPrint) {
    float error = 0.0f;
    checkCublasStatus(cublasSasum(
      *cublasHandle,
      network0->parameters[network0->layers + 1] * network0->batchSize,
      network0->outputGradients[network0->layers + 1], 1,
      &error));
    printf("Error: %f\n", error / network0->batchSize);
  }
  
  const float one = 1.0f;
  const float zero = 0.0f;
  if (debug) printf("Back propagation:\n");
  
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
    network0->parameters[network0->layers + 1], network0->parameters[network0->layers], network0->batchSize,
    &one,
    network0->outputGradients[network0->layers + 1], network0->parameters[network0->layers + 1],
    network0->outputs[network0->layers], network0->parameters[network0->layers],
    &zero,
    network0->weightGradients[network0->layers], network0->parameters[network0->layers + 1]));
  if (debug) printDTensor(network0->weightGradients[network0->layers], network0->parameters[network0->layers + 1], network0->parameters[network0->layers], "weight gradient");
  
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
    network0->parameters[network0->layers], network0->batchSize, network0->parameters[network0->layers + 1],
    &one,
    network0->weights[network0->layers], network0->parameters[network0->layers + 1],
    network0->outputGradients[network0->layers + 1], network0->parameters[network0->layers + 1],
    &zero,
    network0->outputGradients[network0->layers], network0->parameters[network0->layers]));
  if (debug) printDTensor(network0->outputGradients[network0->layers], network0->parameters[network0->layers], network0->batchSize, "output gradient");
  
  for (uint32_t i = network0->layers; i > 0; i--) {
    reluBackward(network0->outputs[i], network0->outputGradients[i], network0->parameters[i] * network0->batchSize);
    if (debug) printDTensor(network0->outputGradients[i], network0->parameters[i], network0->batchSize, "relu gradient");
    
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
      network0->parameters[i], network0->parameters[i - 1], network0->batchSize,
      &one,
      network0->outputGradients[i], network0->parameters[i],
      network0->outputs[i - 1], network0->parameters[i - 1],
      &zero,
      network0->weightGradients[i - 1], network0->parameters[i]));
    if (debug) printDTensor(network0->weightGradients[i - 1], network0->parameters[i], network0->parameters[i - 1], "weight gradient");
    
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      network0->parameters[i - 1], network0->batchSize, network0->parameters[i],
      &one,
      network0->weights[i - 1], network0->parameters[i],
      network0->outputGradients[i], network0->parameters[i],
      &zero,
      network0->outputGradients[i - 1], network0->parameters[i - 1]));
    if (debug) printDTensor(network0->outputGradients[i - 1], network0->parameters[i - 1], network0->batchSize, "output gradient");
  }
  if (debug) printf("\n");
}

void updateWeights(cublasHandle_t *cublasHandle, network* network0, float learningRate, bool debug = false) {
  if (debug) printf("Update weights:\n");
  for (uint32_t i = 0; i < network0->layers + 1; i++) {
    checkCublasStatus(cublasSaxpy(
      *cublasHandle,
      network0->parameters[i + 1] * network0->parameters[i],
      &learningRate,
      network0->weightGradients[i], 1,
      network0->weights[i], 1));
    if (debug) printDTensor(network0->weights[i], network0->parameters[i + 1], network0->parameters[i], "weight");
  }
  if (debug) printf("\n");
}

void freeNetwork(network* network0, bool debug = false) {
  if (debug) printf("Free network\n");
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
  
  const float learningRate = 0.1f;
  
  
  network network0;
  uint32_t batchSize = 8;
  uint32_t parameters[] = {2, 3, 1};
  uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 2;
  initializeNetwork(&network0, batchSize, layers, parameters, &seed1, &seed2);
  
  float input[parameters[0] * batchSize];
  float target[parameters[layers + 1] * batchSize];
  for (uint32_t i = 100; i--;) {
    for (uint32_t j = batchSize; j--;) {
      uint8_t a = rand() % 2;
      uint8_t b = rand() % 2;
      uint8_t c = a ^ b;
      input[j * 2 + 0] = a;
      input[j * 2 + 1] = b;
      target[j] = c;
    }
    
    forwardPropagate(&handle, &network0, input);
    backPropagate(&handle, &network0, target, i % 10 == 0);
    updateWeights(&handle, &network0, learningRate);
  }
  freeNetwork(&network0);
  
  
  checkCublasStatus(cublasDestroy(handle));

  return 0;
}