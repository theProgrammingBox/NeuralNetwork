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

void initializeNetwork(network* net, const uint32_t batchSize, const uint32_t layers, const uint32_t* parameters, uint32_t *seed1, uint32_t *seed2, bool debug = false) {
  net->batchSize = batchSize;
  net->layers = layers;
  net->parameters = (uint32_t*)malloc((net->layers + 2) * sizeof(uint32_t));
  memcpy(net->parameters, parameters, (net->layers + 2) * sizeof(uint32_t));
  
  net->outputs = (float**)malloc((net->layers + 2) * sizeof(float*));
  net->outputGradients = (float**)malloc((net->layers + 2) * sizeof(float*));
  
  if (debug) printf("Initialize network:\n");
  for (uint32_t i = 0; i < net->layers + 2; i++) {
    checkCudaStatus(cudaMalloc(&net->outputs[i], net->parameters[i] * net->batchSize * sizeof(float)));
    if (debug) printDTensor(net->outputs[i], net->parameters[i], net->batchSize, "output");
    
    checkCudaStatus(cudaMalloc(&net->outputGradients[i], net->parameters[i] * net->batchSize * sizeof(float)));
    if (debug) printDTensor(net->outputGradients[i], net->parameters[i], net->batchSize, "output gradient");
  }
  
  net->weights = (float**)malloc((net->layers + 1) * sizeof(float*));
  net->weightGradients = (float**)malloc((net->layers + 1) * sizeof(float*));
  
  for (uint32_t i = 0; i < net->layers + 1; i++) {
    checkCudaStatus(cudaMalloc(&net->weights[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    fillDTensor(net->weights[i], net->parameters[i + 1] * net->parameters[i], seed1, seed2);
    if (debug) printDTensor(net->weights[i], net->parameters[i + 1], net->parameters[i], "weight");
    
    checkCudaStatus(cudaMalloc(&net->weightGradients[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    if (debug) printDTensor(net->weightGradients[i], net->parameters[i + 1], net->parameters[i], "weight gradient");
  }
  if (debug) printf("\n");
}

void setRandomInput(network* net, uint32_t *seed1, uint32_t *seed2) {
    fillDTensor(net->outputs[0], net->parameters[0] * net->batchSize, seed1, seed2);
}

void setInput(network* net, float* inputs, bool host = true) {
  checkCudaStatus(cudaMemcpy(net->outputs[0], inputs, net->parameters[0] * net->batchSize * sizeof(float), host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}

void forwardPropagate(cublasHandle_t *cublasHandle, network* net, bool debug = false) {
  const float zero = 0.0f;
  if (debug) printf("Forward propagation:\n");
  if (debug) printDTensor(net->outputs[0], net->parameters[0], net->batchSize, "input");
  
  for (uint32_t i = 0; i < net->layers; i++) {
    float alpha = 2.0f / sqrtf(net->parameters[i]);
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      net->parameters[i + 1], net->batchSize, net->parameters[i],
      &alpha,
      net->weights[i], net->parameters[i + 1],
      net->outputs[i], net->parameters[i],
      &zero,
      net->outputs[i + 1], net->parameters[i + 1]));
    if (debug) printDTensor(net->outputs[i + 1], net->parameters[i + 1], net->batchSize, "sum");
    
    reluForward(net->outputs[i + 1], net->parameters[i + 1] * net->batchSize);
    if (debug) printDTensor(net->outputs[i + 1], net->parameters[i + 1], net->batchSize, "relu");
  }
  
  float alpha = 2.0f / sqrtf(net->parameters[net->layers]);
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    net->parameters[net->layers + 1], net->batchSize, net->parameters[net->layers],
    &alpha,
    net->weights[net->layers], net->parameters[net->layers + 1],
    net->outputs[net->layers], net->parameters[net->layers],
    &zero,
    net->outputs[net->layers + 1], net->parameters[net->layers + 1]));
  if (debug) printDTensor(net->outputs[net->layers + 1], net->parameters[net->layers + 1], net->batchSize, "output");
  
  if (debug) printf("\n");
}

void setOutputTarget(cublasHandle_t *cublasHandle, network* net, float* target, bool debug = false) {
  checkCudaStatus(cudaMemcpy(net->outputGradients[net->layers + 1], target, net->parameters[net->layers + 1] * net->batchSize * sizeof(float), cudaMemcpyHostToDevice));
  const float negativeOne = -1.0f;
  checkCublasStatus(cublasSaxpy(
    *cublasHandle,
    net->parameters[net->layers + 1] * net->batchSize,
    &negativeOne,
    net->outputs[net->layers + 1], 1,
    net->outputGradients[net->layers + 1], 1));
  if (debug) printDTensor(net->outputGradients[net->layers + 1], net->parameters[net->layers + 1], net->batchSize, "output gradient");
}

void setOutputGradients(network* net, float* gradients, bool host = true) {
  checkCudaStatus(cudaMemcpy(net->outputGradients[net->layers + 1], gradients, net->parameters[net->layers + 1] * net->batchSize * sizeof(float), host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}

void backPropagate(cublasHandle_t *cublasHandle, network* net, bool errorPrint = false, bool debug = false) {
  if (errorPrint) {
    float error = 0.0f;
    checkCublasStatus(cublasSasum(
      *cublasHandle,
      net->parameters[net->layers + 1] * net->batchSize,
      net->outputGradients[net->layers + 1], 1,
      &error));
    printf("Error: %f\n", error / net->batchSize);
  }
  
  const float zero = 0.0f;
  if (debug) printf("Back propagation:\n");
  
  float alpha = 2.0f / sqrtf(net->batchSize);
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
    net->parameters[net->layers + 1], net->parameters[net->layers], net->batchSize,
    &alpha,
    net->outputGradients[net->layers + 1], net->parameters[net->layers + 1],
    net->outputs[net->layers], net->parameters[net->layers],
    &zero,
    net->weightGradients[net->layers], net->parameters[net->layers + 1]));
  if (debug) printDTensor(net->weightGradients[net->layers], net->parameters[net->layers + 1], net->parameters[net->layers], "weight gradient");
  
  float beta = 2.0f / sqrtf(net->parameters[net->layers + 1]);
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
    net->parameters[net->layers], net->batchSize, net->parameters[net->layers + 1],
    &beta,
    net->weights[net->layers], net->parameters[net->layers + 1],
    net->outputGradients[net->layers + 1], net->parameters[net->layers + 1],
    &zero,
    net->outputGradients[net->layers], net->parameters[net->layers]));
  if (debug) printDTensor(net->outputGradients[net->layers], net->parameters[net->layers], net->batchSize, "output gradient");
  
  for (uint32_t i = net->layers; i > 0; i--) {
    reluBackward(net->outputs[i], net->outputGradients[i], net->parameters[i] * net->batchSize);
    if (debug) printDTensor(net->outputGradients[i], net->parameters[i], net->batchSize, "relu gradient");
    
    alpha = 2.0f / sqrtf(net->batchSize);
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
      net->parameters[i], net->parameters[i - 1], net->batchSize,
      &alpha,
      net->outputGradients[i], net->parameters[i],
      net->outputs[i - 1], net->parameters[i - 1],
      &zero,
      net->weightGradients[i - 1], net->parameters[i]));
    if (debug) printDTensor(net->weightGradients[i - 1], net->parameters[i], net->parameters[i - 1], "weight gradient");
    
    beta = 2.0f / sqrtf(net->parameters[i]);
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      net->parameters[i - 1], net->batchSize, net->parameters[i],
      &beta,
      net->weights[i - 1], net->parameters[i],
      net->outputGradients[i], net->parameters[i],
      &zero,
      net->outputGradients[i - 1], net->parameters[i - 1]));
    if (debug) printDTensor(net->outputGradients[i - 1], net->parameters[i - 1], net->batchSize, "output gradient");
  }
  if (debug) printf("\n");
}

void updateWeights(cublasHandle_t *cublasHandle, network* net, float learningRate, bool debug = false) {
  if (debug) printf("Update weights:\n");
  for (uint32_t i = 0; i < net->layers + 1; i++) {
    checkCublasStatus(cublasSaxpy(
      *cublasHandle,
      net->parameters[i + 1] * net->parameters[i],
      &learningRate,
      net->weightGradients[i], 1,
      net->weights[i], 1));
    if (debug) printDTensor(net->weights[i], net->parameters[i + 1], net->parameters[i], "weight");
  }
  if (debug) printf("\n");
}

void freeNetwork(network* net, bool debug = false) {
  if (debug) printf("Free network\n");
  for (uint32_t i = 0; i < net->layers; i++)
    checkCudaStatus(cudaFree(net->weights[i]));
  
  free(net->weights);
  
  for (uint32_t i = 0; i < net->layers + 1; i++)
    checkCudaStatus(cudaFree(net->outputs[i]));
  
  free(net->outputs);
  
  free(net->parameters);
}

struct player {
  uint32_t idx;
  int32_t score;
};

int comparePlayers(const void *a, const void *b) {
  player *playerA = (player *)a;
  player *playerB = (player *)b;
  return playerA->score - playerB->score;
}

int main() {
  uint32_t seed1, seed2;
  initializeSeeds(&seed1, &seed2);
  
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));
  
  const float learningRate = 0.001f;
  const uint32_t epochs = 100;
  const uint32_t batchSize = 1024;
  
  
  network policy;
  const uint32_t policyParameters[] = {8, 16, 16, 3};
  const uint32_t policyLayers = sizeof(policyParameters) / sizeof(uint32_t) - 2;
  initializeNetwork(&policy, batchSize, policyLayers, policyParameters, &seed1, &seed2);
  
  network value;
  const uint32_t valueParameters[] = {3, 16, 16, 16, 16, 1};
  const uint32_t valueLayers = sizeof(valueParameters) / sizeof(uint32_t) - 2;
  initializeNetwork(&value, batchSize, valueLayers, valueParameters, &seed1, &seed2);
  
  float policyOutput[policyParameters[policyLayers + 1] * batchSize];
  float valueTarget[policyParameters[policyLayers + 1] * batchSize];
  float valueGradient[policyParameters[policyLayers + 1] * batchSize];
  uint32_t actions[batchSize];
  const uint32_t samples = 64;
  const int outcomes[9] = {
    0, -1,  1,
    1,  0, -1,
    -1,  1,  0
  };
  
  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    setRandomInput(&policy, &seed1, &seed2);
    forwardPropagate(&handle, &policy);
    
    setInput(&value, policy.outputs[policyLayers + 1], false);
    forwardPropagate(&handle, &value);
    
    checkCudaStatus(cudaMemcpy(policyOutput, policy.outputs[policyLayers + 1], policyParameters[policyLayers + 1] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint32_t batch = 0; batch < batchSize; batch++) {
      uint32_t action = 0;
      float max = policyOutput[batch * policyParameters[policyLayers + 1]];
      for (uint32_t i = 1; i < policyParameters[policyLayers + 1]; i++) {
        if (policyOutput[batch * policyParameters[policyLayers + 1] + i] > max) {
          max = policyOutput[batch * policyParameters[policyLayers + 1] + i];
          action = i;
        }
      }
      actions[batch] = action;
    }
    
    player players[batchSize];
    for (uint32_t batch = 0; batch < batchSize; batch++) {
      int32_t score = 0;
      uint32_t action = actions[batch];
      for (uint32_t sample = 0; sample < samples; sample++) {
        mixSeed(&seed1, &seed2);
        uint32_t opponentAction = actions[seed1 % batchSize];
        score += outcomes[action * 3 + opponentAction];
      }
      
      players[batch].idx = batch;
      players[batch].score = score;
    }
    
    qsort(players, batchSize, sizeof(player), comparePlayers);
    for (uint32_t batch = 0; batch < batchSize; batch++) {
      valueTarget[players[batch].idx * valueParameters[valueLayers + 1]] = (float)batch / (batchSize - 1);
    }
    
    setOutputTarget(&handle, &value, valueTarget);
    backPropagate(&handle, &value, epoch % 100 == 0);
    updateWeights(&handle, &value, learningRate);
    
    for (uint32_t batch = 0; batch < batchSize; batch++) {
      valueGradient[players[batch].idx * valueParameters[valueLayers + 1]] = 1.0f;
    }
    setOutputGradients(&value, valueGradient);
    backPropagate(&handle, &value);
    setOutputGradients(&policy, value.outputGradients[0], false);
    backPropagate(&handle, &policy);
    updateWeights(&handle, &policy, learningRate);
  }
  freeNetwork(&policy);
  freeNetwork(&value);
  
  
  checkCublasStatus(cublasDestroy(handle));

  return 0;
}