#include "Header.cuh"

/*
TASKS:
- maybe reimplement everything cuz that custom rand fill bug could have really messed up past results
- add bias at least for first layer
*/

struct network {
  float meanCorrection;
  float varianceCorrection;
  uint32_t batchSize;
  uint32_t layers;
  uint32_t* parameters;
  float** outputs;
  float** weights;
  float** outputGradients;
  float** weightGradients;
  float** weightGradientsMean;
  float** weightGradientsVariance;
};

void initializeNetwork(network* net, const uint32_t batchSize, const uint32_t layers, uint32_t* parameters, uint32_t *seed1, uint32_t *seed2, bool debug = false) {
  net->batchSize = batchSize;
  net->layers = layers;
  net->parameters = parameters;
  
  net->meanCorrection = 1.0f;
  net->varianceCorrection = 1.0f;
  
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
  net->weightGradientsMean = (float**)malloc((net->layers + 1) * sizeof(float*));
  net->weightGradientsVariance = (float**)malloc((net->layers + 1) * sizeof(float*));
  
  for (uint32_t i = 0; i < net->layers + 1; i++) {
    checkCudaStatus(cudaMalloc(&net->weights[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    fillDTensor(net->weights[i], net->parameters[i + 1] * net->parameters[i], seed1, seed2);
    if (debug) printDTensor(net->weights[i], net->parameters[i + 1], net->parameters[i], "weight");
    
    checkCudaStatus(cudaMalloc(&net->weightGradients[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    if (debug) printDTensor(net->weightGradients[i], net->parameters[i + 1], net->parameters[i], "weight gradient");
    
    checkCudaStatus(cudaMalloc(&net->weightGradientsMean[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    customFillDTensorConstant(net->weightGradientsMean[i], net->parameters[i + 1] * net->parameters[i], 0.0f);
    if (debug) printDTensor(net->weightGradientsMean[i], net->parameters[i + 1], net->parameters[i], "weight gradient mean");
    
    checkCudaStatus(cudaMalloc(&net->weightGradientsVariance[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    customFillDTensorConstant(net->weightGradientsVariance[i], net->parameters[i + 1] * net->parameters[i], 0.0f);
    if (debug) printDTensor(net->weightGradientsVariance[i], net->parameters[i + 1], net->parameters[i], "weight gradient variance");
  }
  if (debug) printf("\n");
}

void setRandomInput(network* net, uint32_t *seed1, uint32_t *seed2) {
    fillDTensor(net->outputs[0], net->parameters[0] * net->batchSize, seed1, seed2);
}

void setCustomRandomInput(network* net, uint32_t *seed1, uint32_t *seed2) {
  customFillDTensor(net->outputs[0], net->parameters[0] * net->batchSize, net->parameters[0], seed1, seed2);
}

void setInput(network* net, float* inputs, bool host = true) {
  checkCudaStatus(cudaMemcpy(net->outputs[0], inputs, net->parameters[0] * net->batchSize * sizeof(float), host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}

void forwardPropagate(cublasHandle_t *cublasHandle, network* net, bool debug = false) {
  const float zero = 0.0f;
  if (debug) printf("Forward propagation:\n");
  if (debug) printDTensor(net->outputs[0], net->parameters[0], net->batchSize, "input");
  
  for (uint32_t i = 0; i < net->layers; i++) {
    // float alpha = 2.0f / sqrtf(net->parameters[i]);
    float alpha = 1.0f;
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
  
  // float alpha = 2.0f / sqrtf(net->parameters[net->layers]);
  float alpha = 1.0f;
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

void setOutputGradientsToConstant(network* net, float constant) {
  customFillDTensorConstant(net->outputGradients[net->layers + 1], net->parameters[net->layers + 1] * net->batchSize, constant);
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
  
  if (debug) printDTensor(net->outputGradients[net->layers + 1], net->parameters[net->layers + 1], net->batchSize, "output gradient");
  
  // float alpha = 2.0f / sqrtf(net->batchSize);
  float alpha = 1.0f;
  checkCublasStatus(cublasSgemm(
    *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
    net->parameters[net->layers + 1], net->parameters[net->layers], net->batchSize,
    &alpha,
    net->outputGradients[net->layers + 1], net->parameters[net->layers + 1],
    net->outputs[net->layers], net->parameters[net->layers],
    &zero,
    net->weightGradients[net->layers], net->parameters[net->layers + 1]));
  if (debug) printDTensor(net->weightGradients[net->layers], net->parameters[net->layers + 1], net->parameters[net->layers], "weight gradient");
  
  // float beta = 2.0f / sqrtf(net->parameters[net->layers + 1]);
  float beta = 1.0f;
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
    
    // alpha = 2.0f / sqrtf(net->batchSize);
    alpha = 1.0f;
    checkCublasStatus(cublasSgemm(
      *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
      net->parameters[i], net->parameters[i - 1], net->batchSize,
      &alpha,
      net->outputGradients[i], net->parameters[i],
      net->outputs[i - 1], net->parameters[i - 1],
      &zero,
      net->weightGradients[i - 1], net->parameters[i]));
    if (debug) printDTensor(net->weightGradients[i - 1], net->parameters[i], net->parameters[i - 1], "weight gradient");
    
    // beta = 2.0f / sqrtf(net->parameters[i]);
    beta = 1.0f;
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

void updateWeights(cublasHandle_t *cublasHandle, network* net, float betaMean, float betaVar, float learningRate, float weightDecay, bool debug = false) {
  if (debug) printf("Update weights:\n");
  net->meanCorrection *= betaMean;
  net->varianceCorrection *= betaVar;
  for (uint32_t i = 0; i < net->layers + 1; i++) {
    integratedAdamUpdate(net->weights[i], net->weightGradients[i], net->weightGradientsMean[i], net->weightGradientsVariance[i], betaMean, betaVar, net->meanCorrection, net->varianceCorrection, learningRate, weightDecay, net->parameters[i + 1] * net->parameters[i]);
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
}

int main() {
  uint32_t seed1, seed2;
  initializeSeeds(&seed1, &seed2);
  
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));
  
  const uint32_t epochs = 4096 * 4;
  const uint32_t batchSize = 4096;
  const float meanBeta = 0.9f;
  const float varianceBeta = 0.999f;
  const float weightDecay = 0.1f;
  const float policyLearningRate = 0.001f;
  const float valueLearningRate = 0.001f;
  
  network policy;
  uint32_t policyParameters[] = {2, 8, 8, 1};
  const uint32_t policyLayers = sizeof(policyParameters) / sizeof(uint32_t) - 2;
  initializeNetwork(&policy, batchSize, policyLayers, policyParameters, &seed1, &seed2);
  
  network value;
  uint32_t valueParameters[] = {2, 8, 8, 1};
  const uint32_t valueLayers = sizeof(valueParameters) / sizeof(uint32_t) - 2;
  initializeNetwork(&value, batchSize, valueLayers, valueParameters, &seed1, &seed2);
  
  float policyOutput[policyParameters[policyLayers + 1] * batchSize];
  float valueTarget[policyParameters[policyLayers + 1] * batchSize];
  uint32_t actions[batchSize];
  const int outcomes[9] = {
    0, -1,  1,
    1,  0, -1,
    -1,  1,  0
  };
  
  float policyTargetOutGrad[batchSize];
  float valueInput[valueParameters[0] * batchSize];
  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    setCustomRandomInput(&policy, &seed1, &seed2);
    forwardPropagate(&handle, &policy);
    
    float policyOutput[policyParameters[policyLayers + 1] * batchSize];
    checkCudaStatus(cudaMemcpy(policyOutput, policy.outputs[policyLayers + 1], policyParameters[policyLayers + 1] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    float err = 0;
    for (uint32_t batch = 0; batch < batchSize; batch++) {
      float in = policyOutput[batch];
      valueInput[batch * 2] = in;
      valueInput[batch * 2 + 1] = 1.0f;
      float temp = (0.2f - in);
      valueTarget[batch] = -temp * temp;
    }
    
    setInput(&value, valueInput);
    forwardPropagate(&handle, &value);
      setOutputTarget(&handle, &value, valueTarget);
      backPropagate(&handle, &value, epoch % 1024 == 0);
      updateWeights(&handle, &value, meanBeta, varianceBeta, valueLearningRate, weightDecay);
    
      setOutputGradientsToConstant(&value, 1.0f);
      backPropagate(&handle, &value);
      setOutputGradients(&policy, value.outputGradients[0], false);
      backPropagate(&handle, &policy);
        updateWeights(&handle, &policy, meanBeta, varianceBeta, policyLearningRate, weightDecay);
  }
  printf("\n");
  
  float policyInput[policyParameters[0] * batchSize];
  for (uint32_t point = 0; point < 21; point++) {
    policyInput[0] = 1;
    policyInput[1] = ((float)point- 10) / 10.0f;
    policyInput[2] = 1.0f;
    setInput(&policy, policyInput);
    forwardPropagate(&handle, &policy);
    
    float policyOutput[policyParameters[policyLayers + 1] * batchSize];
    checkCudaStatus(cudaMemcpy(policyOutput, policy.outputs[policyLayers + 1], policyParameters[policyLayers + 1] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    float input;
    float valuet;
    
    float in = policyOutput[0];
    float temp = (0.2f - in);
    
    valueInput[0] = in;
    valueInput[1] = 1.0f;
    setInput(&value, valueInput);
    forwardPropagate(&handle, &value);
    setOutputGradientsToConstant(&value, 1.0f);
    backPropagate(&handle, &value);
    
    float valueGradient[valueParameters[0] * batchSize];
    checkCudaStatus(cudaMemcpy(valueGradient, value.outputGradients[0], valueParameters[0] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("In: %f, Out: %f, Score: %f, Grad: %f\n", policyInput[1], in, -temp * temp, valueGradient[0]);    
  }
  printf("\n");
  
  for (uint32_t point = 0; point < 21; point++) {
    valueInput[0] = ((float)point-0) / (20.0f / 0.3f);
    setInput(&value, valueInput);
    forwardPropagate(&handle, &value);
    
    float valueOutput[valueParameters[valueLayers + 1] * batchSize];
    checkCudaStatus(cudaMemcpy(valueOutput, value.outputs[valueLayers + 1], valueParameters[valueLayers + 1] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    float valueGradient[valueParameters[valueLayers + 1] * batchSize];
    
    setOutputGradientsToConstant(&value, 1.0f);
    backPropagate(&handle, &value);
    
    float valueInputGradient[valueParameters[valueLayers + 1] * batchSize];
    checkCudaStatus(cudaMemcpy(valueInputGradient, value.outputGradients[0], valueParameters[valueLayers + 1] * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    float in = valueInput[0];
    float temp = (0.2f - in);
    printf("In: %f, Out: %f, expected: %f, Grad: %f\n", in, valueOutput[0], -temp * temp, valueInputGradient[0]);
  }
  
  freeNetwork(&policy);
  freeNetwork(&value);
  
  
  checkCublasStatus(cublasDestroy(handle));

  return 0;
}