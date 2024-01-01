struct network {
  uint32_t batchSize;
  uint32_t layers;
  float meanCorrection;
  float varianceCorrection;
  uint32_t* parameters;
  float** outputs;
  float** outputGradients;
  float** weights;
  float** weightGradients;
  float** weightGradientsMean;
  float** weightGradientsVariance;
};

void initializeNetwork(network* net, uint32_t batchSize, uint32_t layers, int32_t* parameters, uint32_t *seed1, uint32_t *seed2, bool debug) {
  if (debug) printf("Initialize network:\n");
  net->batchSize = batchSize;
  net->layers = layers;
  net->parameters = parameters;
  net->meanCorrection = 1.0f;
  net->varianceCorrection = 1.0f;
  
  net->outputs = (float**)malloc((net->layers + 2) * sizeof(float*));
  net->outputGradients = (float**)malloc((net->layers + 2) * sizeof(float*));
  for (uint32_t i = 0; i < net->layers + 2; i++) {
    checkCudaStatus(cudaMalloc(&net->outputs[i], net->parameters[i] * net->batchSize * sizeof(float)));
    checkCudaStatus(cudaMalloc(&net->outputGradients[i], net->parameters[i] * net->batchSize * sizeof(float)));
    if (debug) printTensor(net->outputs[i], net->parameters[i], net->batchSize, "output", true);
    if (debug) printTensor(net->outputGradients[i], net->parameters[i], net->batchSize, "output gradient", true);
  }
  
  net->weights = (float**)malloc((net->layers + 1) * sizeof(float*));
  net->weightGradients = (float**)malloc((net->layers + 1) * sizeof(float*));
  net->weightGradientsMean = (float**)malloc((net->layers + 1) * sizeof(float*));
  net->weightGradientsVariance = (float**)malloc((net->layers + 1) * sizeof(float*));
  for (uint32_t i = 0; i < net->layers + 1; i++) {
    checkCudaStatus(cudaMalloc(&net->weights[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    checkCudaStatus(cudaMalloc(&net->weightGradients[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    checkCudaStatus(cudaMalloc(&net->weightGradientsMean[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    checkCudaStatus(cudaMalloc(&net->weightGradientsVariance[i], net->parameters[i + 1] * net->parameters[i] * sizeof(float)));
    
    fillDTensorRandom(net->weights[i], net->parameters[i + 1] * net->parameters[i], seed1, seed2);
    fillDTensorConstant(net->weightGradientsMean[i], net->parameters[i + 1] * net->parameters[i], 0.0f);
    fillDTensorConstant(net->weightGradientsVariance[i], net->parameters[i + 1] * net->parameters[i], 0.0f);
    
    if (debug) printTensor(net->weights[i], net->parameters[i + 1], net->parameters[i], "weight");
    if (debug) printTensor(net->weightGradients[i], net->parameters[i + 1], net->parameters[i], "weight gradient");
    if (debug) printTensor(net->weightGradientsMean[i], net->parameters[i + 1], net->parameters[i], "weight gradient mean");
    if (debug) printTensor(net->weightGradientsVariance[i], net->parameters[i + 1], net->parameters[i], "weight gradient variance");
  }
  if (debug) printf("\n");
}

void setRandomInput(network* net, uint32_t *seed1, uint32_t *seed2) {
    fillDTensorRandom(net->outputs[0], net->parameters[0] * net->batchSize, seed1, seed2);
}

void setCustomRandomInput(network* net, uint32_t *seed1, uint32_t *seed2) {
  fillDTensorCustomRandom(net->outputs[0], net->parameters[0] * net->batchSize, seed1, seed2);
}

void setInput(network* net, float* inputs, bool host = true) {
  checkCudaStatus(cudaMemcpy(net->outputs[0], inputs, net->parameters[0] * net->batchSize * sizeof(float), host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}
