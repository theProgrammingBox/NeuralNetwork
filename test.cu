#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Tensor dimensions
    int batchSize = 1, channels = 1, height = 1, width = 10;

    // Host data for forward pass
    std::vector<float> h_input(width);
    for (int i = 0; i < width; ++i) {
        h_input[i] = i; // Incrementally increase the input data
    }

    // Device data for forward pass
    float *d_input, *d_output;
    cudaMalloc(&d_input, batchSize * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, batchSize * channels * height * width * sizeof(float));

    // Copy host data to device for forward pass
    cudaMemcpy(d_input, h_input.data(), batchSize * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Tensor descriptor
    cudnnTensorDescriptor_t inOutTensorDesc;
    cudnnCreateTensorDescriptor(&inOutTensorDesc);
    cudnnSetTensor4dDescriptor(inOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, channels, height, width);

    // Softmax forward
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, inOutTensorDesc, d_input,
                        &beta, inOutTensorDesc, d_output);

    // Copy device data back to host for forward pass
    std::vector<float> h_output(width);
    cudaMemcpy(h_output.data(), d_output, batchSize * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output tensor
    std::cout << "Softmax Output:" << std::endl;
    for (int i = 0; i < width; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Allocate memory for the gradient tensors
    float *d_input_grad, *d_output_grad;
    cudaMalloc(&d_input_grad, batchSize * channels * height * width * sizeof(float));
    cudaMalloc(&d_output_grad, batchSize * channels * height * width * sizeof(float));

    // Initialize the output gradient with alternating i and -i
    std::vector<float> h_output_grad(width);
    for (int i = 0; i < width; ++i) {
        h_output_grad[i] = (i % 2 == 0) ? i : -i;
    }

    // Copy host output gradient data to device
    cudaMemcpy(d_output_grad, h_output_grad.data(), batchSize * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Softmax backward
    cudnnSoftmaxBackward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                         &alpha, inOutTensorDesc, d_output, // softmax output as input
                         inOutTensorDesc, d_output_grad,    // gradient w.r.t softmax output
                         &beta, inOutTensorDesc, d_input_grad); // gradient w.r.t softmax input

    // Copy device input gradient data back to host
    std::vector<float> h_input_grad(width);
    cudaMemcpy(h_input_grad.data(), d_input_grad, batchSize * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the input gradient tensor
    std::cout << "Softmax Gradient:" << std::endl;
    for (int i = 0; i < width; ++i) {
        std::cout << h_input_grad[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_grad);
    cudaFree(d_output_grad);
    cudnnDestroyTensorDescriptor(inOutTensorDesc);
    cudnnDestroy(cudnn);

    return 0;
}
