#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include <cudnn.h>
#include <cublas_v2.h>

#define BOARD_SIZE 8
#define VISION_SIZE 3

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 ^= (*seed1 >> 13) * 0x9c7493ad;
    *seed1 ^= (*seed2 >> 17) * 0xbf324c81;
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) mixSeed(seed1, seed2);
}

uint32_t generateRandomUI32(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return *seed1;
}

float generateRandomFloat(uint32_t *seed1, uint32_t *seed2) {
    mixSeed(seed1, seed2);
    return (int32_t)*seed1 * 0.0000000004656612875245797f;
}

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    float* input;
    float* output;
    float* kernel;
    cudaMalloc(&input, 3 * 3 * sizeof(float));
    cudaMalloc(&output, 3 * 3 * sizeof(float));
    cudaMalloc(&kernel, 3 * 3 * sizeof(float));
    
    float h_input[9] = { 
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9 };
    float h_kernel[9] = { 
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6 ,
        0.7, 0.8, 0.9 };
    cudaMemcpy(input, h_input, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1, 3, 3);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1, 3, 3);
    cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 3);

    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    int maxPropagationAlgorithms = 1;
    cudnnConvolutionFwdAlgoPerf_t propagationAlgorithms[1];
    cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, propagationAlgorithms);
    cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm = propagationAlgorithms[0].algo;
    printf("Forward propagation algorithm: %d\n\n", forwardPropagationAlgorithm);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input, kernel_descriptor, kernel, conv_descriptor, forwardPropagationAlgorithm, NULL, 0, &beta, output_descriptor, output);
    
    float h_output[9];
    cudaMemcpy(h_output, output, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint8_t i = 0; i < 9; i++) printf("%f ", h_output[i]);
    printf("\n");
    
    // uint8_t board[BOARD_SIZE * BOARD_SIZE] = {0};
    // uint8_t px, py;
    // uint8_t gx, gy;
    // char move;
    
    // uint32_t seed1, seed2;
    // initializeSeeds(&seed1, &seed2);
    
    // px = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    // py = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    // do {
    //     gx = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    //     gy = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    // } while (gx == px && gy == py);
    // board[py * BOARD_SIZE + px] = 1;
    // board[gy * BOARD_SIZE + gx] = 2;
    // while (true) {
    //     system("clear");
        
    //     for (uint8_t ry = 0; ry < BOARD_SIZE; ry++) {
    //         for (uint8_t rx = 0; rx < BOARD_SIZE; rx++) {
    //             switch (board[ry * BOARD_SIZE + rx]) {
    //                 case 0:
    //                     printf("- ");
    //                     break;
    //                 case 1:
    //                     printf("P ");
    //                     break;
    //                 case 2:
    //                     printf("C ");
    //                     break;
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
        
    //     for (int8_t ry = py - VISION_SIZE; ry <= py + VISION_SIZE; ry++) {
    //         for (int8_t rx = px - VISION_SIZE; rx <= px + VISION_SIZE; rx++) {
    //             switch (board[(ry + BOARD_SIZE) % BOARD_SIZE * BOARD_SIZE + (rx + BOARD_SIZE) % BOARD_SIZE]) {
    //                 case 0:
    //                     printf("- ");
    //                     break;
    //                 case 1:
    //                     printf("P ");
    //                     break;
    //                 case 2:
    //                     printf("C ");
    //                     break;
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
        
    //     board[py * BOARD_SIZE + px] = 0;
        
    //     printf("Move (wasd): ");
    //     scanf(" %c", &move);
    //     px += (move == 'd') - (move == 'a');
    //     py += (move == 's') - (move == 'w');
    //     px = (px + BOARD_SIZE) % BOARD_SIZE;
    //     py = (py + BOARD_SIZE) % BOARD_SIZE;
        
    //     if (px == gx && py == gy) {
    //         board[gy * BOARD_SIZE + gx] = 0;
    //         do {
    //             gx = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    //             gy = generateRandomUI32(&seed1, &seed2) % BOARD_SIZE;
    //         } while (gx == px && gy == py);
    //         board[gy * BOARD_SIZE + gx] = 2;
    //     }
    //     board[py * BOARD_SIZE + px] = 1;
    // }
    
    return 0;
}