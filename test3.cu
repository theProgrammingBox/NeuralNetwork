#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <sys/time.h>

void mixSeed(uint32_t* seed1, uint32_t* seed2) {
	*seed1 ^= (*seed2 >> 13) * 0xbf324c81;
	*seed2 ^= (*seed1 >> 7) * 0x9c7493ad;
}

void initSeed(uint32_t* seed1, uint32_t* seed2) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	*seed1 = tv.tv_sec + 0x4ba1bb47;
	*seed2 = tv.tv_usec + 0xb7ebcb79;
	for (int i = 0; i < 8; i++) {
		mixSeed(seed1, seed2);
	}
}

__global__ void fillRand(uint32_t size, float *data, uint32_t seed1, uint32_t seed2) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t itr = ceil(size / float(blockDim.x));
	uint32_t idx;
	uint32_t hash;
	for (uint32_t j = 0; j < itr; j++) {
		idx = blockDim.x * j + i;
		if (idx >= size) break;
		hash = (idx ^ seed1) * 0xEDEF0EA3;
		hash ^= ((hash >> 13) ^ seed2) * 0xC24F8CDB;
		hash ^= hash << 17;
		data[idx] = int32_t(hash) * 0.0000000004656612875245797f;
		// data[idx] = -1;
	}
}

void fillRand(uint32_t size, float *inputDev, uint32_t* seed1, uint32_t* seed2) {
	fillRand<<<1, 1024>>>(size, inputDev, *seed1, *seed2);
	mixSeed(seed1, seed2);
}

void printMatrices(float *inputDev, int size, int n, uint32_t jointSize, uint32_t elemOffset, uint32_t w, uint32_t h) {
	float *data = (float *)malloc(size * sizeof(float));
	cudaMemcpy(data, inputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t pair = 0; pair < n; pair++) {
		uint32_t max_height = h > w ? h : w;
		for (uint32_t row = 0; row < max_height; row++) {
			if (row < w) for (uint32_t col = 0; col < w; col++) printf("%.2f, ", data[pair * jointSize + row * w + col]);
			else for (uint32_t col = 0; col < w; col++) printf("      ");
			printf("\t");
			if (row < h) for (uint32_t col = 0; col < w; col++) printf("%.2f, ", data[pair * jointSize + elemOffset + row * w + col]);
			printf("\n");
		}
		printf("\n");
	}
	free(data);
}

__global__ void inputDependentScan(uint32_t size, float *data, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	uint32_t exp = jointSize;
	uint32_t rep = ceil(elems / float(blockDim.x));
	uint32_t itr = floor(log2f(elems));
	uint32_t idx;
	for (uint32_t j = 0; j < itr; j++) {
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - jointSize;
			if (idx >= size) break;	// if want to check true out of bounds as in not allocated, you need seperate error handling or idx + jointSize - 1 < size
			data[idx] = data[idx - exp] * data[idx + elemOffset] + data[idx];
			data[idx + elemOffset] = data[idx + elemOffset] * data[idx - exp + elemOffset];
		}
		__syncthreads();
		exp <<= 1;
		rep = ceil(rep / 2.0f);
	}
	
	for (uint32_t j = 0; j < itr; j++) {
		exp >>= 1;
		rep <<= 1;	// not correct, but the break should compensate for the extra iterations due to ceil nonlinearity
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - jointSize + exp;
			if (idx >= size) break;
			data[idx] = data[idx - exp] * data[idx + elemOffset] + data[idx];
			data[idx + elemOffset] = data[idx + elemOffset] * data[idx - exp + elemOffset];
		}
		__syncthreads();
	}
}
void inputDependentScan(uint32_t size, float *inputDev, float *outputDev, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	cudaMemcpy(outputDev, inputDev, size * sizeof(float), cudaMemcpyDeviceToDevice);
	inputDependentScan<<<1, 1024>>>(size, outputDev, elems, jointSize, elemOffset);
}

void checkError(uint32_t size, float *inputDev, float *outputDev, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	float *input = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	cudaMemcpy(input, inputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (uint32_t i = 1; i < elems; i++) {
		uint32_t idx = i * jointSize;
		float res = input[idx - jointSize] * input[idx + elemOffset] + input[idx];
		input[idx] = res;
		if (fabs(res - output[idx]) > 0.0001f) {
			printf("%f != %f\n", res, output[idx]);
			break;
		}
		printf("%f == %f\n", res, output[idx]);
		// printf("input: %f, output: %f, res: %f\n", input[idx], output[idx], res);
	}
	free(input);
	free(output);
}

int main(int argc, char *argv[])
{
	const uint32_t width = 2;
	const uint32_t height = 3;
	const uint32_t elems = 1025;
	const uint32_t elemOffset = width * width;	// weight
	const uint32_t jointSize = elemOffset + width * height;	// weight + bias
	const uint32_t size = elems * jointSize;
	uint32_t seed1, seed2;
	initSeed(&seed1, &seed2);
	
	float *inputDev, *outputDev;
	cudaMalloc(&inputDev, size * sizeof(float));
	cudaMalloc(&outputDev, size * sizeof(float));
	fillRand(size, inputDev, &seed1, &seed2);
	// printMatrices(inputDev, size, elems, jointSize, elemOffset, width, height);
	
	inputDependentScan(size, inputDev, outputDev, elems, jointSize, elemOffset);
	printMatrices(outputDev, size, elems, jointSize, elemOffset, width, height);
	// checkError(size, inputDev, outputDev, elems, jointSize, elemOffset);
	cudaFree(inputDev);
	cudaFree(outputDev);
	
	return 0;
}