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

__global__ void prefixSum(uint32_t size, float *data, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	uint32_t exp = jointSize;
	uint32_t rep = ceil(elems / float(blockDim.x));
	uint32_t itr = floor(log2f(elems));
	uint32_t idx;
	for (uint32_t j = 0; j < itr; j++) {
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - jointSize;
			// alter if to represent block level out of bounds
			if (idx >= size) break;	// 0th index, doesn't really check out of bounds in the element level, checks on the block leve;
			data[idx] = data[idx - exp] * data[idx + elemOffset] + data[idx];
			data[idx + elemOffset] = data[idx + elemOffset] * data[idx - exp + elemOffset];
		}
		__syncthreads();
		exp <<= 1;
		rep = ceil(rep / 2.0f);
	}
	
	for (uint32_t j = 0; j < itr; j++) {
		exp >>= 1;
		rep <<= 1;	// not correct, but the break should compensate
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - jointSize + exp;
			if (idx >= size) break;	// if want to check true out of bounds as in not allocated, you need seperate error handling or idx + jointSize - 1 < size
			data[idx] = data[idx - exp] * data[idx + elemOffset] + data[idx];
			data[idx + elemOffset] = data[idx + elemOffset] * data[idx - exp + elemOffset];
		}
		__syncthreads();
	}
}
void prefixSum(uint32_t size, float *inputDev, float *outputDev, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	cudaMemcpy(outputDev, inputDev, size * sizeof(float), cudaMemcpyDeviceToDevice);
	prefixSum<<<1, 1024>>>(size, outputDev, elems, jointSize, elemOffset);
}

void printDataDev(float *inputDev, int size, int elems, uint32_t jointSize) {
	float *data = (float *)malloc(size * sizeof(float));
	cudaMemcpy(data, inputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < elems; i++) {
		for (int j = 0; j < jointSize; j++) {
			printf("%f, ", data[i * jointSize + j]);
		}
		printf("\n");
	}
	printf("\n");
	free(data);
}

void checkError(uint32_t size, float *inputDev, float *outputDev, uint32_t elems, uint32_t jointSize, uint32_t elemOffset) {
	float *input = (float *)malloc(size * sizeof(float));
	float *output = (float *)malloc(size * sizeof(float));
	cudaMemcpy(input, inputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	// recursive check x[i] = x[i-1] * x'[i] + x[i]
	for (uint32_t i = 1; i < elems; i++) {
		uint32_t idx = i * jointSize;
		float res = input[idx - jointSize] * input[idx + elemOffset] + input[idx];
		input[idx] = res;
		if (fabs(res - output[idx]) > 0.0001f) {
			printf("%f != %f\n", res, output[idx]);
			break;
		} else {
			printf("%f == %f\n", res, output[idx]);
		}
		// printf("input: %f, output: %f, res: %f\n", input[idx], output[idx], res);
	}
	free(input);
	free(output);
}

int main(int argc, char *argv[])
{
	const uint32_t elems = 1025;
	const uint32_t jointSize = 2;
	const uint32_t elemOffset = 1;
	const uint32_t size = elems * jointSize;
	uint32_t seed1, seed2;
	initSeed(&seed1, &seed2);
	
	float *inputDev, *outputDev;
	cudaMalloc(&inputDev, size * sizeof(float));
	cudaMalloc(&outputDev, size * sizeof(float));
	fillRand(size, inputDev, &seed1, &seed2);
	
	prefixSum(size, inputDev, outputDev, elems, jointSize, elemOffset);
	// printDataDev(outputDev, size, elems, jointSize);
	checkError(size, inputDev, outputDev, elems, jointSize, elemOffset);
	cudaFree(inputDev);
	cudaFree(outputDev);
	
	return 0;
}