#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <sys/time.h>

void mixSeed(uint32_t* seed1, uint32_t* seed2) {
	*seed1 ^= (*seed2 >> 16) * 0xbf324c81;
	*seed2 ^= (*seed1 >> 16) * 0x9c7493ad;
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
		hash ^= ((hash >> 16) ^ seed2) * 0xC24F8CDB;
		data[idx] = int32_t(hash) * 0.0000000004656612875245797f;
	}
}

__global__ void prefixSum(uint32_t size, float *data, uint32_t jointSize, uint32_t elemOffset) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	uint32_t exp = 1;
	uint32_t rep = ceil(size / float(blockDim.x));
	uint32_t itr = floor(log2f(size));
	uint32_t idx;
	for (uint32_t j = 0; j < itr; j++) {
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - 1;
			if (idx >= size) break;
			data[idx] = data[idx] + data[idx - exp];
		}
		__syncthreads();
		exp <<= 1;
		rep = ceil(rep / 2.0f);
	}
	
	for (uint32_t j = 0; j < itr; j++) {
		exp >>= 1;
		rep <<= 1;	// not correct, but the break should compensate
		for (uint32_t k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - 1 + exp;
			if (idx >= size) break;
			data[idx] = data[idx] + data[idx - exp];
		}
		__syncthreads();
	}
}

void printDataDev(float *dataDev, int size) {
	float *data = (float *)malloc(size * sizeof(float));
	cudaMemcpy(data, dataDev, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		printf("%.2f, ", data[i]);
	}
	printf("\n");
	free(data);
}

int main(int argc, char *argv[])
{
	const uint32_t size = 1025;
	const uint32_t jointSize = 2;
	const uint32_t elemOffset = 1;
	uint32_t seed1, seed2;
	initSeed(&seed1, &seed2);
	
	float *dataDev;
	cudaMalloc(&dataDev, size * sizeof(float));
	fillRand<<<1, 1024>>>(size, dataDev, seed1, seed2);
	mixSeed(&seed1, &seed2);
	
	prefixSum<<<1, 1024>>>(size, dataDev, jointSize, elemOffset);
	printDataDev(dataDev, size);
	cudaFree(dataDev);
	
	return 0;
}