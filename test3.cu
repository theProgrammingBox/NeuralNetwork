#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fillOne(int size, float *data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int itr = ceil(size / float(blockDim.x));
	int idx;
	for (int j = 0; j < itr; j++) {
		idx = blockDim.x * j + i;
		if (idx >= size) break;
		data[idx] = 1;
	}
}

__global__ void prefixSum(int size, float *data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int exp = 1;
	int rep = ceil(size / float(blockDim.x));
	int itr = floor(log2f(size));
	int idx;
	for (int j = 0; j < itr; j++) {
		for (int k = 0; k < rep; k++) {
			idx = (blockDim.x * k + i) * exp * 2 - 1;
			if (idx >= size) break;
			data[idx] = data[idx] + data[idx - exp];
		}
		__syncthreads();
		exp <<= 1;
		rep = ceil(rep / 2.0f);
	}
	
	for (int j = 0; j < itr; j++) {
		exp >>= 1;
		rep <<= 1;	// not correct, but the break should compensate
		for (int k = 0; k < rep; k++) {
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
		printf("%.0f, ", data[i]);
	}
	printf("\n");
	free(data);
}

int main(int argc, char *argv[])
{
	const int size = 1025;
	float *dataDev;
	cudaMalloc(&dataDev, size * sizeof(float));
	fillOne<<<1, 1024>>>(size, dataDev);
	prefixSum<<<1, 1024>>>(size, dataDev);
	printDataDev(dataDev, size);
	cudaFree(dataDev);
	
	return 0;
}