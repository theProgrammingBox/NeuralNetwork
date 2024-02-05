#include <stdio.h>

#include <cuda_runtime.h>

__global__ void fillOne(int size, float *data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) return;
	data[i] = 1;
}

__global__ void prefixSum(int size, float *data, int itr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int threads = blockDim.x;
	int temp;
	int exp = 1;
	for (int j = 0; j < itr; j++) {
		for (int k = 0; k < size; k++) {
			temp = threads * k + i * exp * 2 - 1;
			if (temp < size) {
				data[temp] = data[temp] + data[temp - exp];
			}
		}
		__syncthreads();
		exp *= 2;
	}
	
	for (int j = 0; j < itr; j++) {
		exp >>= 1;
		temp = i * exp * 2 - 1 + exp;
		if (temp < size) {
			data[temp] = data[temp] + data[temp - exp];
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
	const int size = 100;
	float *dataDev;
	
	cudaMalloc(&dataDev, size * sizeof(float));
	
	fillOne<<<1, 16>>>(size, dataDev);
	
	prefixSum<<<1, 16>>>(size, dataDev, floor(log2(size)));
	printDataDev(dataDev, size);
	printf("floor(log2(size)) = %.0f\n", floor(log2(size)));
	
	cudaFree(dataDev);
	
	return 0;
}