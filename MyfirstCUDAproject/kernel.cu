#include<stdio.h>
#include<cuda_runtime.h>

__global__ void add(float* a, float* b, float* c, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) c[i] = a[i] + b[i];
}

int main() {
	const int N = 1000;
	size_t size=N*sizeof(float);

	float* h_a, * h_b, * h_c;
	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	h_c = (float*)malloc(size);

	for (int i = 0; i < N;i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}

	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	add <<< blocksPerGrid, threadsPerBlock >>>(d_a, d_b, d_c, N);

	cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);
	printf("Result[0] = %f\n", h_c[0]);

	free(h_a);free(h_b);free(h_c);
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
	return 0;
}