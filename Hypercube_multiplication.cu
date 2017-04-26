#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<device_functions.h>
#include<stdio.h>
#include<Windows.h>
#include<string.h>

#define n 2
__device__ int getGlobalIdx_1D_2D() {
	return blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
}

__global__ void hypercubeMultiplication(int *C, const int *A, const int *B) {
	int world_rank = getGlobalIdx_1D_2D();
	int i;
	__shared__ int a[n*n], b[n*n], p[n*n];
	int x[n], y[n], c[n];
	memset(c, 0, n * sizeof(int));
	memset(p, 0, n * n * sizeof(int));
	//initialize
	if(world_rank==0)
	for (i = 0; i<n*n; i++) {
		a[i] = A[i];
		b[i] = B[i];
	}
	__syncthreads();

	//step 3 of the algorithm
	for (i = 0; i < n; i++) {
		y[i] = b[i + threadIdx.y*n];
	
	}
	__syncthreads();

		//step 4 of the algorithm
	for(i = 0; i < n; i++)
		x[i] = a[threadIdx.x*n + i];
	__syncthreads();

	//calculate the product
	for (i = 0; i < n; i++) {
		c[i] = x[i] * y[i];
		p[world_rank] += c[i];

	}
	__syncthreads();

	//send the values to the global memory
	if((world_rank%(n+1))==0)
		for (i = 0; i < n; i++) {
			C[threadIdx.y * n + i] = p[i*n + threadIdx.x];
		}
	__syncthreads();
	
}

int main(int argc, char **argv) {
	int a1[n*n], b11[n*n], b1[n*n], c1[n*n];
	int *A, *B, *C;
	int i,j;

	memset(a1, 0, sizeof(a1));
	memset(b1, 0, sizeof(a1));
	memset(c1, 0, sizeof(a1));

	cudaMalloc((void**)&A, sizeof(a1));
	cudaMalloc((void**)&B, sizeof(b1));
	cudaMalloc((void**)&C, sizeof(c1));

	printf("Enter the values of the matrices A and B:\n");

	for (i = 0; i < n*n; i++) {
		scanf("%d", &a1[i]);
	}
	for (i = 0; i < n*n; i++) {
		scanf("%d", &b11[i]);
	}
	for (i = 0; i < n; i++) {								//transpose the matrix B
			for (j = 0; j < n; j++) {
				b1[i*n + j] = b11[i + j*n];
		}
	}

	cudaMemcpy(A, a1, sizeof(a1), cudaMemcpyHostToDevice);
	cudaMemcpy(B, b1, sizeof(a1), cudaMemcpyHostToDevice);

	dim3 nThreads(n, n);
	hypercubeMultiplication << <1, nThreads >> > (C, A, B);

	cudaMemcpy(c1, C, sizeof(c1), cudaMemcpyDeviceToHost);
	printf("The result is:\n");
	for (i = 0; i < n*n; i++) {
		printf("%d ", c1[i]);
		if ((i + 1) % n == 0)
			printf("\n");
	}

	Sleep(20000);
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	return 1;
}