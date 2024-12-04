/********************************************************************/
/* naive.cu                                                         */
/* ----------------------------                                     */
/* Naive implementation of GEMM                                     */
/* ----------------------------                                     */
/* Author: Mathias Otnes                                            */
/* year:   2024                                                     */
/********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gemm_naive_kernel( float* A_d, float* B_d, float* C_d, int N )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += A_d[row * N + i] * B_d[i * N + col];
        }
        C_d[row * N + col] = sum;
    }
}

void gemm_naive( float* A, float* B, float* C, int N ) 
{

	// Allocate memory on device
	float *A_d, *B_d, *C_d;
	cudaMalloc(&A_d, N * N * sizeof(float));
	cudaMalloc(&B_d, N * N * sizeof(float));
	cudaMalloc(&C_d, N * N * sizeof(float));

	// Copy data to device
	cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	// Kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	// Run kernel
    gemm_naive_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

	// Copy data back to host
	cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory on device
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

}
