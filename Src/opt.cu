/****************************************************************************************/
/* opt.cu                                                                               */
/* --------------------------------                                                     */
/* Optimized implementation of GEMM                                                     */
/* --------------------------------                                                     */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/*                                                                                      */
/* Inspirations:                                                                        */
/* - https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/          */
/*                                                                                      */
/****************************************************************************************/

/**
 * TODO:
 * - Figure out way to benchmark implementations (must test agains cuBLAS and naive GEMM)
 * - Implement tiled version of GEMM
 * - Optimize shared memory usage
 * - Memory coalescing
 * - Asynchronous memory copy (Elaborate on why this is good)
 * - Optimize specifically for A2000 GPU (dimensions, etc.)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h> 
#inlcude "gemm.h"

#define BDIMX 16
#define BDIMY 8

__global__ void gemm_opt_kernel(float* A, float* B, float* C, int N)
{
	// Define block and thread indices
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ float s_A[BDIMY][BDIMX];
	__shared__ float s_B[BDIMY][BDIMX];

	// Accumulate partial sum in register
	float sum = 0.0f;

	// Loop over tiles
	for (int t = 0; t < N; t += BDIMX)
	{
		// Load tiles into shared memory
		s_A[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
		s_B[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];

		// Synchronize threads
		__syncthreads();

		// Accumulate partial sum
		for (int i = 0; i < BDIMX; i++)
		{
			sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
		}

		// Synchronize threads
		__syncthreads();
	}

	// Write to global memory
	if (row < N && col < N)
	{
		C[row * N + col] = sum;
	}
}

// Host wrapper function
void gemm_opt(float* A_d, float* B_d, float* C_d, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    gemm_opt_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
}
