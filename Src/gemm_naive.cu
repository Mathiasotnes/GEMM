/********************************************************************/
/* Naive implementation of GEMM                                     */
/* ----------------------------                                     */
/* Author: Mathias Otnes                                            */
/* year:   2024                                                     */
/********************************************************************/

#include <stdio.h>

__global__ void gemm_naive(float* A, float* B, float* C, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
	{
		float sum = 0.0f;
		for (int i = 0; i < N; i++)
		{
			sum += A[row * N + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}
