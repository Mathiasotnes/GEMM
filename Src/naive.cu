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

__global__ void gemm_naive_kernel(float* A, float* B, float* C, int N)
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

// Host wrapper function
void gemm_naive(float* A_d, float* B_d, float* C_d, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    gemm_naive_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
}
