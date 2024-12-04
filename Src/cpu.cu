/****************************************************************************************/
/* cpu.cu                                                                               */
/* --------------------------                                                           */
/* CPU implementation of GEMM                                                           */
/* --------------------------                                                           */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h> 

void gemm_cpu(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Wrapper for gemm_cpu to accept device pointers
void gemm_cpu_wrapper(float* A_d, float* B_d, float* C_d, int N) {
    int matrix_size = N * N * sizeof(float);
    float *A = (float*)malloc(matrix_size);
    float *B = (float*)malloc(matrix_size);
    float *C = (float*)malloc(matrix_size);

    cudaMemcpy(A, A_d, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, B_d, matrix_size, cudaMemcpyDeviceToHost);

    gemm_cpu(A, B, C, N);

    cudaMemcpy(C_d, C, matrix_size, cudaMemcpyHostToDevice);

    free(A);
    free(B);
    free(C);
}
