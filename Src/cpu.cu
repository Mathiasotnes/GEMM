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
