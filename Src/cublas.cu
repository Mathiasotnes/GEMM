/****************************************************************************************/
/* cublas.cu                                                                            */
/* ----------------------                                                               */
/* Wrapper for CBLAS GEMM                                                               */
/* ----------------------                                                               */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cublas_v2.h>

void gemm_cublas(float* A_d, float* B_d, float* C_d, int N)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS uses column-major order by default
    // To use row-major data, we need to adjust the operation
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose
        N, N, N,                   // M, N, K
        &alpha,
        A_d, N,                    // A device pointer, leading dimension
        B_d, N,                    // B device pointer, leading dimension
        &beta,
        C_d, N                     // C device pointer, leading dimension
    );

    cublasDestroy(handle);
}

