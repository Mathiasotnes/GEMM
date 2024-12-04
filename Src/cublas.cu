/********************************************************************************************************/
/* cublas.cu                                                                                            */
/* ----------------------                                                                               */
/* Wrapper for CBLAS GEMM                                                                               */
/* Inspired by NVIDIA sample:                                                                           */
/* https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu  */
/* ----------------------                                                                               */
/* Author: Mathias Otnes                                                                                */
/* year:   2024                                                                                         */
/********************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

void gemm_cublas(float* A_d, float* B_d, float* C_d, int N)
{
    cublasHandle_t handle;
    cudaStream_t stream;

    // Step 1: Create cuBLAS handle and bind a stream
    cublasCreate(&handle);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);

    // Step 2: Set up constants and leading dimensions
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = N;
    int ldb = N;
    int ldc = N;

    // Step 3: Call cuBLAS SGEMM
    // cuBLAS is column-major by default. Since you're working with row-major matrices,
    // transpose both matrices.
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both A and B for row-major compatibility
        N, N, N,                   // M, N, K
        &alpha,                    // Scalar for multiplication
        B_d, ldb,                  // B device pointer, leading dimension ldb
        A_d, lda,                  // A device pointer, leading dimension lda
        &beta,                     // Scalar for accumulation
        C_d, ldc                   // C device pointer, leading dimension ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM failed\n");
    }

    // Step 4: Synchronize
    cudaStreamSynchronize(stream);

    // Step 5: Clean up
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}
