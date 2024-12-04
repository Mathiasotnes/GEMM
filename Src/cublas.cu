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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

void gemm_cublas( float* A_d, float* B_d, float* C_d, int N )
{
    cublasHandle_t handle;
    cudaStream_t stream;

    // Create cuBLAS handle and stream
    cublasCreate(&handle);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);

    // Configure SGEMM
    float alpha     = 1.0f;
    float beta      = 0.0f;
    int   lda       = N; 
    int   ldb       = N; 
    int   ldc       = N;

    // To compute C = A * B in row-major, we call cublasSgemm with B and A swapped
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for both
        N, N, N,                   // M, N, K
        &alpha,
        B_d, ldb,                  // Note that B_d comes first (swap order)
        A_d, lda,                  // A_d comes second
        &beta,
        C_d, ldc                   // C matrix
    );

    if ( status != CUBLAS_STATUS_SUCCESS ) {
        printf("cuBLAS SGEMM failed\n");
    }

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}
