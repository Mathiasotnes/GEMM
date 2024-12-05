/********************************************************************************************************/
/* cublas.cu                                                                                            */
/* ---------------------------------------------------------------------------------------------------- */
/* Wrapper for CBLAS GEMM                                                                               */
/* Inspired by NVIDIA sample:                                                                           */
/* https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu  */
/* ---------------------------------------------------------------------------------------------------- */
/* Author: Mathias Otnes                                                                                */
/* year:   2024                                                                                         */
/********************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

void gemm_cublas_kernel( float* A_d, float* B_d, float* C_d, int N )
{
    cublasHandle_t  handle;
    cudaStream_t    stream;

    cublasCreate(&handle);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);

    // Configure SGEMM to match our problem
    float alpha     = 1.0f;
    float beta      = 0.0f;
    int   lda       = N; 
    int   ldb       = N; 
    int   ldc       = N;

    // To compute C = A * B in row-major, we call cublasSgemm with B and A swapped, because
    // cuBLAS assumes column-major order.
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        B_d, ldb, // Note that B_d comes before A_d
        A_d, lda,
        &beta,
        C_d, ldc
    );

    if ( status != CUBLAS_STATUS_SUCCESS ) {
        printf("cuBLAS SGEMM failed\n");
    }

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}

void gemm_cublas( float* A, float* B, float* C, int N ) 
{

    // Memory allocation
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, N * N * sizeof(float));
    cudaMalloc(&B_d, N * N * sizeof(float));
    cudaMalloc(&C_d, N * N * sizeof(float));

    // Host -> Device
    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    gemm_cublas_kernel(A_d, B_d, C_d, N);

    // Device -> Host
    cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Memory deallocation
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
