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

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

#define STREAMS 4 // Number of streams / tiles

__global__ void gemm_stream_shmem_kernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gemm_stream_shmem(float* A, float* B, float* C, int N)
{
    int tile_rows = N / STREAMS;
    int remainder_rows = N % STREAMS;

    size_t matrix_size = N * N * sizeof(float);

    // Allocate device memory for B once
    float *d_B;
    checkCudaErrors(cudaMalloc((void**)&d_B, matrix_size));
    checkCudaErrors(cudaMemcpy(d_B, B, matrix_size, cudaMemcpyHostToDevice));

    // Allocate device memory for A_tiles and C_tiles
    float *d_A_tiles[STREAMS], *d_C_tiles[STREAMS];
    cudaStream_t streams[STREAMS];

    if ( VERBOSE ) {
        printf("Launching stream kernel with %d streams\n", STREAMS);
    }

    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaStreamCreate(&streams[i]));

        int rows_in_tile = tile_rows;
        if (i == STREAMS - 1)
        {
            rows_in_tile += remainder_rows;
        }
        size_t tile_bytes = rows_in_tile * N * sizeof(float);

        // Allocate device memory for A_tile and C_tile
        checkCudaErrors(cudaMalloc((void**)&d_A_tiles[i], tile_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_C_tiles[i], tile_bytes));
    }

    // For each tile
    for (int i = 0; i < STREAMS; i++)
    {
        int rows_in_tile = tile_rows;
        if (i == STREAMS - 1)
        {
            rows_in_tile += remainder_rows;
        }
        size_t tile_bytes = rows_in_tile * N * sizeof(float);

        // Copy A_tile to device asynchronously
        checkCudaErrors(cudaMemcpyAsync(d_A_tiles[i], A + i * tile_rows * N, tile_bytes, cudaMemcpyHostToDevice, streams[i]));

        // Define grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (rows_in_tile + blockSize.y -1) / blockSize.y);

        // Launch kernel in the stream
        gemm_stream_kernel<<<gridSize, blockSize, 0, streams[i]>>>(d_A_tiles[i], d_B, d_C_tiles[i], N);

        // Copy C_tile back to host asynchronously
        checkCudaErrors(cudaMemcpyAsync(C + i * tile_rows * N, d_C_tiles[i], tile_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to finish
    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Free device memory and destroy streams
    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaFree(d_A_tiles[i]));
        checkCudaErrors(cudaFree(d_C_tiles[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // Free device memory for B
    checkCudaErrors(cudaFree(d_B));
}
