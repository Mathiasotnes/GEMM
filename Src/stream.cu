/********************************************************************/
/* stream.cu                                                        */
/* ---------------------------------------------------------------- */
/* Naive implementation of GEMM using streams for memory allocation */
/* ---------------------------------------------------------------- */
/* Author: Mathias Otnes                                            */
/* year:   2024                                                     */
/********************************************************************/

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

#define STREAMS 4
#define TILE_SIZE 16

__global__ void gemm_stream_kernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
            __synchthreads();
        }
        C[row * N + col] = sum;
    }
}

void gemm_stream( float* A, float* B, float* C, int N )
{
    int tile_rows = N / STREAMS;
    int remainder_rows = N % STREAMS;

    size_t matrix_size = N * N * sizeof(float);

    // Allocate device memory for B and C once
    float *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_B, matrix_size));
    checkCudaErrors(cudaMalloc((void**)&d_C, matrix_size));

    checkCudaErrors(cudaMemcpy(d_B, B, matrix_size, cudaMemcpyHostToDevice));

    // Allocate device memory for A_tiles
    float *d_A_tiles[STREAMS];
    cudaStream_t streams[STREAMS];

    if (VERBOSE)
    {
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

        // Allocate device memory for A_tile
        checkCudaErrors(cudaMalloc((void**)&d_A_tiles[i], tile_bytes));
    }

    // For each tile
    for (int i = 0; i < STREAMS; i++)
    {
        int row_offset = i * tile_rows;
        int rows_in_tile = tile_rows;
        if (i == STREAMS - 1)
        {
            rows_in_tile += remainder_rows;
        }
        size_t tile_bytes = rows_in_tile * N * sizeof(float);

        // Copy A_tile to device asynchronously
        checkCudaErrors(cudaMemcpyAsync(d_A_tiles[i], A + row_offset * N, tile_bytes, cudaMemcpyHostToDevice, streams[i]));

        // Define grid and block dimensions
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (rows_in_tile + blockSize.y - 1) / blockSize.y);

        // Launch kernel in the stream
        gemm_stream_kernel<<<gridSize, blockSize, 0, streams[i]>>>(d_A_tiles[i], d_B, d_C, N, row_offset, col_offset);
    }

    // Wait for all streams to finish
    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Copy back d_C to C
    checkCudaErrors(cudaMemcpy(C, d_C, matrix_size, cudaMemcpyDeviceToHost));

    // Free device memory and destroy streams
    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaFree(d_A_tiles[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // Free device memory for B and C
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}
