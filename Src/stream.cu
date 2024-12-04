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

#define STREAMS 4 // Number of streams / tiles

__global__ void gemm_stream_kernel(float* A, float* B, float* C,
                                   int N, int tile_size,
                                   int a_row_offset, int a_col_offset,
                                   int b_row_offset, int b_col_offset,
                                   int c_row_offset, int c_col_offset)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Compute global indices
    int global_row = c_row_offset + row;
    int global_col = c_col_offset + col;

    float Cvalue = 0.0f;

    // Loop over the tiles of A and B required for this tile of C
    for (int k = 0; k < (N + tile_size - 1) / tile_size; ++k)
    {
        // Compute the indices for the tiles
        int a_col = a_col_offset + k * tile_size;
        int b_row = b_row_offset + k * tile_size;

        // Load elements of A and B
        float A_element = 0.0f;
        float B_element = 0.0f;

        if ((global_row < N) && (a_col + col < N))
            A_element = A[global_row * N + a_col + col];

        if ((b_row + row < N) && (global_col < N))
            B_element = B[(b_row + row) * N + global_col];

        // Accumulate the product
        Cvalue += A_element * B_element;
    }

    // Write the result to global memory
    if (global_row < N && global_col < N)
    {
        C[global_row * N + global_col] = Cvalue;
    }
}


void gemm_stream(float* A, float* B, float* C, int N)
{
    const int tile_size = 16; // Adjust as needed

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));
    checkCudaErrors(cudaMalloc(&d_C, size));

    // Copy entire matrices to device
    checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // Calculate the number of tiles
    int num_tiles = (N + tile_size - 1) / tile_size;

    // Create streams
    const int num_streams = STREAMS;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // Launch kernels for each tile of C
    int stream_idx = 0;
    for (int tile_row = 0; tile_row < num_tiles; ++tile_row)
    {
        for (int tile_col = 0; tile_col < num_tiles; ++tile_col)
        {
            // Compute offsets
            int c_row_offset = tile_row * tile_size;
            int c_col_offset = tile_col * tile_size;
            int a_row_offset = c_row_offset;
            int b_col_offset = c_col_offset;

            // Assign stream in a round-robin fashion
            cudaStream_t stream = streams[stream_idx];
            stream_idx = (stream_idx + 1) % num_streams;

            // Kernel launch parameters
            dim3 blockSize(tile_size, tile_size);
            dim3 gridSize(1, 1);

            // Launch kernel
            gemm_stream_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_A, d_B, d_C,
                N, tile_size,
                a_row_offset, 0,       // a_col_offset starts at 0
                0, b_col_offset,       // b_row_offset starts at 0
                c_row_offset, c_col_offset);

            // Note: The kernel itself loops over the k-tiles
        }
    }

    // Synchronize streams
    for (int i = 0; i < num_streams; ++i)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // Clean up
    for (int i = 0; i < num_streams; ++i)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}
