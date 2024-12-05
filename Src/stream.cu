/****************************************************************************************/
/* stream.cu                                                                            */
/* ------------------------------------------------------------------------------------ */
/* Implementation of GEMM using streams. The idea was to split up the A-matrix into     */
/* tiles and transfer one tile to the device at a time. Then start the kernel for that  */
/* tile asynchronously.                                                                 */
/* ------------------------------------------------------------------------------------ */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

__global__ void gemm_stream_kernel( float* A, float* B, float* C, int N, int row_offset, int rows_in_tile )
{
    // Global coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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

void gemm_stream( float* A, float* B, float* C, int N )
{
    int     tile_rows       = N / STREAMS;
    int     remainder_rows  = N % STREAMS;
    size_t  matrix_size     = N * N * sizeof(float);

    // Memory allocation
    float *B_d, *C_d;
    checkCudaErrors(cudaMalloc((void**)&B_d, matrix_size));
    checkCudaErrors(cudaMalloc((void**)&C_d, matrix_size));

    // Host -> Device
    checkCudaErrors( cudaMemcpy(B_d, B, matrix_size, cudaMemcpyHostToDevice) );

    float *A_d_tiles[STREAMS];
    cudaStream_t streams[STREAMS];

    if ( VERBOSE ) {
        printf("Launching stream kernel with %d streams\n", STREAMS);
    }

    // Stream initialization
    for (int i = 0; i < STREAMS; i++)
    {
        checkCudaErrors(cudaStreamCreate(&streams[i]));

        int rows_in_tile = tile_rows;
        if (i == STREAMS - 1)
        {
            rows_in_tile += remainder_rows;
        }
        size_t tile_bytes = rows_in_tile * N * sizeof(float);

        checkCudaErrors( cudaMalloc((void**)&A_d_tiles[i], tile_bytes) );
    }

    // Start DMA transfer and kernel for each stream
    for (int i = 0; i < STREAMS; i++)
    {
        int row_offset = i * tile_rows;
        int rows_in_tile = tile_rows;
        if (i == STREAMS - 1)
        {
            rows_in_tile += remainder_rows;
        }
        size_t tile_bytes = rows_in_tile * N * sizeof(float);

        // Host -> Device (Async DMA transfer)
        checkCudaErrors( cudaMemcpyAsync(A_d_tiles[i], A + row_offset * N, tile_bytes, cudaMemcpyHostToDevice, streams[i]) );

        // Launch kernel
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (rows_in_tile + blockSize.y - 1) / blockSize.y);
        gemm_stream_kernel<<<gridSize, blockSize, 0, streams[i]>>>(A_d_tiles[i], B_d, C_d, N, row_offset, rows_in_tile);
    }

    // Wait for all streams to finish
    for ( int i = 0; i < STREAMS; i++ ) {
        checkCudaErrors( cudaStreamSynchronize(streams[i]) );
    }

    // Device -> Host
    checkCudaErrors( cudaMemcpy(C, C_d, matrix_size, cudaMemcpyDeviceToHost) );

    // Memory deallocation
    for ( int i = 0; i < STREAMS; i++ ) {
        checkCudaErrors(cudaFree(A_d_tiles[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}
