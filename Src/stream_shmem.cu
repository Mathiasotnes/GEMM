/****************************************************************************************/
/* stream_shmem.cu                                                                      */
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

#define TILE_SIZE 16

__global__ void gemm_stream_shmem_kernel( float* A_tile, float* B_tile, float* C, int N, int C_row_offset, int C_col_offset )
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x + blockIdx.x * TILE_SIZE;

    float val = 0.0f;
    for ( int i = 0; i < (N + TILE_SIZE -1)/ TILE_SIZE; i++ ) {
        if(row < N && (i * TILE_SIZE + threadIdx.x) < N){
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
        } 
        
        else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col < N && (i * TILE_SIZE + threadIdx.y) < N){
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        } 
        
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        for(int j = 0; j < TILE_SIZE; j++){
            val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }
        __syncthreads();
    }
    if ( row < N && col < N ) {
        C[row * N + col] += val;
    }
}

void gemm_stream_shmem(float* A, float* B, float* C, int N)
{
    const int TILE_SIZE = 16;
    const int num_streams = 4; // Adjust as needed

    size_t matrix_size = N * N * sizeof(float);

    // Calculate the number of tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate device memory for C
    float *C_d;
    checkCudaErrors(cudaMalloc((void**)&C_d, matrix_size));
    checkCudaErrors(cudaMemcpy(C_d, C, matrix_size, cudaMemcpyHostToDevice));

    // Allocate arrays for tiles and streams
    float **tile_A = new float*[num_streams];
    float **tile_B = new float*[num_streams];
    cudaStream_t *streams = new cudaStream_t[num_streams];

    // Allocate device memory for tiles and create streams
    for (int i = 0; i < num_streams; i++)
    {
        checkCudaErrors(cudaMalloc((void**)&tile_A[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&tile_B[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    int stream_idx = 0;
    for (int row = 0; row < num_tiles; row++)
    {
        for (int col = 0; col < num_tiles; col++)
        {
            cudaStream_t stream = streams[stream_idx];

            // Calculate tile offsets
            int C_row_offset = row * TILE_SIZE;
            int C_col_offset = col * TILE_SIZE;
            int A_row_offset = C_row_offset;
            int A_col_offset = 0;
            int B_row_offset = 0;
            int B_col_offset = C_col_offset;

            // Copy tiles to device
            // Copy tile from A
            for (int i = 0; i < TILE_SIZE; i++)
            {
                int A_row = A_row_offset + i;
                if (A_row < N)
                {
                    checkCudaErrors(cudaMemcpyAsync(
                        tile_A[stream_idx] + i * TILE_SIZE,
                        A + A_row * N + A_col_offset,
                        TILE_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                }
                else
                {
                    checkCudaErrors(cudaMemsetAsync(tile_A[stream_idx] + i * TILE_SIZE, 0, TILE_SIZE * sizeof(float), stream));
                }
            }

            // Copy tile from B
            for (int i = 0; i < TILE_SIZE; i++)
            {
                int B_row = B_row_offset + i;
                if (B_row < N)
                {
                    checkCudaErrors(cudaMemcpyAsync(
                        tile_B[stream_idx] + i * TILE_SIZE,
                        B + B_row * N + B_col_offset,
                        TILE_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                }
                else
                {
                    checkCudaErrors(cudaMemsetAsync(tile_B[stream_idx] + i * TILE_SIZE, 0, TILE_SIZE * sizeof(float), stream));
                }
            }

            // Kernel launch parameters
            dim3 blockSize(TILE_SIZE, TILE_SIZE);
            dim3 gridSize(1, 1);

            // Launch kernel
            gemm_stream_shmem_kernel<<<gridSize, blockSize, 0, stream>>>(tile_A[stream_idx], tile_B[stream_idx], C_d, N, C_row_offset, C_col_offset);

            // Update stream index
            stream_idx = (stream_idx + 1) % num_streams;
        }
    }

    // Synchronize streams
    for (int i = 0; i < num_streams; ++i)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(C, C_d, matrix_size, cudaMemcpyDeviceToHost));

    // Free device memory and destroy streams
    for (int i = 0; i < num_streams; i++)
    {
        checkCudaErrors(cudaFree(tile_A[i]));
        checkCudaErrors(cudaFree(tile_B[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    delete[] tile_A;
    delete[] tile_B;
    delete[] streams;

    checkCudaErrors(cudaFree(C_d));
}
