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

#define TILE_SIZE 16

__global__ void gemm_stream_shmem_kernel( float* A, float* B, float* C, int N )
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
    size_t matrix_size = N * N * sizeof(float);

    // Calculate the number of tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    if ( VERBOSE ) {
        printf("Launching stream shmem kernel with %d tiles\n", num_tiles);
    }

    // Allocate device memory for A, B, and C
    float *tile_A[num_tiles];
    float *tile_B[num_tiles];
    float *tile_C[num_tiles];
    
    for (int i = 0; i < num_tiles; i++)
    {
        checkCudaErrors(cudaMalloc((void**)&tile_A[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&tile_B[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&tile_C[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
    }


    // Create streams
    cudaStream_t streams[num_tiles];
    for (int i = 0; i < num_tiles; i++)
    {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // This assumes that N is divisible by TILE_SIZE
    int nRows = N / num_tiles;
    int nCols = N / num_tiles;

    // Launch kernels for each tile of C
    int stream_idx = 0;
    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            // Assign stream in a round-robin fashion
            cudaStream_t stream = streams[stream_idx];

            // Calculate tile offsets
            int C_row_offset = row * TILE_SIZE;
            int C_col_offset = col * TILE_SIZE;
            int A_row_offset = C_row_offset;
            int B_col_offset = C_col_offset;

            // Copy part of A and B that makes up the tile to device memory
            for (int i = 0; i < TILE_SIZE; i++)
            {
                int A_row = A_row_offset + i;
                int B_col = B_col_offset + i;

                checkCudaErrors(cudaMemcpyAsync(tile_A[stream_idx] + i * TILE_SIZE, A + A_row * N + B_col, TILE_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
                checkCudaErrors(cudaMemcpyAsync(tile_B[stream_idx] + i * TILE_SIZE, B + A_row * N + B_col, TILE_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            }

            // Kernel launch parameters
            dim3 blockSize(TILE_SIZE, TILE_SIZE);
            dim3 gridSize(1, 1);

            // Launch kernel
            gemm_stream_shmem_kernel<<<gridSize, blockSize, 0, stream>>>(tile_A[stream_idx], tile_B[stream_idx], tile_C[stream_idx], TILE_SIZE);
            
            stream_idx++;
        }
    }

    // Synchronize streams
    for (int i = 0; i < num_tiles; ++i)
    {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // Copy the result matrix C back to host memory
    checkCudaErrors(cudaMemcpy(C, d_C, matrix_size, cudaMemcpyDeviceToHost));

    // Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}
