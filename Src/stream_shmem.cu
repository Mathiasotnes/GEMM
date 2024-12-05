/****************************************************************************************/
/* stream_shmem.cu                                                                      */
/* ------------------------------------------------------------------------------------ */
/* Implementation of GEMM using streams and shared memory. The idea behind this was to  */
/* transfer one tile of the tile-based matrix multiplication to the device at a time.   */
/* And then start the kernel for that tile asynchronously. This turned out to run       */
/* very slowly, and I therefore abandoned this approach.                                */
/* ------------------------------------------------------------------------------------ */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

__global__ void gemm_stream_shmem_kernel( float* A_tile, float* B_tile, float* C, int N, int C_row_offset, int C_col_offset )
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    tile_A[ty][tx] = A_tile[ty * TILE_SIZE + tx];
    tile_B[ty][tx] = B_tile[ty * TILE_SIZE + tx];
    __syncthreads();

    float val = 0.0f;
    for (int k = 0; k < TILE_SIZE; k++)
    {
        val += tile_A[ty][k] * tile_B[k][tx];
    }

    // Global coordinates
    int row = C_row_offset + ty;
    int col = C_col_offset + tx;

    if (row < N && col < N) {
        C[row * N + col] += val;
    }
}

void gemm_stream_shmem( float* A, float* B, float* C, int N )
{

    size_t matrix_size  = N * N * sizeof(float);
    int num_tiles       = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_streams     = num_tiles;

    if ( VERBOSE ) {
        printf("Launching stream shmem kernel with %d streams\n", num_streams);
    }

    // Memory allocation
    float *C_d;
    checkCudaErrors(cudaMalloc((void**)&C_d, matrix_size));
    checkCudaErrors(cudaMemcpy(C_d, C, matrix_size, cudaMemcpyHostToDevice));

    float **tile_A = new float*[num_streams];
    float **tile_B = new float*[num_streams];
    cudaStream_t *streams = new cudaStream_t[num_streams];

    // Initialize streams
    for ( int i = 0; i < num_streams; i++ ) {
        checkCudaErrors(cudaMalloc((void**)&tile_A[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&tile_B[i], TILE_SIZE * TILE_SIZE * sizeof(float)));
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    int stream_idx = 0;
    for ( int row = 0; row < num_tiles; row++ ) {
        for ( int col = 0; col < num_tiles; col++ ) {
            cudaStream_t stream = streams[stream_idx];

            // Tile offsets
            int C_row_offset = row * TILE_SIZE;
            int C_col_offset = col * TILE_SIZE;
            int A_row_offset = C_row_offset;
            int A_col_offset = 0;
            int B_row_offset = 0;
            int B_col_offset = C_col_offset;

            // Copy tile from A
            for ( int i = 0; i < TILE_SIZE; i++ ) {
                int A_row = A_row_offset + i;
                if ( A_row < N ) {
                    checkCudaErrors(cudaMemcpyAsync(
                        tile_A[stream_idx] + i * TILE_SIZE,
                        A + A_row * N + A_col_offset,
                        TILE_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                }

                else {
                    checkCudaErrors(cudaMemsetAsync(tile_A[stream_idx] + i * TILE_SIZE, 0, TILE_SIZE * sizeof(float), stream));
                }
            }

            // Copy tile from B
            for ( int i = 0; i < TILE_SIZE; i++ ) {
                int B_row = B_row_offset + i;
                if ( B_row < N ) {
                    checkCudaErrors(cudaMemcpyAsync(
                        tile_B[stream_idx] + i * TILE_SIZE,
                        B + B_row * N + B_col_offset,
                        TILE_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                }

                else {
                    checkCudaErrors(cudaMemsetAsync(tile_B[stream_idx] + i * TILE_SIZE, 0, TILE_SIZE * sizeof(float), stream));
                }
            }

            // Launch kernel
            dim3 blockSize(TILE_SIZE, TILE_SIZE);
            dim3 gridSize(1, 1);
            gemm_stream_shmem_kernel<<<gridSize, blockSize, 0, stream>>>(tile_A[stream_idx], tile_B[stream_idx], C_d, N, C_row_offset, C_col_offset);

            stream_idx = (stream_idx + 1) % num_streams;
        }
    }

    // Wait for all streams to finish
    for ( int i = 0; i < num_streams; ++i ) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    // Device -> Host
    checkCudaErrors(cudaMemcpy(C, C_d, matrix_size, cudaMemcpyDeviceToHost));

    // Memory deallocation
    for ( int i = 0; i < num_streams; i++ ) {
        checkCudaErrors(cudaFree(tile_A[i]));
        checkCudaErrors(cudaFree(tile_B[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    
    delete[] tile_A;
    delete[] tile_B;
    delete[] streams;

    checkCudaErrors(cudaFree(C_d));
}
