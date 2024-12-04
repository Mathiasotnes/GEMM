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

__global__ void gemm_stream_kernel( float* d_a, float* d_b, float* d_c, int N )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += d_a[row * N + i] * d_b[i * N + col];
        }
        d_c[row * N + col] = sum;
    }
}

/**
 * @brief Uses streams for memory allocation and kernel execution
 * 
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param N Size of matrices
 */
void gemm_stream( float* A, float* B, float* C, int N ) 
{

    float *d_a[STREAMS], *d_b[STREAMS], *d_c[STREAMS];
    int streamSize = N / STREAMS;
	int streamBytes = streamSize * sizeof( int );

    if ( VERBOSE ) {
        printf("Launching streamed kernel with %d streams\n", STREAMS);
    }

    cudaStream_t stream[STREAMS];
    for ( int i = 0; i < STREAMS; i++ )
    {
        cudaStreamCreate(&stream[i]);
    }

	// Allocate memory on device
	for ( int i = 0; i < STREAMS; i++ )
    {
        checkCudaErrors( cudaMalloc(&d_a[i], streamBytes) );
        checkCudaErrors( cudaMalloc(&d_b[i], streamBytes) );
        checkCudaErrors( cudaMalloc(&d_c[i], streamBytes) );
    }

    // Enable DMA transfer operation by allocating pinned host memory
    checkCudaErrors( cudaMallocHost((void **)&A, N * N * sizeof(float)) );
    checkCudaErrors( cudaMallocHost((void **)&B, N * N * sizeof(float)) );
    checkCudaErrors( cudaMallocHost((void **)&C, N * N * sizeof(float)) );

	// Copy data to device and start streams asynchronously
    for ( int i = 0; i < STREAMS; i++ )
    {
        checkCudaErrors( cudaMemcpyAsync(d_a[i], A + i * streamSize, streamBytes, cudaMemcpyHostToDevice, stream[i]) );
        checkCudaErrors( cudaMemcpyAsync(d_b[i], B + i * streamSize, streamBytes, cudaMemcpyHostToDevice, stream[i]) );

        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

        if ( VERBOSE ) {
            printf("Launching streamed kernel %d with grid size %d, %d and block size %d, %d\n", i, gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        }

        gemm_stream_kernel<<<gridSize, blockSize, 0, stream[i]>>>(d_a[i], d_b[i], d_c[i], N);

        checkCudaErrors( cudaMemcpyAsync(C + i * streamSize, d_c[i], streamBytes, cudaMemcpyDeviceToHost, stream[i]) );

    }

    // Wait for the streams to finish
    for ( int i = 0; i < STREAMS; i++ )
    {
        checkCudaErrors( cudaStreamSynchronize(stream[i]) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );

    for ( int i = 0; i < STREAMS; i++ )
    {
        checkCudaErrors( cudaFree(d_a[i]) );
        checkCudaErrors( cudaFree(d_b[i]) );
        checkCudaErrors( cudaFree(d_c[i]) );
        checkCudaErrors( cudaStreamDestroy(stream[i]) );
    }

}
