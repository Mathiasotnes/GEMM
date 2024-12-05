/****************************************************************************************/
/* shmem.cu                                                                             */
/* ------------------------------------------------------------------------------------ */
/* Shared memory implementation of GEMM                                                 */
/* ------------------------------------------------------------------------------------ */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

__global__ void gemm_shared_kernel( float* A, float* B, float* C, int N )
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

        if ( col < N && (i * TILE_SIZE + threadIdx.y) < N ) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        } 
        
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        for ( int j = 0; j < TILE_SIZE; j++ ) {
            val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }
        __syncthreads();
    }

    if ( row < N && col < N ) {
        C[row * N + col] = val;
    }
}

void gemm_shmem( float* A, float* B, float* C, int N ) 
{
	// Memory allocation
	float *A_d, *B_d, *C_d;

	checkCudaErrors( cudaMalloc(&A_d, N * N * sizeof(float)) );
	checkCudaErrors( cudaMalloc(&B_d, N * N * sizeof(float)) );
	checkCudaErrors( cudaMalloc(&C_d, N * N * sizeof(float)) );

	// Host -> Device
	checkCudaErrors( cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice) );

	// Launch kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	if ( VERBOSE ) {
		printf("Launching naive kernel with grid size %d, %d and block size %d, %d\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
	}

    gemm_shared_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);

	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf("Kernel launch error: %s\n", cudaGetErrorString(err));
	}

    checkCudaErrors( cudaDeviceSynchronize() );

	// Device -> Host
	checkCudaErrors( cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost) );

	// Memory deallocation
	checkCudaErrors( cudaFree(A_d) );
	checkCudaErrors( cudaFree(B_d) );
	checkCudaErrors( cudaFree(C_d) );
}
