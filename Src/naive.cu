/****************************************************************************************/
/* naive.cu                                                                             */
/* ------------------------------------------------------------------------------------ */
/* Naive implementation of GEMM                                                 		*/
/* ------------------------------------------------------------------------------------ */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/*                                                                                      */
/* Inspiration:                                                                         */
/* - https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/          */
/*                                                                                      */
/****************************************************************************************/

#include <cuda_runtime.h>
#include <helper_cuda.h> 
#include <stdio.h>
#include "gemm.h"

__global__ void gemm_naive_kernel( float* A_d, float* B_d, float* C_d, int N )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row < N && col < N ) {
        float sum = 0.0f;
        for ( int i = 0; i < N; i++ ) {
            sum += A_d[row * N + i] * B_d[i * N + col];
        }
        C_d[row * N + col] = sum;
    }
}

void gemm_naive( float* A, float* B, float* C, int N ) 
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

    gemm_naive_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);

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
