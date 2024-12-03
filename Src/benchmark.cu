/****************************************************************************************/
/* Benchmark script for GEMM                                                            */
/* -------------------------                                                            */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include "gemm.h"

int main()
{
    // CUDA event objects
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording time
    cudaEventRecord(start);
	printf("\nHello world!\n");
    
    // Stop recording time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time elapsed to print hello world: %f ms\n", milliseconds);
	
	return 0;
} 
