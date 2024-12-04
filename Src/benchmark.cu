/****************************************************************************************/
/* benchmark.cu                                                                         */
/* -------------------------                                                            */
/* Benchmark script for GEMM                                                            */
/* -------------------------                                                            */
/* Author: Mathias Otnes                                                                */
/* Year:   2024                                                                         */
/****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gemm.h"

/****************************************************************************************/
/* Local Definitions & Datatypes                                                        */
/****************************************************************************************/

#define LOCAL static

typedef void (*gemm_func_t)(float*, float*, float*, int);

typedef struct {
    gemm_func_t func;
    const char* name;
} gemm_method_t;


/****************************************************************************************/
/* Configuration                                                                        */
/****************************************************************************************/


LOCAL gemm_method_t methods[] = {
    {gemm_cpu,          "CPU"            },
    {gemm_naive,        "Naive GPU"      },
    {gemm_opt,          "Optimized GPU"  }
    // {gemm_cublas,       "cuBLAS"         }
};
LOCAL int num_methods   = sizeof(methods) / sizeof(methods[0]);
LOCAL int sizes[]       = { 16, 32, 64, 128, 256 };
LOCAL int num_sizes     = sizeof(sizes) / sizeof(sizes[0]);


/****************************************************************************************/
/* Local Function Prototypes                                                            */
/****************************************************************************************/

LOCAL int compare_matrices(float* mat1, float* mat2, int N);


/****************************************************************************************/
/* Main Program                                                                         */
/****************************************************************************************/

int main() {

    float ms;

    FILE* result_file = fopen("gemm_benchmark_results.csv", "w");
    if (!result_file) {
        printf("Error opening result file.\n");
        return -1;
    }

    fprintf(result_file, "Size,Method,Time(ms)\n");

    for (int size_idx = 0; size_idx < num_sizes; ++size_idx) {
        int N = sizes[size_idx];
        int matrix_size = N * N * sizeof(float);

        // Host memory
        float* A        = (float*)malloc(matrix_size);
        float* B        = (float*)malloc(matrix_size);
        float* C        = (float*)malloc(matrix_size);
        float* C_ref    = (float*)malloc(matrix_size);

        // Initialize A and B
        for ( int i = 0; i < N * N; ++i ) {
            A[i] = rand() / (float)RAND_MAX;
            B[i] = rand() / (float)RAND_MAX;
        }

        // Calculate reference result
        gemm_cpu(A, B, C_ref, N);

        for ( int method_idx = 0; method_idx < num_methods; ++method_idx ) {
            gemm_func_t func = methods[method_idx].func;
            const char* method = methods[method_idx].name;

            memset(C, 0, matrix_size);

            // Start timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            // Call method
            func(A, B, C, N);

            // Stop timing
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);

            // Log result
            if ( !compare_matrices(C_ref, C, N) ) {
                printf("Error in %s method\n", method);
            }
            else {
                printf("Size: %d, Method: %s, Time: %.3f ms\n", N, method, ms);
                fprintf(result_file, "%d,%s,%.3f\n", N, method, ms);
            }

            // Clean up events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Free memory
        free(A);
        free(B);
        free(C);
        free(C_ref);
    }

    // Close result file
    fclose(result_file);

    return 0;
}


/****************************************************************************************/
/* Local Function Definitions                                                           */
/****************************************************************************************/

/**
 * @brief Compares two matrices element-wise
 * 
 * @param mat1  Matrix 1
 * @param mat2  Matrix 2
 * @param N     Size of matrices
 * @return      1 if matrices are equal, 0 otherwise
 */
LOCAL int compare_matrices(float* mat1, float* mat2, int N) {
    float epsilon = 1e-5f;
    for (int i = 0; i < N * N; ++i) {
        if (fabsf(mat1[i] - mat2[i]) > epsilon) {
            return 0;
        }
    }
    return 1;
}

