/****************************************************************************************/
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

typedef void (*gemm_func_t)(float*, float*, float*, int);

typedef struct {
    gemm_func_t func;
    const char* name;
} gemm_method_t;



// Function to compare two matrices
int compare_matrices(float* mat1, float* mat2, int N) {
    float epsilon = 1e-5f;
    for (int i = 0; i < N * N; ++i) {
        if (fabsf(mat1[i] - mat2[i]) > epsilon) {
            return 0;
        }
    }
    return 1;
}

int main() {
    // List of methods
    gemm_method_t methods[] = {
        {gemm_cpu_wrapper, "CPU"},
        {gemm_naive, "Naive GPU"},
        {gemm_opt, "Optimized GPU"}
    };
    int num_methods = sizeof(methods) / sizeof(methods[0]);

    // List of sizes
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Open result file
    FILE* result_file = fopen("gemm_benchmark_results.csv", "w");
    if (!result_file) {
        printf("Error opening result file.\n");
        return -1;
    }
    // Write header
    fprintf(result_file, "Size,Method,Time(ms),Correct\n");

    for (int size_idx = 0; size_idx < num_sizes; ++size_idx) {
        int N = sizes[size_idx];
        int matrix_size = N * N * sizeof(float);

        // Allocate host memory
        float* A = (float*)malloc(matrix_size);
        float* B = (float*)malloc(matrix_size);
        float* C_ref = (float*)malloc(matrix_size);
        float* C_test = (float*)malloc(matrix_size);

        // Allocate device memory
        float *A_d, *B_d, *C_d;
        cudaMalloc((void**)&A_d, matrix_size);
        cudaMalloc((void**)&B_d, matrix_size);
        cudaMalloc((void**)&C_d, matrix_size);

        // Initialize A and B
        for (int i = 0; i < N * N; ++i) {
            A[i] = rand() / (float)RAND_MAX;
            B[i] = rand() / (float)RAND_MAX;
        }

        // Copy to device
        cudaMemcpy(A_d, A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B, matrix_size, cudaMemcpyHostToDevice);

        // Compute reference result
        gemm_cpu(A, B, C_ref, N);

        for (int method_idx = 0; method_idx < num_methods; ++method_idx) {
            gemm_func_t func = methods[method_idx].func;
            const char* method_name = methods[method_idx].name;

            // Zero C_test and C_d
            memset(C_test, 0, matrix_size);
            cudaMemset(C_d, 0, matrix_size);

            // Start timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            // Call method
            func(A_d, B_d, C_d, N);

            // Stop timing
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);

            // Copy result back to host
            cudaMemcpy(C_test, C_d, matrix_size, cudaMemcpyDeviceToHost);

            // Check correctness
            int correct = compare_matrices(C_ref, C_test, N);

            if (!correct) {
                printf("Error in %s method\n", method_name);
            }

            // Log results
            printf("Size: %d, Method: %s, Time: %.3f ms, Correct: %s\n", N, method_name, milliseconds, correct ? "Yes" : "No");
            fprintf(result_file, "%d,%s,%.3f,%s\n", N, method_name, milliseconds, correct ? "Yes" : "No");

            // Clean up events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Free memory
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        free(A);
        free(B);
        free(C_ref);
        free(C_test);
    }

    // Close result file
    fclose(result_file);

    return 0;
}
