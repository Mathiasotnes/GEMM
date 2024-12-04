/****************************************************************************************/
/* Header file for GEMM library                                                         */
/* ----------------------------                                                         */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#ifndef GEMM_H
#define GEMM_H

/**
 * Verbosity:
 * 0: No output
 * 1: Kernel and grid sizes
 * 2: Entire matrixes
 */
#define VERBOSE 0

void gemm_cublas            ( float* A, float* B, float* C, int N );
void gemm_cpu               ( float* A, float* B, float* C, int N );
void gemm_naive             ( float* A, float* B, float* C, int N );
void gemm_stream            ( float* A, float* B, float* C, int N );
void gemm_stream_shmem      ( float* A, float* B, float* C, int N );

#endif // GEMM_H