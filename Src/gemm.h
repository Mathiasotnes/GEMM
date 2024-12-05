/****************************************************************************************/
/* gemm.h                                                                               */
/* ------------------------------------------------------------------------------------ */
/* Header file for GEMM API                                                             */
/* ------------------------------------------------------------------------------------ */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#ifndef GEMM_H
#define GEMM_H

/****************************************************************************************/
/* Configuration                                                                        */
/****************************************************************************************/

/**
 * Verbosity:
 * 0: No output
 * 1: Kernel and grid sizes
 * 2: Entire matrixes
 */
#define VERBOSE     1

/**
 * Number of streams to use for the stream version of the GEMM algorithm.
 */
#define STREAMS     4

/**
 * Size of the tiles used in the shared memory for all the tile-based algorithms.
 */
#define TILE_SIZE   16


/****************************************************************************************/
/* GEMM API                                                                             */
/****************************************************************************************/

void gemm_cublas            ( float* A, float* B, float* C, int N );
void gemm_cpu               ( float* A, float* B, float* C, int N );
void gemm_naive             ( float* A, float* B, float* C, int N );
void gemm_shmem             ( float* A, float* B, float* C, int N );
void gemm_stream            ( float* A, float* B, float* C, int N );
void gemm_stream_shmem      ( float* A, float* B, float* C, int N );

#endif // GEMM_H
