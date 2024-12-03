/****************************************************************************************/
/* Header file for GEMM library                                                         */
/* ----------------------------                                                         */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#ifndef GEMM_H
#define GEMM_H

void gemm_cpu(float* A, float* B, float* C, int N);
void gemm_naive(float* A, float* B, float* C, int N);
void gemm_opt(float* A, float* B, float* C, int N);

#endif // GEMM_H