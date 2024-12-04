/****************************************************************************************/
/* cblas.cu                                                                             */
/* ----------------------                                                               */
/* Wrapper for CBLAS GEMM                                                               */
/* ----------------------                                                               */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/****************************************************************************************/

#include <cblas.h>

void gemm_cblas(float* A, float* B, float* C, int N)
{
    // CBLAS uses column-major order, so we need to adjust our code or data accordingly.
    // For simplicity, we can tell CBLAS that our matrices are in row-major order using CBLAS_TRANSPOSE enums.

    cblas_sgemm(
        CblasRowMajor,   // Order
        CblasNoTrans,    // TransA
        CblasNoTrans,    // TransB
        N,               // M
        N,               // N
        N,               // K
        1.0f,            // alpha
        A,               // A
        N,               // lda
        B,               // B
        N,               // ldb
        0.0f,            // beta
        C,               // C
        N                // ldc
    );
}


