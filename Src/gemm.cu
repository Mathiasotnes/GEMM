/****************************************************************************************/
/* Optimized implementation of GEMM                                                     */
/* --------------------------------                                                     */
/* Author: Mathias Otnes                                                                */
/* year:   2024                                                                         */
/*                                                                                      */
/* Inspirations:                                                                        */
/* - https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/          */
/*                                                                                      */
/****************************************************************************************/

/**
 * TODO:
 * - Figure out way to benchmark implementations (must test agains cuBLAS and naive GEMM)
 * - Implement tiled version of GEMM
 * - Optimize shared memory usage
 * - Memory coalescing
 * - Asynchronous memory copy (Elaborate on why this is good)
 * - Optimize specifically for A2000 GPU (dimensions, etc.)
 */

#include <stdio.h>

int gemm()
{
	printf("Hello world!\n");

	return 0;
}