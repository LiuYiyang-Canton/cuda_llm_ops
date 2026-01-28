// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for FP16 GEMM.
// ==============================================================================
#pragma once

#include <cuda_fp16.h>

/**
 * @brief Launches the fp16 GEMM kernel with fp32 accumulation.
 * @param a Pointer to device matrix A (row-major) of shape [m, k].
 * @param b Pointer to device matrix B (column-major) of shape [k, n].
 * @param c Pointer to device output matrix C (row-major) of shape [m, n].
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A and rows in B.
 */
void LaunchGemmFp16Kernel(const half* a, const half* b, float* c, int m, int n, int k);
