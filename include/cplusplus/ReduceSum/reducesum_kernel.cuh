// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for ReduceSum.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kReduceSumRowsPerBlock = 4;
constexpr int kReduceSumThreadsPerBlock = 256;

/**
 * @brief Launches the fp32 reduce-sum kernel for a row-major matrix.
 * @param matrix Pointer to device input matrix (row-major).
 * @param rowSum Pointer to device output row-sum vector.
 * @param rows Number of rows in the matrix; must be divisible by kReduceSumRowsPerBlock.
 * @param cols Number of columns in the matrix; must be divisible by 4.
 */
void LaunchReduceSumFp32Kernel(const float* matrix, float* rowSum, int rows, int cols);
