// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for elementwise add.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kElementwiseAddWorkPerThread = 4;

/**
 * @brief Launches the fp32 elementwise add kernel.
 * @param a Device pointer to input matrix A.
 * @param b Device pointer to input matrix B.
 * @param result Device pointer to output matrix.
 * @param n Matrix dimension (square). Must be divisible by 4.
 * @param threadsPerBlock Number of threads per block for the kernel launch.
 */
void LaunchElementwiseAddFp32Kernel(const float* a,
                                   const float* b,
                                   float* result,
                                   int n,
                                   int threadsPerBlock);
