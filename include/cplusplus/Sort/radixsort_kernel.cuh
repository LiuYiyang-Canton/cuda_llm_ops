// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for radix sort.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

/**
 * @brief Launches the radix sort pipeline for fp32 arrays.
 * @param input Pointer to device input buffer (modified in-place).
 * @param output Pointer to device output buffer.
 * @param histogram Pointer to device histogram buffer.
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 */
void LaunchRadixSortFp32Kernel(float* input,
                               float* output,
                               int* histogram,
                               int batchSize,
                               int arrayLength);
