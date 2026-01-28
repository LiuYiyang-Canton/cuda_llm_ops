// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for bitonic sort.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

/**
 * @brief Launches the bitonic sort kernels for fp32 data.
 * @param data Pointer to device input/output buffer of shape [batchSize, arrayLength].
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 * @param ascending True to sort ascending, false for descending.
 */
void LaunchBitonicSortFp32Kernel(float* data, int batchSize, int arrayLength, bool ascending);
