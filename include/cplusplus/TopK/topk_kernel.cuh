// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for TopK.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Launches the radix-select top-k kernels to produce a boolean mask of top-k indices.
 * @param input Pointer to device input buffer (modified in-place).
 * @param topkMask Pointer to device output boolean mask [batchSize, featureSize].
 * @param histogram Pointer to device histogram buffer [batchSize, kTopkRadixHistogramSize].
 * @param targetTopK Pointer to device target-k buffer [batchSize].
 * @param selectedRadix Pointer to device selected radix buffer [batchSize].
 * @param batchSize Number of rows to process.
 * @param featureSize Number of columns per row.
 */
void LaunchTopkRadixSelectFp32(float* input,
                               bool* topkMask,
                               int* histogram,
                               int* targetTopK,
                               uint32_t* selectedRadix,
                               int batchSize,
                               int featureSize);
