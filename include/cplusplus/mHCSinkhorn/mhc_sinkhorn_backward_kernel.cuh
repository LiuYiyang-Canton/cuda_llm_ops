// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for mHCSinkhorn backward pass.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kMhcSinkhornThreadsPerSlice = 2;
constexpr int kThreadsPerBlock = 256;
constexpr int kSlicesPerBlock = kThreadsPerBlock / kMhcSinkhornThreadsPerSlice;

/**
 * @brief Launches the Sinkhorn-Knopp backward kernel.
 * @param input Pointer to device input tensor X^{(0)}.
 * @param gradOutput Pointer to device gradient w.r.t. output.
 * @param gradX Pointer to device gradient w.r.t. input.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param epsilon Small constant added to row/column sums.
 */
template<int matrixSize, int iterations>
void LaunchMhcSinkhornBackwardKernel(const float* input,
                                     const float* gradOutput,
                                     float* gradX,
                                     int batchSize,
                                     int seqLength,
                                     float epsilon);
