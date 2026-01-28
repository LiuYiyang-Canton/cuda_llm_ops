// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for mHCSinkhorn.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

/**
 * @brief Launches the Sinkhorn-Knopp forward kernel for (B, S, N, N) tensors.
 * @param input Pointer to device tensor H with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param iterations Number of Sinkhorn iterations to run.
 * @param epsilon Small constant to avoid division by zero.
 */
template <int matrixSize>
void LaunchMhcSinkhornKernel(float* input,
                             int batchSize,
                             int seqLength,
                             int iterations,
                             float epsilon);
