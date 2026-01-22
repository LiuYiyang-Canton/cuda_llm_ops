// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// ==============================================================================
#ifndef MHC_SINKHORN_H
#define MHC_SINKHORN_H

#include <cstddef>

/**
 * @brief Applies Sinkhorn-Knopp iterations to make each (N, N) slice approximately doubly stochastic.
 * @param input Pointer to input tensor H with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param output Pointer to output tensor with the same shape as input.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param iterations Number of Sinkhorn iterations to run.
 */
void MHCSinkhornCpu(const float* input,
                    float* output,
                    int batchSize,
                    int seqLength,
                    int matrixSize,
                    int iterations);

#endif  // MHC_SINKHORN_H
