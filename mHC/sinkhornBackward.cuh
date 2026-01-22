// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// Purpose: Declares CPU reference Sinkhorn backward routine interfaces.
// ==============================================================================
#pragma once

/**
 * @brief Computes Sinkhorn-Knopp backward using stored row/column sums per iteration.
 * @param output Pointer to X^{(iterations)} with shape (batchSize, seqLength, matrixSize, matrixSize),
 *               overwritten in-place to restore X^{(0)}.
 * @param gradOutput Pointer to gradient with respect to output, same shape as output.
 * @param gradX Output pointer for gradient with respect to X^{(0)}, same shape as output.
 * @param rowSum Pointer to stored row sums with shape (batchSize, seqLength, iterations, matrixSize).
 * @param colSum Pointer to stored column sums with shape (batchSize, seqLength, iterations, matrixSize).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param iterations Number of Sinkhorn-Knopp iterations.
 */
void mHCSinkhornBackwardGolden(float* output,
                               const float* gradOutput,
                               float* gradX,
                               const float* rowSum,
                               const float* colSum,
                               int batchSize,
                               int seqLength,
                               int matrixSize,
                               int iterations);
