// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// Purpose: Reference CPU implementation for Sinkhorn backward pass
// ==============================================================================
#include <algorithm>
#include <cassert>

#include "sinkhornBackward.cuh"

/**
 * @brief Backpropagates column normalization in-place and reconstructs X^{(k+1/2)}.
 * @param xNextInOut Pointer to X^{(k+1)} input, overwritten with X^{(k+1/2)}.
 * @param gradNextInOut Pointer to gradient wrt X^{(k+1)}, overwritten with gradient wrt X^{(k+1/2)}.
 * @param colSum Pointer to column sums c_j^{(k)} (length matrixSize), including epsilon.
 * @param matrixSize Matrix dimension N.
 */
void NormalizeColumnsBackward(float* xNextInOut, float* gradNextInOut, const float* colSum, int matrixSize) {
    // loop over columns
    for (int j = 0; j < matrixSize; ++j) {
        float sumProd = 0.0f;

        // loop over rows to compute sum of products
        for (int i = 0; i < matrixSize; ++i) {
            sumProd += gradNextInOut[i * matrixSize + j] * xNextInOut[i * matrixSize + j];
        }

        const float denom = colSum[j];

        // loop over rows
        for (int i = 0; i < matrixSize; ++i) {
            xNextInOut[i * matrixSize + j] = xNextInOut[i * matrixSize + j] * denom;
            gradNextInOut[i * matrixSize + j] = (gradNextInOut[i * matrixSize + j] - sumProd) / denom;
        }
    }
}

/**
 * @brief Backpropagates row normalization in-place and reconstructs X^{(k)}.
 * @param xNextInOut Pointer to X^{(k+1/2)} input, overwritten with X^{(k)}.
 * @param gradHalfInOut Pointer to gradient wrt X^{(k+1/2)}, overwritten with gradient wrt X^{(k)}.
 * @param rowSum Pointer to row sums r_i^{(k)} (length matrixSize), including epsilon.
 * @param matrixSize Matrix dimension N.
 */
void NormalizeRowsBackward(float* xNextInOut,
                               float* gradNextInOut,
                               const float* rowSum,
                               int matrixSize) {
    // loop over rows
    for (int i = 0; i < matrixSize; ++i) {
        // loop over columns to compute sum of products
        float sumProd = 0.0f;
        for (int j = 0; j < matrixSize; ++j) {
            sumProd += gradNextInOut[i * matrixSize + j] * xNextInOut[i * matrixSize + j];
        }

        // loop over columns
        const float denom = rowSum[i];
        for (int j = 0; j < matrixSize; ++j) {
            xNextInOut[i * matrixSize + j] = xNextInOut[i * matrixSize + j] * denom;
            gradNextInOut[i * matrixSize + j] = (gradNextInOut[i * matrixSize + j] - sumProd) / denom;
        }
    }
}

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
                               int iterations) {
    for (int b = 0; b < batchSize; ++b) {
        for (int s = 0; s < seqLength; ++s) {
            // Pointers to the current matrix in the batch and sequence
            float* xNextInOut = output + ((b * seqLength + s) * matrixSize * matrixSize);
            float* gradXOut = gradX + ((b * seqLength + s) * matrixSize * matrixSize);

            // before starting backpropagation, copy gradOutput to gradXOut
            std::copy_n(gradOutput + ((b * seqLength + s) * matrixSize * matrixSize),
                          matrixSize * matrixSize,
                          gradXOut);

            // Backpropagate through iterations in reverse order
            for (int iter = iterations - 1; iter >= 0; --iter) {
                const float* currentColSum = colSum + (iter * batchSize * seqLength * matrixSize + \
                                                        b * seqLength * matrixSize + s * matrixSize);
                const float* currentRowSum = rowSum + (iter * batchSize * seqLength * matrixSize + \
                                                        b * seqLength * matrixSize + s * matrixSize);

                // Backpropagate column normalization
                NormalizeColumnsBackward(xNextInOut, gradXOut, currentColSum, matrixSize);

                // Backpropagate row normalization
                NormalizeRowsBackward(xNextInOut, gradXOut, currentRowSum, matrixSize);
            }
        }
    }
}
