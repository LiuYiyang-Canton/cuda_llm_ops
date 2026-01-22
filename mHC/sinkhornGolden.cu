// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// ==============================================================================
#include <algorithm>
#include <cstddef>

#include "sinkhorn.cuh"

/**
 * @brief Normalizes each row of a square matrix in-place so rows sum to 1.
 * @param data Pointer to the matrix data of shape (matrixSize, matrixSize).
 * @param matrixSize Matrix dimension N.
 * @param epsilon Small constant to avoid division by zero.
 */
void NormalizeRows(float* data, int matrixSize, float epsilon) {
    for (int i = 0; i < matrixSize; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; ++j) {
            rowSum += static_cast<double>(data[i * matrixSize + j]);
        }
        double denom = rowSum + epsilon;
        for (int j = 0; j < matrixSize; ++j) {
            data[i * matrixSize + j] = static_cast<float>(data[i * matrixSize + j] / denom);
        }
    }
}

/**
 * @brief Normalizes each column of a square matrix in-place so columns sum to 1.
 * @param data Pointer to the matrix data of shape (matrixSize, matrixSize).
 * @param matrixSize Matrix dimension N.
 * @param epsilon Small constant to avoid division by zero.
 */
void NormalizeCols(float* data, int matrixSize, float epsilon) {
    for (int j = 0; j < matrixSize; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < matrixSize; ++i) {
            colSum += static_cast<double>(data[i * matrixSize + j]);
        }
        double denom = colSum + epsilon;
        for (int i = 0; i < matrixSize; ++i) {
            data[i * matrixSize + j] = static_cast<float>(data[i * matrixSize + j] / denom);
        }
    }
}

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
    int iterations) {
    const size_t matrixElemCount = static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
    const size_t sliceSize = matrixElemCount;
    const size_t totalSlices = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength);
    const float epsilon = 1.0e-6f;

    std::copy(input, input + totalSlices * sliceSize, output);

    for (size_t slice = 0; slice < totalSlices; ++slice) {
        float* matrix = output + slice * sliceSize;
        for (int iter = 0; iter < iterations; ++iter) {
            NormalizeRows(matrix, matrixSize, epsilon);
            NormalizeCols(matrix, matrixSize, epsilon);
        }
    }
}
