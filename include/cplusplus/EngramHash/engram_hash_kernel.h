// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose:
// Build:
// ==============================================================================
#pragma once

#include <cstdint>
#include <vector>

/**
 * @brief Contiguous 2D matrix with row-major storage.
 */
struct Matrix {
    int batchSize;
    int sequenceLength;
    std::vector<int64_t> data;

    /**
     * @brief Constructs an empty matrix with zero dimensions.
     */
    Matrix();

    /**
     * @brief Constructs a matrix of shape [batchSize][sequenceLength] filled with value.
     * @param batchSize Number of rows (batch size).
     * @param sequenceLength Number of columns (sequence length).
     * @param value Fill value for every element.
     */
    Matrix(int batchSize, int sequenceLength, int64_t value = 0);

    /**
     * @brief Returns a mutable reference to an element.
     * @param batchIndex Batch index.
     * @param tokenIndex Token index.
     * @return Reference to the selected element.
     */
    int64_t& At(int batchIndex, int tokenIndex);

    /**
     * @brief Returns a const reference to an element.
     * @param batchIndex Batch index.
     * @param tokenIndex Token index.
     * @return Const reference to the selected element.
     */
    const int64_t& At(int batchIndex, int tokenIndex) const;
};

/**
 * @brief Contiguous 3D tensor with row-major storage.
 */
struct Tensor3 {
    int batchSize;
    int sequenceLength;
    int headCount;
    std::vector<int64_t> data;

    /**
     * @brief Constructs an empty tensor with zero dimensions.
     */
    Tensor3();

    /**
     * @brief Constructs a tensor of shape [batchSize][sequenceLength][headCount] filled with value.
     * @param batchSize Number of batches.
     * @param sequenceLength Sequence length.
     * @param headCount Number of heads.
     * @param value Fill value for every element.
     */
    Tensor3(int batchSize, int sequenceLength, int headCount, int64_t value = 0);

    /**
     * @brief Returns a mutable reference to an element.
     * @param batchIndex Batch index.
     * @param tokenIndex Token index.
     * @param headIndex Head index.
     * @return Reference to the selected element.
     */
    int64_t& At(int batchIndex, int tokenIndex, int headIndex);

    /**
     * @brief Returns a const reference to an element.
     * @param batchIndex Batch index.
     * @param tokenIndex Token index.
     * @param headIndex Head index.
     * @return Const reference to the selected element.
     */
    const int64_t& At(int batchIndex, int tokenIndex, int headIndex) const;
};

/**
 * @brief C++ implementation of GetNgramHashes for a single layer.
 * @param inputIds Input token ids with shape [batchSize][sequenceLength].
 * @param maxNgramSize Max n-gram size (computes n=2..maxNgramSize).
 * @param nHeadPerNgram Number of heads per n-gram size.
 * @param multipliers Per-position multipliers with shape [maxNgramSize].
 * @param vocabSizeByNgram Flattened moduli with shape [maxNgramSize-1][nHeadPerNgram].
 * @return Hash tensor with shape [batchSize][sequenceLength][(maxNgramSize-1)*nHeadPerNgram].
 */
Tensor3 GetNgramHashes(
    const Matrix& inputIds,
    int maxNgramSize,
    int nHeadPerNgram,
    const int64_t* multipliers,
    const int64_t* vocabSizeByNgram);

/**
 * @brief Checks whether two Tensor3 values are identical.
 * @param lhs First tensor.
 * @param rhs Second tensor.
 * @return True if shapes and data match.
 */
bool EqualTensor3(const Tensor3& lhs, const Tensor3& rhs);
