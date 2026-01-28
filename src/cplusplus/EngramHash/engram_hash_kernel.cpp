// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: C++ implementation for Engram hash.
// ==============================================================================
#include "EngramHash/engram_hash_kernel.h"

#include <algorithm>

/**
 * @brief Constructs an empty matrix with zero dimensions.
 */
Matrix::Matrix() : batchSize(0), sequenceLength(0), data() {}

/**
 * @brief Constructs a matrix of shape [batchSize][sequenceLength] filled with value.
 * @param batchSize Number of rows (batch size).
 * @param sequenceLength Number of columns (sequence length).
 * @param value Fill value for every element.
 */
Matrix::Matrix(int batchSize, int sequenceLength, int64_t value)
    : batchSize(batchSize),
      sequenceLength(sequenceLength),
      data(static_cast<size_t>(batchSize) * sequenceLength, value) {}

/**
 * @brief Returns a mutable reference to an element.
 * @param batchIndex Batch index.
 * @param tokenIndex Token index.
 * @return Reference to the selected element.
 */
int64_t& Matrix::At(int batchIndex, int tokenIndex) {
    return data[static_cast<size_t>(batchIndex) * sequenceLength + tokenIndex];
}

/**
 * @brief Returns a const reference to an element.
 * @param batchIndex Batch index.
 * @param tokenIndex Token index.
 * @return Const reference to the selected element.
 */
const int64_t& Matrix::At(int batchIndex, int tokenIndex) const {
    return data[static_cast<size_t>(batchIndex) * sequenceLength + tokenIndex];
}

/**
 * @brief Constructs an empty tensor with zero dimensions.
 */
Tensor3::Tensor3() : batchSize(0), sequenceLength(0), headCount(0), data() {}

/**
 * @brief Constructs a tensor of shape [batchSize][sequenceLength][headCount] filled with value.
 * @param batchSize Number of batches.
 * @param sequenceLength Sequence length.
 * @param headCount Number of heads.
 * @param value Fill value for every element.
 */
Tensor3::Tensor3(int batchSize, int sequenceLength, int headCount, int64_t value)
    : batchSize(batchSize),
      sequenceLength(sequenceLength),
      headCount(headCount),
      data(static_cast<size_t>(batchSize) * sequenceLength * headCount, value) {}

/**
 * @brief Returns a mutable reference to an element.
 * @param batchIndex Batch index.
 * @param tokenIndex Token index.
 * @param headIndex Head index.
 * @return Reference to the selected element.
 */
int64_t& Tensor3::At(int batchIndex, int tokenIndex, int headIndex) {
    return data[(static_cast<size_t>(batchIndex) * sequenceLength + tokenIndex) * headCount + headIndex];
}

/**
 * @brief Returns a const reference to an element.
 * @param batchIndex Batch index.
 * @param tokenIndex Token index.
 * @param headIndex Head index.
 * @return Const reference to the selected element.
 */
const int64_t& Tensor3::At(int batchIndex, int tokenIndex, int headIndex) const {
    return data[(static_cast<size_t>(batchIndex) * sequenceLength + tokenIndex) * headCount + headIndex];
}

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
    const int64_t* vocabSizeByNgram) {
    const int batchSize = inputIds.batchSize;
    const int sequenceLength = inputIds.sequenceLength;
    const int totalHeads = (maxNgramSize - 1) * nHeadPerNgram;

    Tensor3 output(batchSize, sequenceLength, totalHeads, 0);

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        for (int tokenIndex = 0; tokenIndex < maxNgramSize; ++tokenIndex) {
            for (int ngramSize = 2; ngramSize <= maxNgramSize; ++ngramSize) {
                const int ngramIndex = ngramSize - 2;

                int64_t hash = 0;
                for (int offset = 0; offset < std::min(ngramSize, tokenIndex + 1); ++offset) {
                    const int shift = tokenIndex - offset;
                    const int64_t tokenId = inputIds.At(batchIndex, shift);
                    hash ^= multipliers[offset] * tokenId;
                }
                for (int headIndex = 0; headIndex < nHeadPerNgram; ++headIndex) {
                    const int outputIndex = ngramIndex * nHeadPerNgram + headIndex;
                    const int64_t mod = vocabSizeByNgram[ngramIndex * nHeadPerNgram + headIndex];
                    output.At(batchIndex, tokenIndex, outputIndex) = hash % mod;
                }
            }
        }
#pragma omp parallel for
        for (int tokenIndex = maxNgramSize; tokenIndex < sequenceLength; ++tokenIndex) {
            for (int ngramSize = 2; ngramSize <= maxNgramSize; ++ngramSize) {
                const int ngramIndex = ngramSize - 2;

                int64_t hash = (inputIds.At(batchIndex, tokenIndex) * multipliers[0]) ^
                               (inputIds.At(batchIndex, tokenIndex - 1) * multipliers[1]);
                for (int offset = 2; offset < ngramSize; ++offset) {
                    const int shift = tokenIndex - offset;
                    const int64_t tokenId = inputIds.At(batchIndex, shift);
                    hash ^= multipliers[offset] * tokenId;
                }
                for (int headIndex = 0; headIndex < nHeadPerNgram; ++headIndex) {
                    const int outputIndex = ngramIndex * nHeadPerNgram + headIndex;
                    const int64_t mod = vocabSizeByNgram[ngramIndex * nHeadPerNgram + headIndex];
                    const int64_t reduced = hash % mod;
                    output.At(batchIndex, tokenIndex, outputIndex) =
                        reduced < 0 ? reduced + mod : reduced;
                }
            }
        }
    }

    return output;
}

/**
 * @brief Checks whether two Tensor3 values are identical.
 * @param lhs First tensor.
 * @param rhs Second tensor.
 * @return True if shapes and data match.
 */
bool EqualTensor3(const Tensor3& lhs, const Tensor3& rhs) {
    return lhs.batchSize == rhs.batchSize &&
           lhs.sequenceLength == rhs.sequenceLength &&
           lhs.headCount == rhs.headCount &&
           lhs.data == rhs.data;
}
