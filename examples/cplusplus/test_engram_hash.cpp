// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: C++ test for Engram hash.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

/**
 * @brief Create a matrix filled with a value.
 * @param batchSize Number of rows (batch size).
 * @param sequenceLength Number of columns (sequence length).
 * @param value Fill value for every element.
 * @return Matrix with shape [batchSize][sequenceLength].
 */
Matrix MakeMatrix(int batchSize, int sequenceLength, int64_t value) {
    return Matrix(batchSize, sequenceLength, value);
}

/**
 * @brief Shift input matrix by k to the right, padding on the left.
 * @param x Input matrix with shape [batchSize][sequenceLength].
 * @param k Shift amount (non-negative). k=0 returns x unchanged.
 * @param padId Value used for left padding.
 * @return Shifted matrix with shape [batchSize][sequenceLength].
 */
Matrix ShiftK(const Matrix& x, int k, int64_t padId) {
    const int batchSize = x.batchSize;
    const int sequenceLength = x.sequenceLength;
    Matrix out = MakeMatrix(batchSize, sequenceLength, padId);
    if (k == 0) {
        return x;
    }

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
            const int srcIndex = tokenIndex - k;
            if (srcIndex >= 0) {
                out.At(batchIndex, tokenIndex) = x.At(batchIndex, srcIndex);
            }
        }
    }
    return out;
}

/**
 * @brief Golden reference implementation of GetNgramHashes for a single layer.
 * @param inputIds Input token ids with shape [batchSize][sequenceLength].
 * @param maxNgramSize Max n-gram size (computes n=2..maxNgramSize).
 * @param nHeadPerNgram Number of heads per n-gram size.
 * @param padId Token id used for left padding when shifting.
 * @param multipliers Per-position multipliers with shape [maxNgramSize].
 * @param vocabSizeByNgram Flattened moduli with shape [maxNgramSize-1][nHeadPerNgram].
 * @return Hash tensor with shape [batchSize][sequenceLength][(maxNgramSize-1)*nHeadPerNgram].
 */
Tensor3 GetNgramHashesGolden(
    const Matrix& inputIds,
    int maxNgramSize,
    int nHeadPerNgram,
    int64_t padId,
    const int64_t* multipliers,
    const int64_t* vocabSizeByNgram) {
    const int batchSize = inputIds.batchSize;
    const int sequenceLength = inputIds.sequenceLength;

    std::vector<Matrix> baseShifts;
    baseShifts.reserve(static_cast<size_t>(maxNgramSize));
    for (int k = 0; k < maxNgramSize; ++k) {
        baseShifts.push_back(ShiftK(inputIds, k, padId));
    }

    const int totalHeads = (maxNgramSize - 1) * nHeadPerNgram;
    Tensor3 output(batchSize, sequenceLength, totalHeads, 0);

    int headOffset = 0;
    for (int ngramSize = 2; ngramSize <= maxNgramSize; ++ngramSize) {
        const int ngramIndex = ngramSize - 2;

        Matrix mix = MakeMatrix(batchSize, sequenceLength, 0);
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
                int64_t value = baseShifts[0].At(batchIndex, tokenIndex) * multipliers[0];
                for (int k = 1; k < ngramSize; ++k) {
                    value ^= (baseShifts[k].At(batchIndex, tokenIndex) * multipliers[k]);
                }
                mix.At(batchIndex, tokenIndex) = value;
            }
        }

        for (int headIndex = 0; headIndex < nHeadPerNgram; ++headIndex) {
            const int64_t mod = vocabSizeByNgram[ngramIndex * nHeadPerNgram + headIndex];
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
                for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
                    int64_t value = mix.At(batchIndex, tokenIndex) % mod;
                    if (value < 0) {
                        value += mod;
                    }
                    output.At(batchIndex, tokenIndex, headOffset + headIndex) = value;
                }
            }
        }
        headOffset += nHeadPerNgram;
    }

    return output;
}


/**
 * @brief Demo entry point that generates random inputs and times the hash.
 * @return Exit code (0 on success).
 */
int main() {
    const int batchSize = 1;
    const int sequenceLength = 128 * 1024;
    const int maxNgramSize = 3;
    const int nHeadPerNgram = 8;
    const int64_t padId = 0;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> tokenDist(1, 1000000);
    std::uniform_int_distribution<int> multDist(1, 1000000);
    std::uniform_int_distribution<int> modDist(1000000, 100000000);

    Matrix inputIds = MakeMatrix(batchSize, sequenceLength, 0);
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
            inputIds.At(batchIndex, tokenIndex) = tokenDist(rng);
        }
    }

    std::unique_ptr<int64_t[]> multipliers(new int64_t[maxNgramSize]);
    for (int index = 0; index < maxNgramSize; ++index) {
        multipliers[index] = 2 * multDist(rng) + 1;
    }

    const int vocabSizeLen = (maxNgramSize - 1) * nHeadPerNgram;
    std::unique_ptr<int64_t[]> vocabSizeByNgram(new int64_t[vocabSizeLen]);
    for (int ngramIndex = 0; ngramIndex < maxNgramSize - 1; ++ngramIndex) {
        for (int headIndex = 0; headIndex < nHeadPerNgram; ++headIndex) {
            vocabSizeByNgram[ngramIndex * nHeadPerNgram + headIndex] = modDist(rng);
        }
    }

    const int warmupRuns = 1000;
    const int timedRuns = 1000;

    Tensor3 hashes;
    Tensor3 testHashes;

    for (int i = 0; i < warmupRuns; ++i) {
        hashes = GetNgramHashesGolden(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            padId,
            multipliers.get(),
            vocabSizeByNgram.get());
        testHashes = GetNgramHashes(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            multipliers.get(),
            vocabSizeByNgram.get());
    }

    double totalMsGolden = 0.0;
    auto iterStart = std::chrono::steady_clock::now();
    for (int i = 0; i < timedRuns; ++i) {
        hashes = GetNgramHashesGolden(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            padId,
            multipliers.get(),
            vocabSizeByNgram.get());
    }
    auto iterEnd = std::chrono::steady_clock::now();
    totalMsGolden += std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();
    const double avgMsGolden = totalMsGolden / static_cast<double>(timedRuns);

    double totalMsNgram = 0.0;
    iterStart = std::chrono::steady_clock::now();
    for (int i = 0; i < timedRuns; ++i) {
        testHashes = GetNgramHashes(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            multipliers.get(),
            vocabSizeByNgram.get());
    }
    iterEnd = std::chrono::steady_clock::now();
    totalMsNgram = std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();
    const double avgMsNgram = totalMsNgram / static_cast<double>(timedRuns);

    std::cout << "Hashes shape: batch=" << batchSize << ", seq=" << sequenceLength
              << ", heads=" << nHeadPerNgram << "\n";

    std::cout << "Average golden hash compute time: " << avgMsGolden
              << " ms (" << timedRuns << " runs)\n";
    std::cout << "Average GetNgramHashes time: " << avgMsNgram
              << " ms (" << timedRuns << " runs)\n";

    if (!EqualTensor3(testHashes, hashes)) {
        std::cerr << "Error: GetNgramHashes output does not match golden reference!\n";
        return 1;
    }

    std::cout << "GetNgramHashes output matches golden reference.\n";
    return 0;
}
