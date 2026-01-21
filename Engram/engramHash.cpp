// Build: g++ -std=c++17 -O3 -Wall -Wextra -pedantic Engram/engramHash.cpp Engram/golden.cpp -o engramHash
#include "engramHash.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include "omp.h"

/**
 * @brief C++ implementation of GetNgramHashes for a single layer.
 * @param inputIds Input token ids with shape [B][T].
 * @param maxNgramSize Max n-gram size (computes n=2..maxNgramSize).
 * @param nHeadPerNgram Number of heads per n-gram size.
 * @param multipliers Per-position multipliers with shape [maxNgramSize].
 * @param vocabSizeByNgram Flattened moduli with shape [maxNgramSize-1][nHeadPerNgram].
 * @return Hash tensor with shape [B][T][(maxNgramSize-1)*nHeadPerNgram].
 */
Tensor3 GetNgramHashes(
    const Matrix& inputIds,
    int maxNgramSize,
    int nHeadPerNgram,
    const int64_t* multipliers,               // [maxNgramSize]
    const int64_t* vocabSizeByNgram        // [nGramIndex * nHeadPerNgram + head]
) {
    int B = inputIds.B;
    int T = inputIds.T;
    int totalHeads = (maxNgramSize - 1) * nHeadPerNgram;

    // Declare output tensor.
    Tensor3 output(B, T, totalHeads, 0);

    for (int ibatch = 0; ibatch < B; ++ibatch) {
        for (int itoken = 0; itoken < maxNgramSize; ++itoken) {
            for (int n = 2; n <= maxNgramSize; ++n) {
                int nGramIndex = n - 2;

                // Compute hash for n-gram of size n ending at itoken.
                int64_t hash = 0;
                for (int i = 0; i < std::min(n, itoken + 1); ++i) {
                    int shift = itoken - i;
                    int64_t tokenId = inputIds.at(ibatch, shift);
                    hash ^= multipliers[i] * tokenId;
                }
                for (int ihead = 0; ihead < nHeadPerNgram; ++ihead) {
                    int headIndex = nGramIndex * nHeadPerNgram + ihead;
                    // Modulo by vocab size for this n-gram and head.
                    int64_t mod = vocabSizeByNgram[nGramIndex * nHeadPerNgram + ihead];
                    output.at(ibatch, itoken, headIndex) = hash % mod;
                }
            }
        }
#pragma omp parallel for
        for (int itoken = maxNgramSize; itoken < T; ++itoken) {
            for (int n = 2; n <= maxNgramSize; ++n) {
                int nGramIndex = n - 2;

                // Compute hash for n-gram of size n ending at itoken.
                int64_t hash = (inputIds.at(ibatch, itoken) * multipliers[0]) ^
                               (inputIds.at(ibatch, itoken - 1) * multipliers[1]);
                for (int i = 2; i < n; ++i) {
                    int shift = itoken - i;
                    int64_t tokenId = inputIds.at(ibatch, shift);
                    hash ^= multipliers[i] * tokenId;
                }
                for (int ihead = 0; ihead < nHeadPerNgram; ++ihead) {
                    int headIndex = nGramIndex * nHeadPerNgram + ihead;
                    // Modulo by vocab size for this n-gram and head.
                    int64_t mod = vocabSizeByNgram[nGramIndex * nHeadPerNgram + ihead];
                    output.at(ibatch, itoken, headIndex) = hash % mod < 0 ? hash % mod + mod : hash % mod;
                }
            }
        }
    }

    return output;
}

bool EqualTensor3(const Tensor3& lhs, const Tensor3& rhs) {
    return lhs.B == rhs.B &&
           lhs.T == rhs.T &&
           lhs.H == rhs.H &&
           lhs.data == rhs.data;
}

/**
 * @brief Demo entry point that generates random inputs and times the hash.
 * @return Exit code (0 on success).
 */
int main() {
    // Demo dimensions.
    int B = 1;               // batch size
    int T = 128 * 1024;      // sequence length
    int maxNgramSize = 3;
    int nHeadPerNgram = 8;
    int64_t padId = 0;

    // Random generator.
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> tokenDist(1, 1000000);
    std::uniform_int_distribution<int> multDist(1, 1000000);

    // Random moduli for vocab sizes.
    std::uniform_int_distribution<int> modDist(1000000, 100000000);

    // Random inputIds.
    Matrix inputIds = MakeMatrix(B, T, 0);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            inputIds.at(b, t) = tokenDist(rng);
        }
    }

    // Random multipliers for a single layer.
    std::unique_ptr<int64_t[]> multipliers(new int64_t[maxNgramSize]);
    for (int k = 0; k < maxNgramSize; ++k) {
        multipliers[k] = 2 * multDist(rng) + 1;
    }

    // Random vocab sizes per n-gram.
    // For each n-gram index, we store a vector of head mod values.
    int vocabSizeLen = (maxNgramSize - 1) * nHeadPerNgram;
    std::unique_ptr<int64_t[]> vocabSizeByNgram(new int64_t[vocabSizeLen]);
    for (int nGramIndex = 0; nGramIndex < maxNgramSize - 1; ++nGramIndex) {
        for (int j = 0; j < nHeadPerNgram; ++j) {
            vocabSizeByNgram[nGramIndex * nHeadPerNgram + j] = modDist(rng);
        }
    }

    // Warm up and then time several runs to get an average.
    int warmupRuns = 1000;
    int timedRuns = 1000;
    Tensor3 hashes;
    Tensor3 testHashes;

    for (int i = 0; i < warmupRuns; ++i) {
        hashes = GetNgramHashesGolden(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            padId,
            multipliers.get(),
            vocabSizeByNgram.get()
        );
        testHashes = GetNgramHashes(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            multipliers.get(),
            vocabSizeByNgram.get()
        );
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
            vocabSizeByNgram.get()
        );
    }
    auto iterEnd = std::chrono::steady_clock::now();
    totalMsGolden += std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();
    double avgMsGolden = totalMsGolden / static_cast<double>(timedRuns);


    double totalMsNgram = 0.0;
    iterStart = std::chrono::steady_clock::now();
    for (int i = 0; i < timedRuns; ++i) {
        testHashes = GetNgramHashes(
            inputIds,
            maxNgramSize,
            nHeadPerNgram,
            multipliers.get(),
            vocabSizeByNgram.get()
        );
    }
    iterEnd = std::chrono::steady_clock::now();
    totalMsNgram = std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();
    double avgMsNgram = totalMsNgram / static_cast<double>(timedRuns);

    std::cout << "Hashes shape: B=" << B << ", T=" << T
              << ", heads=" << nHeadPerNgram << "\n";


    std::cout << "Average golden hash compute time: " << avgMsGolden << " ms (" << timedRuns << " runs)\n";
    std::cout << "Average GetNgramHashes time: " << avgMsNgram << " ms (" << timedRuns << " runs)\n";

    // Verify that the outputs match.
    if (!EqualTensor3(testHashes, hashes)) {
        std::cerr << "Error: GetNgramHashes output does not match golden reference!\n";
        return 1;
    } else {
        std::cout << "GetNgramHashes output matches golden reference.\n";
    }

    return 0;
}
