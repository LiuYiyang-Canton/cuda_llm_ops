#include "engramHash.h"

// Helper to create a BxT matrix filled with a value.
Matrix MakeMatrix(int B, int T, int64_t value) {
    return Matrix(B, T, value);
}

// Shift input matrix by k to the right, padding on the left with padId.
// This matches the Python np.pad + slice behavior in ShiftK.
Matrix ShiftK(const Matrix& x, int k, int64_t padId) {
    int B = x.B;
    int T = x.T;
    Matrix out = MakeMatrix(B, T, padId);
    if (k == 0) return x;

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int srcT = t - k;
            if (srcT >= 0) {
                out.at(b, t) = x.at(b, srcT);
            }
        }
    }
    return out;
}

// Golden reference implementation of GetNgramHashes for a single layer.
// Returns a [B][T][numHeadsTotal] tensor.
Tensor3 GetNgramHashesGolden(
    const Matrix& inputIds,
    int maxNgramSize,
    int nHeadPerNgram,
    int64_t padId,
    const int64_t* multipliers,               // [maxNgramSize]
    const int64_t* vocabSizeByNgram        // [nGramIndex * nHeadPerNgram + head]
) {
    int B = inputIds.B;
    int T = inputIds.T;

    // Precompute shifted versions of the input.
    std::vector<Matrix> baseShifts;
    for (int k = 0; k < maxNgramSize; ++k) {
        baseShifts.push_back(ShiftK(inputIds, k, padId));
    }

    // Total heads = (maxNgramSize - 1) * nHeadPerNgram.
    int totalHeads = (maxNgramSize - 1) * nHeadPerNgram;
    Tensor3 output(B, T, totalHeads, 0);

    int headOffset = 0;
    for (int n = 2; n <= maxNgramSize; ++n) {
        int nGramIndex = n - 2;

        // Build mix = tokens[0] * mult[0] XOR tokens[1] * mult[1] ...
        Matrix mix = MakeMatrix(B, T, 0);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                // Initialize with first token * multiplier.
                int64_t v = baseShifts[0].at(b, t) * multipliers[0];
                // XOR remaining tokens * multipliers.
                for (int k = 1; k < n; ++k) {
                    // Note: possible overflow but that's actually a feature for hashing.
                    v ^= (baseShifts[k].at(b, t) * multipliers[k]);
                }
                mix.at(b, t) = v;
            }
        }

        // Apply modulo per head and store in output.
        for (int j = 0; j < nHeadPerNgram; ++j) {
            int64_t mod = vocabSizeByNgram[nGramIndex * nHeadPerNgram + j];
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    // C++ % can be negative if mix is negative; make it non-negative for safety.
                    int64_t val = mix.at(b, t) % mod;
                    if (val < 0) val += mod;
                    output.at(b, t, headOffset + j) = val;
                }
            }
        }
        headOffset += nHeadPerNgram;
    }

    return output;
}
