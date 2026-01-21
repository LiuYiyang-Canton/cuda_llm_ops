#ifndef ENGRAM_HASH_H
#define ENGRAM_HASH_H

#include <cstdint>
#include <vector>

/**
 * @brief Contiguous 2D matrix with row-major storage.
 */
struct Matrix {
    int B;
    int T;
    std::vector<int64_t> data;

    Matrix() : B(0), T(0), data() {}
    Matrix(int bSize, int tSize, int64_t value = 0)
        : B(bSize), T(tSize), data(static_cast<size_t>(bSize) * tSize, value) {}

    inline int64_t& at(int b, int t) {
        return data[static_cast<size_t>(b) * T + t];
    }
    inline const int64_t& at(int b, int t) const {
        return data[static_cast<size_t>(b) * T + t];
    }
};

/**
 * @brief Contiguous 3D tensor with row-major storage.
 */
struct Tensor3 {
    int B;
    int T;
    int H;
    std::vector<int64_t> data;

    Tensor3() : B(0), T(0), H(0), data() {}
    Tensor3(int bSize, int tSize, int hSize, int64_t value = 0)
        : B(bSize), T(tSize), H(hSize),
          data(static_cast<size_t>(bSize) * tSize * hSize, value) {}

    inline int64_t& at(int b, int t, int h) {
        return data[(static_cast<size_t>(b) * T + t) * H + h];
    }
    inline const int64_t& at(int b, int t, int h) const {
        return data[(static_cast<size_t>(b) * T + t) * H + h];
    }
};

/**
 * @brief Create a BxT matrix filled with a value.
 * @param B Number of rows (batch size).
 * @param T Number of columns (sequence length).
 * @param value Fill value for every element.
 * @return Matrix with shape [B][T].
 */
Matrix MakeMatrix(int B, int T, int64_t value = 0);

/**
 * @brief Shift input matrix by k to the right, padding on the left.
 * @param x Input matrix with shape [B][T].
 * @param k Shift amount (non-negative). k=0 returns x unchanged.
 * @param padId Value used for left padding.
 * @return Shifted matrix with shape [B][T].
 *
 * This matches the Python np.pad + slice behavior in shift_k.
 */
Matrix ShiftK(const Matrix& x, int k, int64_t padId);

/**
 * @brief Golden reference implementation of GetNgramHashes for a single layer.
 * @param inputIds Input token ids with shape [B][T].
 * @param maxNgramSize Max n-gram size (computes n=2..maxNgramSize).
 * @param nHeadPerNgram Number of heads per n-gram size.
 * @param padId Token id used for left padding when shifting.
 * @param multipliers Per-position multipliers with shape [maxNgramSize].
 * @param vocabSizeByNgram Flattened moduli with shape [maxNgramSize-1][nHeadPerNgram].
 * @return Hash tensor with shape [B][T][(maxNgramSize-1)*nHeadPerNgram].
 */
Tensor3 GetNgramHashesGolden(
    const Matrix& inputIds,
    int maxNgramSize,
    int nHeadPerNgram,
    int64_t padId,
    const int64_t* multipliers,               // [maxNgramSize]
    const int64_t* vocabSizeByNgram        // [nGramIndex * nHeadPerNgram + head]
);

#endif  // ENGRAM_HASH_H
