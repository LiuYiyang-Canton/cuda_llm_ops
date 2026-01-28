// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for TopK.
// ==============================================================================
#include "TopK/topk_kernel.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kRadixBits = 8;
constexpr int kRadixHistogramSize = 1 << kRadixBits;
constexpr int kRadixPasses = 32 / kRadixBits;
constexpr int kWorkPerThread = 16;
constexpr int kWorkPerBlock = kThreadsPerBlock * kWorkPerThread;

/**
 * @brief Returns ceil(numerator / denominator) for positive integers.
 * @param numerator Dividend value.
 * @param denominator Divisor value; must be > 0.
 * @return Rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Kernel to perform one pass of radix select for top-k.
 * @param input Pointer to input values (modified in-place on first pass).
 * @param histogram Pointer to per-batch histogram output.
 * @param targetTopK Pointer to per-batch remaining top-k counts.
 * @param selectedRadix Pointer to per-batch selected radix prefix.
 * @param iterPass Zero-based radix pass index.
 * @param batchSize Number of rows to process.
 * @param featureSize Number of columns per row.
 */
__global__ void RadixSelectTopKFp32Kernel(float* __restrict__ input,
                                          int* __restrict__ histogram,
                                          const int* __restrict__ targetTopK,
                                          const uint32_t* __restrict__ selectedRadix,
                                          int iterPass,
                                          int batchSize,
                                          int featureSize) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int chunkStart = static_cast<int>(blockIdx.x) * kWorkPerBlock;
    const int chunkStop = (static_cast<int>(blockIdx.x) + 1) * kWorkPerBlock;
    const int thread = static_cast<int>(threadIdx.x);

    if (batchIdx >= batchSize) {
        return;
    }

    input += batchIdx * featureSize;
    histogram += batchIdx * kRadixHistogramSize;
    targetTopK += batchIdx;
    selectedRadix += batchIdx;

    if (*targetTopK == 0) {
        return;
    }

    __shared__ int sharedHistogram[kRadixHistogramSize];
    if (thread < kRadixHistogramSize) {
        sharedHistogram[thread] = 0;
    }
    __syncthreads();

    const int shiftBits = (kRadixPasses - iterPass - 1) * kRadixBits;
    const uint32_t mask = (1U << kRadixBits) - 1;
    if (iterPass == 0) {
        for (int i = chunkStart + thread * 4; i < chunkStop; i += static_cast<int>(blockDim.x) * 4) {
            if (i >= featureSize) {
                break;
            }

            uint4 uval = *reinterpret_cast<uint4*>(&input[i]);
            if (~(uval.x >> 31)) {
                uval.x |= (1u << 31);
            } else {
                uval.x = ~uval.x;
            }
            if (~(uval.y >> 31)) {
                uval.y |= (1u << 31);
            } else {
                uval.y = ~uval.y;
            }
            if (~(uval.z >> 31)) {
                uval.z |= (1u << 31);
            } else {
                uval.z = ~uval.z;
            }
            if (~(uval.w >> 31)) {
                uval.w |= (1u << 31);
            } else {
                uval.w = ~uval.w;
            }

            *reinterpret_cast<uint4*>(&input[i]) = uval;

            uval.x = uval.x >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << kRadixBits) - uval.x - 1], 1);
            uval.y = uval.y >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << kRadixBits) - uval.y - 1], 1);
            uval.z = uval.z >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << kRadixBits) - uval.z - 1], 1);
            uval.w = uval.w >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << kRadixBits) - uval.w - 1], 1);
        }
    } else {
        for (int i = chunkStart + thread * 4; i < chunkStop; i += static_cast<int>(blockDim.x) * 4) {
            if (i >= featureSize) {
                break;
            }

            const uint4 uval = *reinterpret_cast<uint4*>(&input[i]);
            if (uval.x >> (shiftBits + kRadixBits) == *selectedRadix) {
                const uint32_t bucket = (uval.x >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << kRadixBits) - bucket - 1], 1);
            }
            if (uval.y >> (shiftBits + kRadixBits) == *selectedRadix) {
                const uint32_t bucket = (uval.y >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << kRadixBits) - bucket - 1], 1);
            }
            if (uval.z >> (shiftBits + kRadixBits) == *selectedRadix) {
                const uint32_t bucket = (uval.z >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << kRadixBits) - bucket - 1], 1);
            }
            if (uval.w >> (shiftBits + kRadixBits) == *selectedRadix) {
                const uint32_t bucket = (uval.w >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << kRadixBits) - bucket - 1], 1);
            }
        }
    }

    __syncthreads();

    if (thread < kRadixHistogramSize) {
        for (int offset = 1; offset < kRadixHistogramSize; offset *= 2) {
            if ((thread + 1) % (offset * 2) == 0) {
                sharedHistogram[thread] += sharedHistogram[thread - offset];
            }
            __syncthreads();
        }

        if (thread == kRadixHistogramSize - 1) {
            sharedHistogram[thread] = 0;
        }
        __syncthreads();
        for (int offset = kRadixHistogramSize / 2; offset >= 1; offset /= 2) {
            if ((thread + 1) % (offset * 2) == 0) {
                const int temp = sharedHistogram[thread - offset];
                sharedHistogram[thread - offset] = sharedHistogram[thread];
                sharedHistogram[thread] += temp;
            }
            __syncthreads();
        }

        if (thread < kRadixHistogramSize - 1) {
            atomicAdd(&histogram[thread], sharedHistogram[thread + 1]);
        }
    }
}

/**
 * @brief Kernel to determine the radix bucket that contains the top-k elements.
 * @param histogram Pointer to per-batch histogram buffer.
 * @param targetTopK Pointer to per-batch remaining top-k counts.
 * @param selectedRadix Pointer to per-batch selected radix prefix.
 * @param iterPass Zero-based radix pass index.
 * @param batchSize Number of rows to process.
 */
__global__ void FindRadixBucketKernel(int* __restrict__ histogram,
                                      int* __restrict__ targetTopK,
                                      uint32_t* __restrict__ selectedRadix,
                                      int iterPass,
                                      int batchSize) {
    const int batchIdx = static_cast<int>(blockIdx.x);
    if (batchIdx >= batchSize) {
        return;
    }
    histogram += batchIdx * kRadixHistogramSize;
    targetTopK += batchIdx;
    selectedRadix += batchIdx;

    if (*targetTopK == 0) {
        return;
    }

    const int thread = static_cast<int>(threadIdx.x);

    if (thread < kRadixHistogramSize) {
        if (thread == 0 && histogram[thread] >= *targetTopK) {
            *selectedRadix <<= kRadixBits;
            *selectedRadix |= (1 << kRadixBits) - thread - 1;
        } else if (thread > 0 && histogram[thread - 1] < *targetTopK && histogram[thread] >= *targetTopK) {
            *selectedRadix <<= kRadixBits;
            *selectedRadix |= (1 << kRadixBits) - thread - 1;
            *targetTopK -= histogram[thread - 1];
            if (*targetTopK == 0) {
                *selectedRadix <<= (kRadixPasses - iterPass - 1) * kRadixBits;
            }
        }
    }

    if (thread < kRadixHistogramSize) {
        histogram[thread] = 0;
    }
}

/**
 * @brief Final kernel to gather top-k indices given the selected radix.
 * @param input Pointer to input values.
 * @param topkMask Pointer to output boolean mask.
 * @param selectedRadix Pointer to per-batch selected radix prefix.
 * @param batchSize Number of rows to process.
 * @param featureSize Number of columns per row.
 */
__global__ void GatherTopKIndicesKernel(const float* __restrict__ input,
                                        bool* __restrict__ topkMask,
                                        const uint32_t* __restrict__ selectedRadix,
                                        int batchSize,
                                        int featureSize) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int chunkStart = static_cast<int>(blockIdx.x) * kWorkPerBlock;
    const int chunkStop = (static_cast<int>(blockIdx.x) + 1) * kWorkPerBlock;
    const int thread = static_cast<int>(threadIdx.x);

    if (batchIdx >= batchSize) {
        return;
    }

    input += batchIdx * featureSize;
    topkMask += batchIdx * featureSize;
    selectedRadix += batchIdx;

    for (int i = chunkStart + thread * 4; i < chunkStop; i += static_cast<int>(blockDim.x) * 4) {
        if (i >= featureSize) {
            break;
        }
        const uint4 uval = *reinterpret_cast<const uint4*>(&input[i]);
        topkMask[i] = (uval.x >= *selectedRadix);
        topkMask[i + 1] = (uval.y >= *selectedRadix);
        topkMask[i + 2] = (uval.z >= *selectedRadix);
        topkMask[i + 3] = (uval.w >= *selectedRadix);
    }
}

}  // namespace

/**
 * @brief Launches the radix-select top-k kernels to produce a boolean mask of top-k indices.
 * @param input Pointer to device input buffer (modified in-place).
 * @param topkMask Pointer to device output boolean mask [batchSize, featureSize].
 * @param histogram Pointer to device histogram buffer [batchSize, kRadixHistogramSize].
 * @param targetTopK Pointer to device target-k buffer [batchSize].
 * @param selectedRadix Pointer to device selected radix buffer [batchSize].
 * @param batchSize Number of rows to process.
 * @param featureSize Number of columns per row.
 */
void LaunchTopkRadixSelectFp32(float* input,
                               bool* topkMask,
                               int* histogram,
                               int* targetTopK,
                               uint32_t* selectedRadix,
                               int batchSize,
                               int featureSize) {
    if (batchSize <= 0 || featureSize <= 0) {
        return;
    }
    if (featureSize % 4 != 0) {
        return;
    }

    const dim3 numBlocks(CeilDiv(featureSize, kWorkPerBlock), batchSize);
    const dim3 numThreads(kThreadsPerBlock);

    for (int iterPass = 0; iterPass < kRadixPasses; ++iterPass) {
        RadixSelectTopKFp32Kernel<<<numBlocks, numThreads>>>(input,
                                                             histogram,
                                                             targetTopK,
                                                             selectedRadix,
                                                             iterPass,
                                                             batchSize,
                                                             featureSize);
        FindRadixBucketKernel<<<batchSize, kRadixHistogramSize>>>(histogram,
                                                                  targetTopK,
                                                                  selectedRadix,
                                                                  iterPass,
                                                                  batchSize);
    }

    GatherTopKIndicesKernel<<<numBlocks, numThreads>>>(input,
                                                       topkMask,
                                                       selectedRadix,
                                                       batchSize,
                                                       featureSize);
}
