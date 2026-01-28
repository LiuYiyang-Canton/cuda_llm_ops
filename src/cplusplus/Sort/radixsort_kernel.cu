// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for radix sort.
// ==============================================================================
#include "Sort/radixsort_kernel.cuh"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kRadixBits = 8;
constexpr int kRadixHistogramSize = 1 << kRadixBits;
constexpr int kRadixPasses = 32 / kRadixBits;
constexpr int kWorkPerThread = 16;
constexpr int kThreadsPerBlock = 256;
constexpr int kWorkPerBlock = kThreadsPerBlock * kWorkPerThread;
constexpr int kWarpSize = 32;
constexpr int kNumWarpsPerBlock = kThreadsPerBlock / kWarpSize;

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
 * @brief Transforms raw float bits into a monotonic sortable key.
 * @param bits Raw IEEE754 bits from a float.
 * @return Key where lexicographic order matches float ordering.
 */
__device__ __forceinline__ uint32_t FloatToSortKey(uint32_t bits) {
    const uint32_t sign = bits & 0x80000000u;
    return bits ^ (sign ? 0xFFFFFFFFu : 0x80000000u);
}

/**
 * @brief Converts a sortable key produced by FloatToSortKey back to the original float.
 * @param keyBits Encoded sortable key.
 * @return Reconstructed floating-point value.
 */
__device__ __forceinline__ float SortKeyToFloat(uint32_t keyBits) {
    const uint32_t sign = keyBits & 0x80000000u;
    const uint32_t raw = sign ? (keyBits ^ 0x80000000u) : (keyBits ^ 0xFFFFFFFFu);
    return __uint_as_float(raw);
}

/**
 * @brief Copies a contiguous fp32 buffer on the device.
 * @param src Source device pointer.
 * @param dst Destination device pointer.
 * @param count Number of float elements to copy.
 */
__global__ void CopyDeviceBuffer(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

/**
 * @brief Builds per-block histograms for a single radix pass.
 * @param input Input data (in-place keys for pass 0).
 * @param histogram Output histogram buffer shaped [batch, blocks, kRadixHistogramSize].
 * @param iterPass Zero-based radix pass index.
 * @param arrayLength Number of elements per row.
 * @param numBlocksPerBatch Number of blocks per batch dimension.
 */
__global__ void RadixSortBuildHistogram(float* __restrict__ input,
                                        int* __restrict__ histogram,
                                        int iterPass,
                                        int arrayLength,
                                        int numBlocksPerBatch) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int chunkStart = static_cast<int>(blockIdx.x) * kWorkPerBlock;
    const int chunkStop = (static_cast<int>(blockIdx.x) + 1) * kWorkPerBlock;
    const int blockX = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);

    input += batchIdx * arrayLength;
    histogram += batchIdx * numBlocksPerBatch * kRadixHistogramSize + blockX * kRadixHistogramSize;

    __shared__ int sharedHistogram[kRadixHistogramSize];
    if (thread < kRadixHistogramSize) {
        sharedHistogram[thread] = 0;
    }
    __syncthreads();

    const int shiftBits = iterPass * kRadixBits;
    const uint32_t mask = (1U << kRadixBits) - 1;
    if (iterPass == 0) {
        for (int i = chunkStart + thread * 4; i < chunkStop; i += static_cast<int>(blockDim.x) * 4) {
            if (i >= arrayLength) {
                break;
            }
            const uint4 uvalF = *reinterpret_cast<const uint4*>(&input[i]);
            uint4 uval{};
            uval.x = FloatToSortKey(uvalF.x);
            uval.y = FloatToSortKey(uvalF.y);
            uval.z = FloatToSortKey(uvalF.z);
            uval.w = FloatToSortKey(uvalF.w);

            *reinterpret_cast<uint4*>(&input[i]) = uval;

            atomicAdd(&sharedHistogram[(uval.x >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.y >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.z >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.w >> shiftBits) & mask], 1);
        }
    } else {
        for (int i = chunkStart + thread * 4; i < chunkStop; i += static_cast<int>(blockDim.x) * 4) {
            if (i >= arrayLength) {
                break;
            }
            const uint4 uval = *reinterpret_cast<const uint4*>(&input[i]);
            atomicAdd(&sharedHistogram[(uval.x >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.y >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.z >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.w >> shiftBits) & mask], 1);
        }
    }

    __syncthreads();
    if (thread < kRadixHistogramSize) {
        histogram[thread] = sharedHistogram[thread];
    }
}

/**
 * @brief Computes an exclusive prefix sum over per-block histograms in-place.
 * @param histogram Histogram buffer shaped [batch, blocks, kRadixHistogramSize].
 * @param numBlocksPerBatch Number of blocks per batch dimension.
 */
__global__ void RadixSortPrefixSum(int* histogram, int numBlocksPerBatch) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int bucket = static_cast<int>(threadIdx.x);

    __shared__ int bucketTotals[kRadixHistogramSize];

    if (bucket >= kRadixHistogramSize || blockIdx.x != 0) {
        return;
    }

    histogram += batchIdx * numBlocksPerBatch * kRadixHistogramSize;

    int runningTotal = 0;
    for (int block = 0; block < numBlocksPerBatch; ++block) {
        const int count = histogram[block * kRadixHistogramSize + bucket];
        histogram[block * kRadixHistogramSize + bucket] = runningTotal;
        runningTotal += count;
    }
    bucketTotals[bucket] = runningTotal;
    __syncthreads();

    int bucketBase = 0;
    for (int b = 0; b < bucket; ++b) {
        bucketBase += bucketTotals[b];
    }
    __syncthreads();

    for (int block = 0; block < numBlocksPerBatch; ++block) {
        histogram[block * kRadixHistogramSize + bucket] += bucketBase;
    }
}

/**
 * @brief Scatters elements into the correct output position for the current radix pass.
 * @param input Source buffer (keys for intermediate passes).
 * @param output Destination buffer.
 * @param histogram Exclusive prefix sums per block and bucket.
 * @param iterPass Zero-based radix pass index.
 * @param arrayLength Number of elements per row.
 * @param numBlocksPerBatch Number of blocks per batch dimension.
 */
__global__ void RadixSortScatter(const float* __restrict__ input,
                                float* __restrict__ output,
                                int* __restrict__ histogram,
                                int iterPass,
                                int arrayLength,
                                int numBlocksPerBatch) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int chunkStart = static_cast<int>(blockIdx.x) * kWorkPerBlock;
    const int blockX = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);

    input += batchIdx * arrayLength;
    output += batchIdx * arrayLength;
    histogram += batchIdx * numBlocksPerBatch * kRadixHistogramSize + blockX * kRadixHistogramSize;

    __shared__ int sharedHistogram[kRadixHistogramSize];
    if (thread < kRadixHistogramSize) {
        sharedHistogram[thread] = histogram[thread];
    }
    __syncthreads();

    __shared__ int warpStart[kNumWarpsPerBlock][kRadixHistogramSize];

    const int warpId = static_cast<int>(threadIdx.x) / kWarpSize;
    const int laneId = static_cast<int>(threadIdx.x) % kWarpSize;

    for (int i = 0; i < kNumWarpsPerBlock; ++i) {
        if (thread < kRadixHistogramSize) {
            warpStart[i][thread] = 0;
        }
    }
    __syncthreads();

    const int shiftBits = iterPass * kRadixBits;
    const uint32_t bucketMask = (1U << kRadixBits) - 1;

    for (int i = chunkStart + warpId * kWarpSize * kWorkPerThread + laneId * 4;
         i < chunkStart + (warpId + 1) * kWarpSize * kWorkPerThread;
         i += kWarpSize * 4) {
        if (i >= arrayLength) {
            break;
        }
        const uint4 uval = *reinterpret_cast<const uint4*>(&input[i]);

        const uint32_t b0 = (uval.x >> shiftBits) & bucketMask;
        const uint32_t b1 = (uval.y >> shiftBits) & bucketMask;
        const uint32_t b2 = (uval.z >> shiftBits) & bucketMask;
        const uint32_t b3 = (uval.w >> shiftBits) & bucketMask;

        atomicAdd(&warpStart[warpId][b0], 1);
        atomicAdd(&warpStart[warpId][b1], 1);
        atomicAdd(&warpStart[warpId][b2], 1);
        atomicAdd(&warpStart[warpId][b3], 1);
    }
    __syncthreads();

    if (thread < kRadixHistogramSize) {
        int running = 0;
#pragma unroll
        for (int w = 0; w < kNumWarpsPerBlock; ++w) {
            const int count = warpStart[w][thread];
            warpStart[w][thread] = running;
            running += count;
        }
    }
    __syncthreads();

    for (int i = chunkStart + warpId * kWarpSize * kWorkPerThread + laneId;
         i < chunkStart + (warpId + 1) * kWarpSize * kWorkPerThread;
         i += kWarpSize) {
        if (i >= arrayLength) {
            break;
        }
        const uint32_t uval = *reinterpret_cast<const uint32_t*>(&input[i]);
        const uint32_t bucket = (uval >> shiftBits) & bucketMask;

        const unsigned activeMask = __activemask();
        const unsigned sameBucketMask = __match_any_sync(activeMask, bucket);

        const int localInWarp = __popc(sameBucketMask & ((1u << laneId) - 1));
        const int warpOffset = warpStart[warpId][bucket];
        const int blockBase = sharedHistogram[bucket];
        const int pos = blockBase + warpOffset + localInWarp;

        if (iterPass < kRadixPasses - 1) {
            *reinterpret_cast<uint32_t*>(&output[pos]) = uval;
        } else {
            output[pos] = SortKeyToFloat(uval);
        }

        const int totalInBallot = __popc(sameBucketMask);
        const int leader = __ffs(sameBucketMask) - 1;
        if (laneId == leader) {
            warpStart[warpId][bucket] += totalInBallot;
        }
        __syncwarp();
    }
}

}  // namespace

/**
 * @brief Launches the radix sort pipeline for fp32 arrays.
 * @param input Pointer to device input buffer (modified in-place).
 * @param output Pointer to device output buffer.
 * @param histogram Pointer to device histogram buffer.
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 */
void LaunchRadixSortFp32Kernel(float* input,
                               float* output,
                               int* histogram,
                               int batchSize,
                               int arrayLength) {
    if (batchSize <= 0 || arrayLength <= 0) {
        return;
    }
    if (arrayLength % 4 != 0) {
        return;
    }

    const int numBlocksPerBatch = CeilDiv(arrayLength, kWorkPerBlock);
    const dim3 blockDim(kThreadsPerBlock);
    const dim3 gridDim(numBlocksPerBatch, batchSize);
    const dim3 blockPrefix(kRadixHistogramSize);
    const dim3 gridPrefix(1, batchSize);

    float* src = input;
    float* dst = output;
    for (int iter = 0; iter < kRadixPasses; ++iter) {
        RadixSortBuildHistogram<<<gridDim, blockDim>>>(src,
                                                       histogram,
                                                       iter,
                                                       arrayLength,
                                                       numBlocksPerBatch);
        RadixSortPrefixSum<<<gridPrefix, blockPrefix>>>(histogram, numBlocksPerBatch);
        RadixSortScatter<<<gridDim, blockDim>>>(src,
                                                dst,
                                                histogram,
                                                iter,
                                                arrayLength,
                                                numBlocksPerBatch);
        if (iter != kRadixPasses - 1) {
            float* temp = src;
            src = dst;
            dst = temp;
        }
    }
    if (output != dst) {
        const size_t totalElements = static_cast<size_t>(batchSize) * arrayLength;
        const int copyBlocks = CeilDiv(static_cast<int>(totalElements), kThreadsPerBlock);
        CopyDeviceBuffer<<<copyBlocks, kThreadsPerBlock>>>(dst, output, totalElements);
    }
}
