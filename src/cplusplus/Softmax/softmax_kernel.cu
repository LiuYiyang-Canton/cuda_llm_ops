// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for softmax.
// ==============================================================================
#include "Softmax/softmax_kernel.cuh"

#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace {

struct TileMaxSum {
    float max;
    float sum;
};

struct TileMaxSumReduce {
    /**
     * @brief Combines two TileMaxSum values into a single max/sum pair.
     * @param a First value to reduce.
     * @param b Second value to reduce.
     * @return Reduced max/sum pair.
     */
    __device__ TileMaxSum operator()(const TileMaxSum& a, const TileMaxSum& b) const {
        TileMaxSum result;
        result.max = fmaxf(a.max, b.max);
        result.sum = a.sum * __expf(a.max - result.max) + b.sum * __expf(b.max - result.max);
        return result;
    }
};

}  // namespace

/**
 * @brief Kernel that computes per-row max and sum for softmax.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxRowMaxRowSumFp32Kernel(const float* __restrict__ x,
                                              float* __restrict__ rowMax,
                                              float* __restrict__ rowSum,
                                              int batchSize,
                                              int featureSize) {
    const int warpId = static_cast<int>(threadIdx.x) / kSoftmaxWarpSize;
    const int batchIdx = static_cast<int>(blockIdx.x);
    const int threadId = static_cast<int>(threadIdx.x);
    if (batchIdx >= batchSize) {
        return;
    }
    x += static_cast<size_t>(batchIdx) * featureSize;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    auto warp = cg::tiled_partition<kSoftmaxWarpSize>(cg::this_thread_block());
    TileMaxSum maxAndSum;
    maxAndSum.max = -INFINITY;
    maxAndSum.sum = 0.0f;
    __shared__ TileMaxSum warpMaxSum[kSoftmaxWarpCount];

    for (int col = threadId * 4; col < featureSize; col += blockDim.x * 4) {
        const float4 val = x4[col / 4];
        TileMaxSum newMaxAndSum;
        newMaxAndSum.max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
        newMaxAndSum.sum = __expf(val.x - newMaxAndSum.max) +
                           __expf(val.y - newMaxAndSum.max) +
                           __expf(val.z - newMaxAndSum.max) +
                           __expf(val.w - newMaxAndSum.max);
        maxAndSum = TileMaxSumReduce()(maxAndSum, newMaxAndSum);
    }

    const TileMaxSum warpResult = cg::reduce(warp, maxAndSum, TileMaxSumReduce());

    if (warp.thread_rank() == 0) {
        warpMaxSum[warpId] = warpResult;
    }
    __syncthreads();

    for (int offset = kSoftmaxWarpCount / 2; offset > 0; offset /= 2) {
        if (threadId < offset) {
            warpMaxSum[threadId] = TileMaxSumReduce()(warpMaxSum[threadId], warpMaxSum[threadId + offset]);
        }
        __syncthreads();
    }

    if (threadId == 0) {
        rowMax[batchIdx] = warpMaxSum[0].max;
        rowSum[batchIdx] = warpMaxSum[0].sum;
    }
}

/**
 * @brief Kernel that computes softmax output using precomputed row max and sum.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param softmaxOut Device pointer to output.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxFp32Kernel(const float* __restrict__ x,
                                  const float* __restrict__ rowMax,
                                  const float* __restrict__ rowSum,
                                  float* __restrict__ softmaxOut,
                                  int batchSize,
                                  int featureSize) {
    const int batchIdx = static_cast<int>(blockIdx.x);
    const int threadId = static_cast<int>(threadIdx.x);
    if (batchIdx >= batchSize) {
        return;
    }
    x += static_cast<size_t>(batchIdx) * featureSize;
    softmaxOut += static_cast<size_t>(batchIdx) * featureSize;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* out4 = reinterpret_cast<float4*>(softmaxOut);
    const float maxVal = rowMax[batchIdx];
    const float sumVal = rowSum[batchIdx];

    for (int col = threadId * 4; col < featureSize; col += blockDim.x * 4) {
        const float4 val = x4[col / 4];
        float4 result;
        result.x = __expf(val.x - maxVal) / sumVal;
        result.y = __expf(val.y - maxVal) / sumVal;
        result.z = __expf(val.z - maxVal) / sumVal;
        result.w = __expf(val.w - maxVal) / sumVal;
        out4[col / 4] = result;
    }
}

/**
 * @brief Launches the softmax row max/sum kernel with configured grid/block sizes.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxRowMaxRowSumFp32Kernel(const float* x,
                                         float* rowMax,
                                         float* rowSum,
                                         int batchSize,
                                         int featureSize) {
    if (batchSize <= 0 || featureSize <= 0) {
        return;
    }
    if (featureSize % 4 != 0) {
        return;
    }
    const dim3 numThreads(kSoftmaxThreadsPerBlock);
    const dim3 numBlocks(batchSize);
    SoftmaxRowMaxRowSumFp32Kernel<<<numBlocks, numThreads>>>(x,
                                                             rowMax,
                                                             rowSum,
                                                             batchSize,
                                                             featureSize);
}

/**
 * @brief Launches the softmax kernel with configured grid/block sizes.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param softmaxOut Device pointer to output.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxFp32Kernel(const float* x,
                             const float* rowMax,
                             const float* rowSum,
                             float* softmaxOut,
                             int batchSize,
                             int featureSize) {
    if (batchSize <= 0 || featureSize <= 0) {
        return;
    }
    if (featureSize % 4 != 0) {
        return;
    }
    const dim3 numThreads(kSoftmaxThreadsPerBlock);
    const dim3 numBlocks(batchSize);
    SoftmaxFp32Kernel<<<numBlocks, numThreads>>>(x,
                                                 rowMax,
                                                 rowSum,
                                                 softmaxOut,
                                                 batchSize,
                                                 featureSize);
}
