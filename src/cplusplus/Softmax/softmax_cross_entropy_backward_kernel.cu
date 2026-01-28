// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for softmax cross-entropy backward pass.
// ==============================================================================
#include "Softmax/softmax_cross_entropy_backward_kernel.cuh"

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
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel(const float* __restrict__ logits,
                                                                  float* __restrict__ rowMax,
                                                                  float* __restrict__ rowSum,
                                                                  int batchSize,
                                                                  int featureSize) {
    const int warpId = static_cast<int>(threadIdx.x) / kSoftmaxCrossEntropyBackwardWarpSize;
    const int batchIdx = static_cast<int>(blockIdx.x);
    const int threadId = static_cast<int>(threadIdx.x);
    if (batchIdx >= batchSize) {
        return;
    }
    logits += static_cast<size_t>(batchIdx) * featureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);
    auto warp = cg::tiled_partition<kSoftmaxCrossEntropyBackwardWarpSize>(cg::this_thread_block());
    TileMaxSum maxAndSum;
    maxAndSum.max = -INFINITY;
    maxAndSum.sum = 0.0f;
    __shared__ TileMaxSum warpMaxSum[kSoftmaxCrossEntropyBackwardWarpCount];

    for (int col = threadId * 4; col < featureSize; col += blockDim.x * 4) {
        const float4 val = logits4[col / 4];
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

    for (int offset = kSoftmaxCrossEntropyBackwardWarpCount / 2; offset > 0; offset /= 2) {
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
 * @brief Kernel that computes softmax-cross-entropy gradient for each row.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param labels Device pointer to label indices.
 * @param gradLogits Device pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyBackwardFp32Kernel(const float* __restrict__ logits,
                                                      const float* __restrict__ rowMax,
                                                      const float* __restrict__ rowSum,
                                                      const int* __restrict__ labels,
                                                      float* __restrict__ gradLogits,
                                                      int batchSize,
                                                      int featureSize) {
    const int batchIdx = static_cast<int>(blockIdx.x);
    const int threadId = static_cast<int>(threadIdx.x);
    if (batchIdx >= batchSize) {
        return;
    }
    logits += static_cast<size_t>(batchIdx) * featureSize;
    gradLogits += static_cast<size_t>(batchIdx) * featureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);
    float4* grad4 = reinterpret_cast<float4*>(gradLogits);
    const float maxVal = rowMax[batchIdx];
    const float sumVal = rowSum[batchIdx];
    const int label = labels[batchIdx];

    for (int col = threadId * 4; col < featureSize; col += blockDim.x * 4) {
        const float4 val = logits4[col / 4];
        float4 result;
        result.x = __expf(val.x - maxVal) / sumVal;
        result.y = __expf(val.y - maxVal) / sumVal;
        result.z = __expf(val.z - maxVal) / sumVal;
        result.w = __expf(val.w - maxVal) / sumVal;
        const int base = col;
        if (label >= base && label < base + 4) {
            const int offset = label - base;
            if (offset == 0) {
                result.x -= 1.0f;
            } else if (offset == 1) {
                result.y -= 1.0f;
            } else if (offset == 2) {
                result.z -= 1.0f;
            } else {
                result.w -= 1.0f;
            }
        }
        grad4[col / 4] = result;
    }
}

/**
 * @brief Launches the row max/sum kernel for softmax cross-entropy backward.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel(const float* logits,
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
    const dim3 numThreads(kSoftmaxCrossEntropyBackwardThreadsPerBlock);
    const dim3 numBlocks(batchSize);
    SoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel<<<numBlocks, numThreads>>>(
        logits, rowMax, rowSum, batchSize, featureSize);
}

/**
 * @brief Launches the softmax cross-entropy backward kernel with configured grid/block sizes.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param labels Device pointer to label indices.
 * @param gradLogits Device pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyBackwardFp32Kernel(const float* logits,
                                                 const float* rowMax,
                                                 const float* rowSum,
                                                 const int* labels,
                                                 float* gradLogits,
                                                 int batchSize,
                                                 int featureSize) {
    if (batchSize <= 0 || featureSize <= 0) {
        return;
    }
    if (featureSize % 4 != 0) {
        return;
    }
    const dim3 numThreads(kSoftmaxCrossEntropyBackwardThreadsPerBlock);
    const dim3 numBlocks(batchSize);
    SoftmaxCrossEntropyBackwardFp32Kernel<<<numBlocks, numThreads>>>(
        logits, rowMax, rowSum, labels, gradLogits, batchSize, featureSize);
}
