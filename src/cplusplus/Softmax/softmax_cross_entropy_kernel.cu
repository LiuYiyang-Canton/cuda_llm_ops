// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for softmax cross-entropy.
// ==============================================================================
#include "Softmax/softmax_cross_entropy_kernel.cuh"

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
 * @brief Kernel for fused softmax + cross-entropy.
 * @param logits Device pointer to input logits.
 * @param labels Device pointer to label indices.
 * @param loss Device pointer to per-row loss, storing sum of exponentials temporarily.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyKernel(const float* __restrict__ logits,
                                          const int* __restrict__ labels,
                                          float* __restrict__ loss,
                                          int batchSize,
                                          int featureSize) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batchSize) {
        return;
    }
    logits += static_cast<size_t>(row) * featureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);

    TileMaxSum maxAndSum;
    maxAndSum.max = -INFINITY;
    maxAndSum.sum = 0.0f;
    for (int col = static_cast<int>(threadIdx.x) * 4; col < featureSize; col += blockDim.x * 4) {
        const float4 localLogits = logits4[col / 4];
        TileMaxSum newMaxAndSum;
        newMaxAndSum.max = fmaxf(fmaxf(localLogits.x, localLogits.y),
                                 fmaxf(localLogits.z, localLogits.w));
        newMaxAndSum.sum = __expf(localLogits.x - newMaxAndSum.max) +
                           __expf(localLogits.y - newMaxAndSum.max) +
                           __expf(localLogits.z - newMaxAndSum.max) +
                           __expf(localLogits.w - newMaxAndSum.max);
        maxAndSum = TileMaxSumReduce()(maxAndSum, newMaxAndSum);
    }
    auto block = cg::tiled_partition<kSoftmaxCrossEntropyThreadsPerBlock>(cg::this_thread_block());
    const TileMaxSum blockResult = cg::reduce(block, maxAndSum, TileMaxSumReduce());

    if (threadIdx.x == 0) {
        const int label = labels[row];
        loss[row] = -logits[label] + __logf(blockResult.sum) + blockResult.max;
    }
}

/**
 * @brief Launches the softmax cross-entropy kernel with configured grid/block sizes.
 * @param logits Device pointer to input logits.
 * @param labels Device pointer to label indices.
 * @param loss Device pointer to per-row loss.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyKernel(const float* logits,
                                     const int* labels,
                                     float* loss,
                                     int batchSize,
                                     int featureSize) {
    const dim3 numThreads(kSoftmaxCrossEntropyThreadsPerBlock);
    const dim3 numBlocks(batchSize);
    SoftmaxCrossEntropyKernel<<<numBlocks, numThreads>>>(logits, labels, loss, batchSize, featureSize);
}
