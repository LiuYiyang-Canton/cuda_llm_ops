// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for softmax backward pass.
// ==============================================================================
#include "Softmax/softmax_backward_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**
 * @brief Computes ceil(numerator / denominator) for positive integers.
 * @param numerator Dividend value.
 * @param denominator Divisor value (must be > 0).
 * @return Rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Computes per-row dot(s, g) and writes s*g into a temporary buffer.
 * @param gradOutput Device pointer to upstream gradient.
 * @param softmaxOutput Device pointer to softmax output.
 * @param gradX Device pointer to s*g buffer.
 * @param dotProduct Device pointer to per-row dot(s, g) accumulator.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxBackwardDotKernel(const float* gradOutput,
                                         const float* softmaxOutput,
                                         float* gradX,
                                         float* dotProduct,
                                         int batchSize,
                                         int featureSize) {
    const int row = static_cast<int>(blockIdx.y);
    if (row >= batchSize) {
        return;
    }

    gradOutput += static_cast<size_t>(row) * featureSize;
    softmaxOutput += static_cast<size_t>(row) * featureSize;
    gradX += static_cast<size_t>(row) * featureSize;
    const int blockStart = static_cast<int>(blockIdx.x) * blockDim.x * kSoftmaxBackwardWorkPerThread;
    int blockEnd = blockStart + blockDim.x * kSoftmaxBackwardWorkPerThread;
    if (blockEnd > featureSize) {
        blockEnd = featureSize;
    }

    float dot = 0.0f;
    const int threadStart = blockStart + static_cast<int>(threadIdx.x) * 4;
    for (int col = threadStart; col < blockEnd; col += blockDim.x * 4) {
        const float4 g = *reinterpret_cast<const float4*>(&gradOutput[col]);
        const float4 s = *reinterpret_cast<const float4*>(&softmaxOutput[col]);
        const float4 sg = {g.x * s.x, g.y * s.y, g.z * s.z, g.w * s.w};
        dot += sg.x + sg.y + sg.z + sg.w;
        *reinterpret_cast<float4*>(&gradX[col]) = sg;
    }

    auto block = cg::tiled_partition<kSoftmaxBackwardThreadsPerBlock>(cg::this_thread_block());
    const float blockSum = cg::reduce(block, dot, cg::plus<float>());
    if (block.thread_rank() == 0) {
        atomicAdd(&dotProduct[row], blockSum);
    }
}

/**
 * @brief Computes final gradient: gradX = s*g - s*dot(s, g).
 * @param softmaxOutput Device pointer to softmax output.
 * @param dotProduct Device pointer to per-row dot(s, g).
 * @param gradX Device pointer to output gradient (in-place over s*g buffer).
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxBackwardGradKernel(const float* softmaxOutput,
                                          const float* dotProduct,
                                          float* gradX,
                                          int batchSize,
                                          int featureSize) {
    const int row = static_cast<int>(blockIdx.y);
    if (row >= batchSize) {
        return;
    }

    softmaxOutput += static_cast<size_t>(row) * featureSize;
    gradX += static_cast<size_t>(row) * featureSize;
    const float dot = dotProduct[row];
    const int blockStart = static_cast<int>(blockIdx.x) * blockDim.x * kSoftmaxBackwardWorkPerThread;
    int blockEnd = blockStart + blockDim.x * kSoftmaxBackwardWorkPerThread;
    if (blockEnd > featureSize) {
        blockEnd = featureSize;
    }

    const int threadStart = blockStart + static_cast<int>(threadIdx.x) * 4;
    for (int col = threadStart; col < blockEnd; col += blockDim.x * 4) {
        const float4 s = *reinterpret_cast<const float4*>(&softmaxOutput[col]);
        const float4 sg = *reinterpret_cast<const float4*>(&gradX[col]);
        const float4 gradXLocal = {sg.x - s.x * dot,
                                   sg.y - s.y * dot,
                                   sg.z - s.z * dot,
                                   sg.w - s.w * dot};

        *reinterpret_cast<float4*>(&gradX[col]) = gradXLocal;
    }
}

/**
 * @brief Launches the softmax backward dot kernel with configured grid/block sizes.
 * @param gradOutput Device pointer to upstream gradient.
 * @param softmaxOutput Device pointer to softmax output.
 * @param gradX Device pointer to s*g buffer.
 * @param dotProduct Device pointer to per-row dot(s, g) accumulator.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxBackwardDotKernel(const float* gradOutput,
                                    const float* softmaxOutput,
                                    float* gradX,
                                    float* dotProduct,
                                    int batchSize,
                                    int featureSize) {
    const int blocksPerRow = CeilDiv(featureSize,
                                    kSoftmaxBackwardThreadsPerBlock * kSoftmaxBackwardWorkPerThread);
    const dim3 gridDim(blocksPerRow, batchSize);
    const dim3 blockDim(kSoftmaxBackwardThreadsPerBlock);
    SoftmaxBackwardDotKernel<<<gridDim, blockDim>>>(gradOutput, softmaxOutput, gradX, dotProduct,
                                                    batchSize, featureSize);
}

/**
 * @brief Launches the softmax backward gradient kernel with configured grid/block sizes.
 * @param softmaxOutput Device pointer to softmax output.
 * @param dotProduct Device pointer to per-row dot(s, g).
 * @param gradX Device pointer to output gradient (in-place over s*g buffer).
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxBackwardGradKernel(const float* softmaxOutput,
                                     const float* dotProduct,
                                     float* gradX,
                                     int batchSize,
                                     int featureSize) {
    const int blocksPerRow = CeilDiv(featureSize,
                                    kSoftmaxBackwardThreadsPerBlock * kSoftmaxBackwardWorkPerThread);
    const dim3 gridDim(blocksPerRow, batchSize);
    const dim3 blockDim(kSoftmaxBackwardThreadsPerBlock);
    SoftmaxBackwardGradKernel<<<gridDim, blockDim>>>(softmaxOutput, dotProduct, gradX,
                                                     batchSize, featureSize);
}
