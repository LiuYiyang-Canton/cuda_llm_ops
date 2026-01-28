// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for RMSNorm.
// ==============================================================================
#include "RMSNorm/rmsnorm_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**
 * @brief Computes RMSNorm forward for a batch, one hidden-dimension row per block.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param rmsnormOut Output tensor with shape (seqLength, hiddenSize).
 * @param hiddenSize Hidden dimension size.
 */
__global__ void RMSNormFp32Kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  float* __restrict__ rmsnormOut,
                                  int hiddenSize) {
    auto warp = cg::tiled_partition<kRmsNormWarpSize>(cg::this_thread_block());
    const int threadId = static_cast<int>(threadIdx.x);
    const int laneId = threadId % kRmsNormWarpSize;
    __shared__ float warpSum[kRmsNormBlockNumWarps];

    const int rowIndex = static_cast<int>(blockIdx.x);
    const int rowOffset = rowIndex * hiddenSize;
    x += rowOffset;

    float sumSq = 0.0f;

#pragma unroll
    for (int i = threadId * 4; i < hiddenSize; i += blockDim.x * 4) {
        if (i + 3 < hiddenSize) {
            const float4 temp = *(reinterpret_cast<const float4*>(&x[i]));
            sumSq += temp.x * temp.x + temp.y * temp.y + temp.z * temp.z + temp.w * temp.w;
        } else {
            for (int j = i; j < hiddenSize; ++j) {
                const float val = x[j];
                sumSq += val * val;
            }
        }
    }

    sumSq = cg::reduce(warp, sumSq, cg::plus<float>());
    if (laneId == 0) {
        warpSum[threadId / kRmsNormWarpSize] = sumSq;
    }
    __syncthreads();

    for (int offset = 0; offset < kRmsNormBlockNumWarps; ++offset) {
        if (threadId < offset) {
            warpSum[threadId] += warpSum[offset];
        }
        __syncthreads();
    }

    const float invRms = rsqrtf(warpSum[0] / static_cast<float>(hiddenSize) + kRmsNormEpsilon);

#pragma unroll
    for (int i = threadId * 4; i < hiddenSize; i += blockDim.x * 4) {
        if (i + 3 < hiddenSize) {
            float4 val = *(reinterpret_cast<const float4*>(&x[i]));
            const float4 w = *(reinterpret_cast<const float4*>(&weight[i]));
            val.x = val.x * invRms * w.x;
            val.y = val.y * invRms * w.y;
            val.z = val.z * invRms * w.z;
            val.w = val.w * invRms * w.w;
            *(reinterpret_cast<float4*>(&rmsnormOut[rowOffset + i])) = val;
        } else {
            for (int j = i; j < hiddenSize; ++j) {
                rmsnormOut[rowOffset + j] = x[j] * invRms * weight[j];
            }
        }
    }
}

/**
 * @brief Launches the RMSNorm forward kernel with configured grid/block sizes.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param rmsnormOut Output tensor with shape (seqLength, hiddenSize).
 * @param seqLength Number of rows.
 * @param hiddenSize Hidden dimension size.
 */
void LaunchRmsNormFp32Kernel(const float* x,
                             const float* weight,
                             float* rmsnormOut,
                             int seqLength,
                             int hiddenSize) {
    if (seqLength <= 0 || hiddenSize <= 0) {
        return;
    }
    if ((hiddenSize % 4) != 0) {
        return;
    }
    const dim3 numThreads(kRmsNormThreadsPerBlock);
    const dim3 numBlocks(seqLength);
    RMSNormFp32Kernel<<<numBlocks, numThreads>>>(x, weight, rmsnormOut, hiddenSize);
}
