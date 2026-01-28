// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for LayerNorm.
// ==============================================================================
#include "LayerNorm/layernorm_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**
 * @brief Computes LayerNorm forward for a batch, one hidden-dimension row per block.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param layernormOut Output tensor with shape (seqLength, hiddenSize).
 * @param hiddenSize Hidden dimension size.
 */
__global__ void LayerNormFp32Kernel(const float* __restrict__ x,
                                    const float* __restrict__ weight,
                                    float* __restrict__ layernormOut,
                                    int hiddenSize) {
    auto warp = cg::tiled_partition<kLayerNormWarpSize>(cg::this_thread_block());
    const int threadId = static_cast<int>(threadIdx.x);
    const int laneId = threadId % kLayerNormWarpSize;
    __shared__ float2 warpSum[kLayerNormBlockNumWarps];

    const int rowIndex = static_cast<int>(blockIdx.x);
    const int rowOffset = rowIndex * hiddenSize;
    x += rowOffset;

    float sumSq = 0.0f;
    float sum = 0.0f;

#pragma unroll
    for (int i = threadId * 4; i < hiddenSize; i += blockDim.x * 4) {
        if (i + 3 < hiddenSize) {
            const float4 temp = *(reinterpret_cast<const float4*>(&x[i]));
            sumSq += temp.x * temp.x + temp.y * temp.y + temp.z * temp.z + temp.w * temp.w;
            sum += temp.x + temp.y + temp.z + temp.w;
        } else {
            for (int j = i; j < hiddenSize; ++j) {
                const float val = x[j];
                sumSq += val * val;
                sum += val;
            }
        }
    }

    sumSq = cg::reduce(warp, sumSq, cg::plus<float>());
    sum = cg::reduce(warp, sum, cg::plus<float>());
    if (laneId == 0) {
        warpSum[threadId / kLayerNormWarpSize].x = sumSq;
        warpSum[threadId / kLayerNormWarpSize].y = sum;
    }
    __syncthreads();

    for (int offset = 0; offset < kLayerNormBlockNumWarps; ++offset) {
        if (threadId < offset) {
            warpSum[threadId].x += warpSum[offset].x;
            warpSum[threadId].y += warpSum[offset].y;
        }
        __syncthreads();
    }

    const float avg = warpSum[0].y / static_cast<float>(hiddenSize);
    const float invRms = rsqrtf(warpSum[0].x / static_cast<float>(hiddenSize) -
                                avg * avg + kLayerNormEpsilon);

#pragma unroll
    for (int i = threadId * 4; i < hiddenSize; i += blockDim.x * 4) {
        if (i + 3 < hiddenSize) {
            float4 val = *(reinterpret_cast<const float4*>(&x[i]));
            const float4 w = *(reinterpret_cast<const float4*>(&weight[i]));
            val.x = (val.x - avg) * invRms * w.x;
            val.y = (val.y - avg) * invRms * w.y;
            val.z = (val.z - avg) * invRms * w.z;
            val.w = (val.w - avg) * invRms * w.w;
            *(reinterpret_cast<float4*>(&layernormOut[rowOffset + i])) = val;
        } else {
            for (int j = i; j < hiddenSize; ++j) {
                layernormOut[rowOffset + j] = (x[j] - avg) * invRms * weight[j];
            }
        }
    }
}

/**
 * @brief Launches the LayerNorm forward kernel with configured grid/block sizes.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param layernormOut Output tensor with shape (seqLength, hiddenSize).
 * @param seqLength Number of rows.
 * @param hiddenSize Hidden dimension size.
 */
void LaunchLayerNormFp32Kernel(const float* x,
                               const float* weight,
                               float* layernormOut,
                               int seqLength,
                               int hiddenSize) {
    if (seqLength <= 0 || hiddenSize <= 0) {
        return;
    }
    const dim3 numThreads(kLayerNormThreadsPerBlock);
    const dim3 numBlocks(seqLength);
    LayerNormFp32Kernel<<<numBlocks, numThreads>>>(x, weight, layernormOut, hiddenSize);
}
