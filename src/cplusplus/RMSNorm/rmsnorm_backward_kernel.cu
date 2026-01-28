// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for RMSNorm backward pass.
// ==============================================================================
#include "RMSNorm/rmsnorm_backward_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**
 * @brief CUDA kernel that computes RMSNorm backward for a batched sequence tensor, one hiddenDim vector per block.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from RMSNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma RMSNorm scale parameter, shape (hiddenDim).
 * @param invRms Saved inverse RMS values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size).
 */
__global__ void RMSNormBackwardKernel(float* __restrict__ gradX,
                                      float* __restrict__ gradGamma,
                                      const float* __restrict__ x,
                                      const float* __restrict__ gradOutput,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ invRms,
                                      int batchSize,
                                      int seqLength,
                                      int hiddenDim) {
    const int tokenIndex = static_cast<int>(blockIdx.x);
    const int tokenCount = batchSize * seqLength;
    if (tokenIndex >= tokenCount) {
        return;
    }

    gradX += static_cast<size_t>(tokenIndex) * hiddenDim;
    gradGamma += static_cast<size_t>(tokenIndex) * hiddenDim;
    x += static_cast<size_t>(tokenIndex) * hiddenDim;
    gradOutput += static_cast<size_t>(tokenIndex) * hiddenDim;
    invRms += tokenIndex;

    const float invRmsValue = *invRms;

    float dotProduct = 0.0f;
    for (int i = static_cast<int>(threadIdx.x) * 4; i < hiddenDim; i += 4 * kRmsNormBackwardThreadsPerBlock) {
        const float4 gradOutputVal = *(reinterpret_cast<const float4*>(&gradOutput[i]));
        const float4 gammaVal = *(reinterpret_cast<const float4*>(&gamma[i]));
        const float4 xVal = *(reinterpret_cast<const float4*>(&x[i]));
        dotProduct += gradOutputVal.x * gammaVal.x * xVal.x +
                      gradOutputVal.y * gammaVal.y * xVal.y +
                      gradOutputVal.z * gammaVal.z * xVal.z +
                      gradOutputVal.w * gammaVal.w * xVal.w;
    }

    auto block = cg::tiled_partition<kRmsNormBackwardThreadsPerBlock>(cg::this_thread_block());
    dotProduct = cg::reduce(block, dotProduct, cg::plus<float>());

    const float scale = dotProduct * invRmsValue * invRmsValue / static_cast<float>(hiddenDim);
    for (int i = static_cast<int>(threadIdx.x) * 4; i < hiddenDim; i += 4 * kRmsNormBackwardThreadsPerBlock) {
        const float4 xVal = *(reinterpret_cast<const float4*>(&x[i]));
        const float4 gradOutputVal = *(reinterpret_cast<const float4*>(&gradOutput[i]));
        const float4 gammaVal = *(reinterpret_cast<const float4*>(&gamma[i]));

        float4 gradXVal;
        gradXVal.x = (gradOutputVal.x * gammaVal.x - xVal.x * scale) * invRmsValue;
        gradXVal.y = (gradOutputVal.y * gammaVal.y - xVal.y * scale) * invRmsValue;
        gradXVal.z = (gradOutputVal.z * gammaVal.z - xVal.z * scale) * invRmsValue;
        gradXVal.w = (gradOutputVal.w * gammaVal.w - xVal.w * scale) * invRmsValue;

        *(reinterpret_cast<float4*>(&gradX[i])) = gradXVal;

        float4 gradGammaVal;
        gradGammaVal.x = gradOutputVal.x * xVal.x * invRmsValue;
        gradGammaVal.y = gradOutputVal.y * xVal.y * invRmsValue;
        gradGammaVal.z = gradOutputVal.z * xVal.z * invRmsValue;
        gradGammaVal.w = gradOutputVal.w * xVal.w * invRmsValue;
        *(reinterpret_cast<float4*>(&gradGamma[i])) = gradGammaVal;
    }
}

/**
 * @brief Launches the RMSNorm backward kernel with configured grid/block sizes.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from RMSNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma RMSNorm scale parameter, shape (hiddenDim).
 * @param invRms Saved inverse RMS values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size).
 */
void LaunchRmsNormBackwardKernel(float* gradX,
                                 float* gradGamma,
                                 const float* x,
                                 const float* gradOutput,
                                 const float* gamma,
                                 const float* invRms,
                                 int batchSize,
                                 int seqLength,
                                 int hiddenDim) {
    const int tokenCount = batchSize * seqLength;
    const dim3 threads(kRmsNormBackwardThreadsPerBlock);
    const dim3 blocks(static_cast<unsigned int>(tokenCount));
    RMSNormBackwardKernel<<<blocks, threads>>>(gradX,
                                               gradGamma,
                                               x,
                                               gradOutput,
                                               gamma,
                                               invRms,
                                               batchSize,
                                               seqLength,
                                               hiddenDim);
}
