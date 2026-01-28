// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for LayerNorm backward pass.
// ==============================================================================
#include "LayerNorm/layernorm_backward_kernel.cuh"

/**
 * @brief Reduces a pair of values within a warp using shuffle operations.
 * @param valueA First value to reduce within the warp.
 * @param valueB Second value to reduce within the warp.
 */
__device__ __forceinline__ void WarpReduceSumPair(float& valueA, float& valueB) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        valueA += __shfl_down_sync(0xffffffff, valueA, offset);
        valueB += __shfl_down_sync(0xffffffff, valueB, offset);
    }
}

/**
 * @brief Reduces a pair of values across a block using warp shuffles and shared memory.
 * @param valueA First value to reduce across the block (in/out).
 * @param valueB Second value to reduce across the block (in/out).
 */
__device__ __forceinline__ void BlockReduceSumPair(float& valueA, float& valueB) {
    __shared__ float2 warpSums[32];
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int warpId = static_cast<int>(threadIdx.x) >> 5;
    const int warpCount = (blockDim.x + warpSize - 1) / warpSize;

    WarpReduceSumPair(valueA, valueB);
    if (lane == 0) {
        warpSums[warpId] = make_float2(valueA, valueB);
    }
    __syncthreads();

    if (warpId == 0) {
        const float2 warpValue = (lane < warpCount) ? warpSums[lane] : make_float2(0.0f, 0.0f);
        float reducedA = warpValue.x;
        float reducedB = warpValue.y;
        WarpReduceSumPair(reducedA, reducedB);
        if (lane == 0) {
            warpSums[0] = make_float2(reducedA, reducedB);
        }
    }
    __syncthreads();

    valueA = warpSums[0].x;
    valueB = warpSums[0].y;
}

/**
 * @brief CUDA kernel that computes LayerNorm backward, where each thread block handles one token.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from LayerNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma LayerNorm scale parameter, shape (hiddenDim).
 * @param invStdDev Saved inverse standard deviation values from forward pass, shape (batchSize, seqLength).
 * @param mean Saved mean values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size), must be divisible by 4.
 */
__global__ void LayerNormBackwardKernel(float* __restrict__ gradX,
                                        float* __restrict__ gradGamma,
                                        const float* __restrict__ x,
                                        const float* __restrict__ gradOutput,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ invStdDev,
                                        const float* __restrict__ mean,
                                        int batchSize,
                                        int seqLength,
                                        int hiddenDim) {
    const int tokenIndex = static_cast<int>(blockIdx.x);
    if (tokenIndex >= batchSize * seqLength) {
        return;
    }
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);

    gradX += static_cast<size_t>(tokenIndex) * hiddenDim;
    gradGamma += static_cast<size_t>(tokenIndex) * hiddenDim;
    x += static_cast<size_t>(tokenIndex) * hiddenDim;
    gradOutput += static_cast<size_t>(tokenIndex) * hiddenDim;
    const float invStdDevValue = invStdDev[tokenIndex];
    const float meanValue = mean[tokenIndex];

    extern __shared__ float sharedData[];
    float* xHatShared = sharedData;
    float* dHatShared = sharedData + hiddenDim;

    float sumDHatX = 0.0f;
    float dotProduct = 0.0f;

    for (int i = static_cast<int>(threadIdx.x) * 4; i < hiddenDim; i += blockDim.x * 4) {
        const float4 xVal = *(reinterpret_cast<const float4*>(&x[i]));
        const float4 gradOutputVal = *(reinterpret_cast<const float4*>(&gradOutput[i]));
        const float4 gammaVal = *(reinterpret_cast<const float4*>(&gamma[i]));

        float4 xHatVal;
        xHatVal.x = (xVal.x - meanValue) * invStdDevValue;
        xHatVal.y = (xVal.y - meanValue) * invStdDevValue;
        xHatVal.z = (xVal.z - meanValue) * invStdDevValue;
        xHatVal.w = (xVal.w - meanValue) * invStdDevValue;

        float4 dHatVal;
        dHatVal.x = gradOutputVal.x * gammaVal.x;
        dHatVal.y = gradOutputVal.y * gammaVal.y;
        dHatVal.z = gradOutputVal.z * gammaVal.z;
        dHatVal.w = gradOutputVal.w * gammaVal.w;

        float4 gradGammaVal;
        gradGammaVal.x = gradOutputVal.x * xHatVal.x;
        gradGammaVal.y = gradOutputVal.y * xHatVal.y;
        gradGammaVal.z = gradOutputVal.z * xHatVal.z;
        gradGammaVal.w = gradOutputVal.w * xHatVal.w;
        *(reinterpret_cast<float4*>(&gradGamma[i])) = gradGammaVal;

        *(reinterpret_cast<float4*>(&xHatShared[i])) = xHatVal;
        *(reinterpret_cast<float4*>(&dHatShared[i])) = dHatVal;

        sumDHatX += dHatVal.x + dHatVal.y + dHatVal.z + dHatVal.w;
        dotProduct += dHatVal.x * xHatVal.x + dHatVal.y * xHatVal.y +
                      dHatVal.z * xHatVal.z + dHatVal.w * xHatVal.w;
    }

    BlockReduceSumPair(sumDHatX, dotProduct);
    const float meanDhat = sumDHatX * invHiddenDim;
    const float meanDhatXhat = dotProduct * invHiddenDim;

    for (int i = static_cast<int>(threadIdx.x) * 4; i < hiddenDim; i += blockDim.x * 4) {
        const float4 xHatVal = *(reinterpret_cast<const float4*>(&xHatShared[i]));
        const float4 dHatVal = *(reinterpret_cast<const float4*>(&dHatShared[i]));

        float4 gradXVal;
        gradXVal.x = (dHatVal.x - meanDhat - xHatVal.x * meanDhatXhat) * invStdDevValue;
        gradXVal.y = (dHatVal.y - meanDhat - xHatVal.y * meanDhatXhat) * invStdDevValue;
        gradXVal.z = (dHatVal.z - meanDhat - xHatVal.z * meanDhatXhat) * invStdDevValue;
        gradXVal.w = (dHatVal.w - meanDhat - xHatVal.w * meanDhatXhat) * invStdDevValue;
        *(reinterpret_cast<float4*>(&gradX[i])) = gradXVal;
    }
}

/**
 * @brief Launches the LayerNorm backward kernel with configured grid/block sizes.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from LayerNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma LayerNorm scale parameter, shape (hiddenDim).
 * @param invStdDev Saved inverse standard deviation values from forward pass, shape (batchSize, seqLength).
 * @param mean Saved mean values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size), must be divisible by 4.
 */
void LaunchLayerNormBackwardKernel(float* gradX,
                                   float* gradGamma,
                                   const float* x,
                                   const float* gradOutput,
                                   const float* gamma,
                                   const float* invStdDev,
                                   const float* mean,
                                   int batchSize,
                                   int seqLength,
                                   int hiddenDim) {
    const int tokenCount = batchSize * seqLength;
    const dim3 grid(static_cast<unsigned int>(tokenCount));
    const dim3 block(kLayerNormBackwardThreadsPerBlock);
    const size_t sharedMemBytes = static_cast<size_t>(hiddenDim) * 2 * sizeof(float);
    LayerNormBackwardKernel<<<grid, block, sharedMemBytes>>>(gradX,
                                                             gradGamma,
                                                             x,
                                                             gradOutput,
                                                             gamma,
                                                             invStdDev,
                                                             mean,
                                                             batchSize,
                                                             seqLength,
                                                             hiddenDim);
}
