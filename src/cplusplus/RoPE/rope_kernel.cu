// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for RoPE.
// ==============================================================================
#include "RoPE/rope_kernel.cuh"

#include <cassert>
#include <cuda_runtime.h>

namespace {

using Bf16 = __nv_bfloat16;

/**
 * @brief Returns ceil(numerator / denominator) for positive integers.
 * @param numerator Dividend value.
 * @param denominator Divisor value; must be > 0.
 * @return Rounded-up quotient.
 */
__host__ __device__ inline int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief CUDA kernel for rotary position embedding in bf16.
 * @param input Pointer to device input tensor of shape [batch, seqLen, hiddenDim].
 * @param output Pointer to device output tensor of shape [batch, seqLen, hiddenDim].
 * @param hiddenDim Hidden dimension size (must be even).
 * @param seqLen Sequence length.
 * @param base RoPE base frequency.
 */
__global__ void RopeBf16Kernel(const Bf16* __restrict__ input,
                               Bf16* __restrict__ output,
                               int hiddenDim,
                               int seqLen,
                               float base) {
    const int batchIdx = static_cast<int>(blockIdx.z);
    const int threadsPerToken = CeilDiv(hiddenDim, kRopeValuesPerThread);
    const int tokensPerBlock = kRopeThreadsPerBlock / threadsPerToken;
    const int blockSeqStart = static_cast<int>(blockIdx.y) * tokensPerBlock;

    input += batchIdx * seqLen * hiddenDim + blockSeqStart * hiddenDim;
    output += batchIdx * seqLen * hiddenDim + blockSeqStart * hiddenDim;

    extern __shared__ float sharedInvFreq[];

    const int thread = static_cast<int>(threadIdx.x);
    if (thread < hiddenDim) {
        sharedInvFreq[thread] = powf(base, -2.0f * (thread / 2) / static_cast<float>(hiddenDim));
    }
    __syncthreads();

    const int localTokenIdx = thread / threadsPerToken;
    const int valueStartIdx = (thread % threadsPerToken) * kRopeValuesPerThread;
    const int globalTokenIdx = blockSeqStart + localTokenIdx;

    if (localTokenIdx >= tokensPerBlock || globalTokenIdx >= seqLen || valueStartIdx >= hiddenDim) {
        return;
    }

    for (int dim = 0; dim < kRopeValuesPerThread; dim += 8) {
        Bf16 values[8];
        *reinterpret_cast<int4*>(&values[0]) =
            *reinterpret_cast<const int4*>(&input[localTokenIdx * hiddenDim + valueStartIdx + dim]);

        Bf16 outputValues[8];

        const int hiddenIdx0 = valueStartIdx + dim;
        const int hiddenIdx1 = valueStartIdx + dim + 2;
        const int hiddenIdx2 = valueStartIdx + dim + 4;
        const int hiddenIdx3 = valueStartIdx + dim + 6;

        const float4 angle = {globalTokenIdx * sharedInvFreq[hiddenIdx0],
                              globalTokenIdx * sharedInvFreq[hiddenIdx1],
                              globalTokenIdx * sharedInvFreq[hiddenIdx2],
                              globalTokenIdx * sharedInvFreq[hiddenIdx3]};

        float4 cosValue;
        float4 sinValue;
        __sincosf(angle.x, &sinValue.x, &cosValue.x);
        __sincosf(angle.y, &sinValue.y, &cosValue.y);
        __sincosf(angle.z, &sinValue.z, &cosValue.z);
        __sincosf(angle.w, &sinValue.w, &cosValue.w);

        const float2 pair0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values[0]));
        const float2 pair1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values[2]));
        const float2 pair2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values[4]));
        const float2 pair3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values[6]));

        *reinterpret_cast<__nv_bfloat162*>(&outputValues[0]) =
            __float22bfloat162_rn({pair0.x * cosValue.x - pair0.y * sinValue.x,
                                   pair0.x * sinValue.x + pair0.y * cosValue.x});
        *reinterpret_cast<__nv_bfloat162*>(&outputValues[2]) =
            __float22bfloat162_rn({pair1.x * cosValue.y - pair1.y * sinValue.y,
                                   pair1.x * sinValue.y + pair1.y * cosValue.y});
        *reinterpret_cast<__nv_bfloat162*>(&outputValues[4]) =
            __float22bfloat162_rn({pair2.x * cosValue.z - pair2.y * sinValue.z,
                                   pair2.x * sinValue.z + pair2.y * cosValue.z});
        *reinterpret_cast<__nv_bfloat162*>(&outputValues[6]) =
            __float22bfloat162_rn({pair3.x * cosValue.w - pair3.y * sinValue.w,
                                   pair3.x * sinValue.w + pair3.y * cosValue.w});

        *reinterpret_cast<int4*>(&output[localTokenIdx * hiddenDim + valueStartIdx + dim]) =
            *reinterpret_cast<int4*>(&outputValues[0]);
    }
}

}  // namespace

/**
 * @brief Launches the bf16 rotary position embedding kernel.
 * @param input Pointer to device input tensor of shape [batchSize, seqLen, hiddenDim].
 * @param output Pointer to device output tensor of shape [batchSize, seqLen, hiddenDim].
 * @param batchSize Batch size.
 * @param seqLen Sequence length.
 * @param hiddenDim Hidden dimension size.
 * @param base RoPE base frequency.
 */
void LaunchRopeBf16Kernel(const __nv_bfloat16* input,
                          __nv_bfloat16* output,
                          int batchSize,
                          int seqLen,
                          int hiddenDim,
                          float base) {
    static_assert(kRopeValuesPerThread % 8 == 0, "kRopeValuesPerThread must be multiple of 8");
    if (batchSize <= 0 || seqLen <= 0 || hiddenDim <= 0) {
        return;
    }
    if ((hiddenDim % 2) != 0 || (hiddenDim % kRopeValuesPerThread) != 0) {
        return;
    }

    const int threadsPerToken = CeilDiv(hiddenDim, kRopeValuesPerThread);
    const int tokensPerBlock = kRopeThreadsPerBlock / threadsPerToken;
    if (tokensPerBlock <= 0 || (kRopeThreadsPerBlock % threadsPerToken) != 0) {
        return;
    }
    const int hiddenPerTile = threadsPerToken * kRopeValuesPerThread;

    const dim3 gridDim(CeilDiv(hiddenDim, hiddenPerTile),
                       CeilDiv(seqLen, tokensPerBlock),
                       batchSize);
    const dim3 blockDim(kRopeThreadsPerBlock);
    const size_t sharedMemBytes = static_cast<size_t>(hiddenDim) * sizeof(float);

    RopeBf16Kernel<<<gridDim, blockDim, sharedMemBytes>>>(input, output, hiddenDim, seqLen, base);
}
