// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for BF16 GLU.
// ==============================================================================
#pragma once

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using Bf16 = __nv_bfloat16;

namespace glu_config {
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kBlockNumWarpsM = 8;
constexpr int kBlockNumWarpsN = 2;
constexpr int kBlockNumWarps = kBlockNumWarpsM * kBlockNumWarpsN;
constexpr int kWarpSize = 32;
constexpr int kFragmentSize = 16;
constexpr int kWarpM = kBlockM / kBlockNumWarpsM;
constexpr int kWarpN = kBlockN / kBlockNumWarpsN;
constexpr int kWarpNumFragmentsM = kWarpM / kFragmentSize;
constexpr int kWarpNumFragmentsN = kWarpN / kFragmentSize;
constexpr int kBlockNumThreads = kBlockNumWarpsM * kBlockNumWarpsN * kWarpSize;
constexpr int kKFragmentsPerIter = 8;
constexpr int kATileM = kBlockM;
constexpr int kATileK = kFragmentSize * kKFragmentsPerIter;
constexpr int kBTileN = kBlockN;
constexpr int kBTileK = kFragmentSize * kKFragmentsPerIter;
constexpr int kNumElementsPerLoad = sizeof(int4) / sizeof(Bf16);
constexpr int kATileNumThreadsPerRow = kATileK / kNumElementsPerLoad;
constexpr int kATileNumRowsPerWarp = kATileM / kBlockNumWarps;
constexpr int kATileNumRowsPerWarpPerIter = kWarpSize / kATileNumThreadsPerRow;
constexpr int kATileNumLoadIters = kATileNumRowsPerWarp / kATileNumRowsPerWarpPerIter;
constexpr int kBTileNumThreadsPerCol = kBTileK / kNumElementsPerLoad;
constexpr int kBTileNumColsPerWarp = kBTileN / kBlockNumWarps;
constexpr int kBTileNumColsPerWarpPerIter = kWarpSize / kBTileNumThreadsPerCol;
constexpr int kBTileNumLoadIters = kBTileNumColsPerWarp / kBTileNumColsPerWarpPerIter;
constexpr int kSharedMemPadding = 8;
constexpr int kSharedMemStride = kATileK + kSharedMemPadding;
constexpr int kSharedMemCStride = kBlockN * 2 + kSharedMemPadding;
constexpr int kS2GNumElementsPerLoad = sizeof(float4) / sizeof(float);
constexpr int kCTileNumThreadsPerRow = kBlockN / kS2GNumElementsPerLoad;
constexpr int kCTileNumRowsPerWarp = kBlockM / kBlockNumWarps;
constexpr int kCTileNumRowsPerWarpPerIter = kWarpSize / kCTileNumThreadsPerRow;
constexpr int kCTileNumWriteIters = kCTileNumRowsPerWarp / kCTileNumRowsPerWarpPerIter;
constexpr int kGluType = 0;
}  // namespace glu_config

constexpr int BLOCK_M = glu_config::kBlockM;
constexpr int BLOCK_N = glu_config::kBlockN;
constexpr int A_TILE_K = glu_config::kATileK;
constexpr int B_TILE_K = glu_config::kBTileK;

/**
 * @brief Computes the sigmoid activation.
 * @param x Input value.
 * @return Sigmoid activation of x.
 */
__host__ __device__ inline float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Computes the Gaussian error linear unit activation.
 * @param x Input value.
 * @return GeLU activation of x.
 */
__host__ __device__ inline float GaussianError(float x) {
    return 0.5f * (1.0f + erf(x * rsqrtf(2.0f)));
}

/**
 * @brief Computes the swish activation.
 * @param x Input value.
 * @return Swish activation of x.
 */
__host__ __device__ inline float Swish(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * @brief Applies the configured GLU activation.
 * @param x Input value.
 * @return Activated value.
 */
__host__ __device__ inline float Activation(float x) {
    if constexpr (glu_config::kGluType == 0) {
        return Sigmoid(x);
    }
    if constexpr (glu_config::kGluType == 1) {
        return Swish(x);
    }
    if constexpr (glu_config::kGluType == 2) {
        return GaussianError(x);
    }
    return x;
}

/**
 * @brief BF16 GLU kernel with fp32 accumulation.
 * @param x Device pointer to input matrix X.
 * @param weight Device pointer to weight matrix.
 * @param result Device pointer to output matrix.
 * @param k Size of the K dimension.
 * @param intermediateSize Size of the intermediate/output dimension.
 */
__global__ void GluBf16Kernel(const Bf16* __restrict__ x,
                              const Bf16* __restrict__ weight,
                              float* __restrict__ result,
                              int k,
                              int intermediateSize);

/**
 * @brief Launches the GLU BF16 kernel with configured grid/block sizes.
 * @param x Device pointer to input matrix X.
 * @param weight Device pointer to weight matrix.
 * @param result Device pointer to output matrix.
 * @param batchSize Number of rows in X.
 * @param k Size of the K dimension.
 * @param intermediateSize Size of the intermediate/output dimension.
 */
void LaunchGluBf16Kernel(const Bf16* x,
                         const Bf16* weight,
                         float* result,
                         int batchSize,
                         int k,
                         int intermediateSize);
