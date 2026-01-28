// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for LayerNorm.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr float kLayerNormEpsilon = 1.0e-6f;
constexpr int kLayerNormThreadsPerBlock = 256;
constexpr int kLayerNormWarpSize = 32;
constexpr int kLayerNormBlockNumWarps = kLayerNormThreadsPerBlock / kLayerNormWarpSize;

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
                                    int hiddenSize);

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
                               int hiddenSize);
