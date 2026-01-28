// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for RMSNorm.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr float kRmsNormEpsilon = 1.0e-6f;
constexpr int kRmsNormThreadsPerBlock = 256;
constexpr int kRmsNormWarpSize = 32;
constexpr int kRmsNormBlockNumWarps = kRmsNormThreadsPerBlock / kRmsNormWarpSize;

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
                                  int hiddenSize);

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
                             int hiddenSize);
