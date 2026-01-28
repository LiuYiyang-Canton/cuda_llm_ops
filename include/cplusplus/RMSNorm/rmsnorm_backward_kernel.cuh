// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for RMSNorm backward pass.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kRmsNormBackwardThreadsPerBlock = 256;

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
                                      int hiddenDim);

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
                                 int hiddenDim);
