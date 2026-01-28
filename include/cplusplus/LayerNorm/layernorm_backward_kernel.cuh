// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for LayerNorm backward pass.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kLayerNormBackwardThreadsPerBlock = 256;

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
                                        int hiddenDim);

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
                                   int hiddenDim);
