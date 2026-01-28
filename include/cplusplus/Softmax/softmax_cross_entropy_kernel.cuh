// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for softmax cross-entropy.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kSoftmaxCrossEntropyThreadsPerBlock = 1024;

/**
 * @brief Kernel for fused softmax + cross-entropy.
 * @param logits Device pointer to input logits.
 * @param labels Device pointer to label indices.
 * @param loss Device pointer to per-row loss, storing sum of exponentials temporarily.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyKernel(const float* __restrict__ logits,
                                          const int* __restrict__ labels,
                                          float* __restrict__ loss,
                                          int batchSize,
                                          int featureSize);

/**
 * @brief Launches the softmax cross-entropy kernel with configured grid/block sizes.
 * @param logits Device pointer to input logits.
 * @param labels Device pointer to label indices.
 * @param loss Device pointer to per-row loss.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyKernel(const float* logits,
                                     const int* labels,
                                     float* loss,
                                     int batchSize,
                                     int featureSize);
