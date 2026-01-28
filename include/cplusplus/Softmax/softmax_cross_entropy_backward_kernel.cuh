// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for softmax cross-entropy backward pass.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kSoftmaxCrossEntropyBackwardThreadsPerBlock = 1024;
constexpr int kSoftmaxCrossEntropyBackwardWarpSize = 32;
constexpr int kSoftmaxCrossEntropyBackwardWarpCount =
    kSoftmaxCrossEntropyBackwardThreadsPerBlock / kSoftmaxCrossEntropyBackwardWarpSize;

/**
 * @brief Kernel that computes per-row max and sum for softmax.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel(const float* __restrict__ logits,
                                                                  float* __restrict__ rowMax,
                                                                  float* __restrict__ rowSum,
                                                                  int batchSize,
                                                                  int featureSize);

/**
 * @brief Kernel that computes softmax-cross-entropy gradient for each row.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param labels Device pointer to label indices.
 * @param gradLogits Device pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxCrossEntropyBackwardFp32Kernel(const float* __restrict__ logits,
                                                      const float* __restrict__ rowMax,
                                                      const float* __restrict__ rowSum,
                                                      const int* __restrict__ labels,
                                                      float* __restrict__ gradLogits,
                                                      int batchSize,
                                                      int featureSize);

/**
 * @brief Launches the row max/sum kernel for softmax cross-entropy backward.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel(const float* logits,
                                                             float* rowMax,
                                                             float* rowSum,
                                                             int batchSize,
                                                             int featureSize);

/**
 * @brief Launches the softmax cross-entropy backward kernel with configured grid/block sizes.
 * @param logits Device pointer to input logits.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param labels Device pointer to label indices.
 * @param gradLogits Device pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxCrossEntropyBackwardFp32Kernel(const float* logits,
                                                 const float* rowMax,
                                                 const float* rowSum,
                                                 const int* labels,
                                                 float* gradLogits,
                                                 int batchSize,
                                                 int featureSize);
