// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for softmax.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kSoftmaxThreadsPerBlock = 1024;
constexpr int kSoftmaxWarpSize = 32;
constexpr int kSoftmaxWarpCount = kSoftmaxThreadsPerBlock / kSoftmaxWarpSize;

/**
 * @brief Kernel that computes per-row max and sum for softmax.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxRowMaxRowSumFp32Kernel(const float* __restrict__ x,
                                              float* __restrict__ rowMax,
                                              float* __restrict__ rowSum,
                                              int batchSize,
                                              int featureSize);

/**
 * @brief Kernel that computes softmax output using precomputed row max and sum.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param softmaxOut Device pointer to output.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxFp32Kernel(const float* __restrict__ x,
                                  const float* __restrict__ rowMax,
                                  const float* __restrict__ rowSum,
                                  float* __restrict__ softmaxOut,
                                  int batchSize,
                                  int featureSize);

/**
 * @brief Launches the softmax row max/sum kernel with configured grid/block sizes.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to output row max values.
 * @param rowSum Device pointer to output row sum values.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxRowMaxRowSumFp32Kernel(const float* x,
                                         float* rowMax,
                                         float* rowSum,
                                         int batchSize,
                                         int featureSize);

/**
 * @brief Launches the softmax kernel with configured grid/block sizes.
 * @param x Device pointer to input.
 * @param rowMax Device pointer to input row max values.
 * @param rowSum Device pointer to input row sum values.
 * @param softmaxOut Device pointer to output.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxFp32Kernel(const float* x,
                             const float* rowMax,
                             const float* rowSum,
                             float* softmaxOut,
                             int batchSize,
                             int featureSize);
