// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for softmax backward pass.
// ==============================================================================
#pragma once

#include <cuda_runtime.h>

constexpr int kSoftmaxBackwardThreadsPerBlock = 256;
constexpr int kSoftmaxBackwardWorkPerThread = 8;

/**
 * @brief Computes per-row dot(s, g) and writes s*g into a temporary buffer.
 * @param gradOutput Device pointer to upstream gradient.
 * @param softmaxOutput Device pointer to softmax output.
 * @param gradX Device pointer to s*g buffer.
 * @param dotProduct Device pointer to per-row dot(s, g) accumulator.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxBackwardDotKernel(const float* gradOutput,
                                         const float* softmaxOutput,
                                         float* gradX,
                                         float* dotProduct,
                                         int batchSize,
                                         int featureSize);

/**
 * @brief Computes final gradient: gradX = s*g - s*dot(s, g).
 * @param softmaxOutput Device pointer to softmax output.
 * @param dotProduct Device pointer to per-row dot(s, g).
 * @param gradX Device pointer to output gradient (in-place over s*g buffer).
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
__global__ void SoftmaxBackwardGradKernel(const float* softmaxOutput,
                                          const float* dotProduct,
                                          float* gradX,
                                          int batchSize,
                                          int featureSize);

/**
 * @brief Launches the softmax backward dot kernel with configured grid/block sizes.
 * @param gradOutput Device pointer to upstream gradient.
 * @param softmaxOutput Device pointer to softmax output.
 * @param gradX Device pointer to s*g buffer.
 * @param dotProduct Device pointer to per-row dot(s, g) accumulator.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxBackwardDotKernel(const float* gradOutput,
                                    const float* softmaxOutput,
                                    float* gradX,
                                    float* dotProduct,
                                    int batchSize,
                                    int featureSize);

/**
 * @brief Launches the softmax backward gradient kernel with configured grid/block sizes.
 * @param softmaxOutput Device pointer to softmax output.
 * @param dotProduct Device pointer to per-row dot(s, g).
 * @param gradX Device pointer to output gradient (in-place over s*g buffer).
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void LaunchSoftmaxBackwardGradKernel(const float* softmaxOutput,
                                     const float* dotProduct,
                                     float* gradX,
                                     int batchSize,
                                     int featureSize);
