// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for RoPE.
// ==============================================================================
#pragma once

#include <cuda_bf16.h>

constexpr int kRopeValuesPerThread = 8;
constexpr int kRopeThreadsPerBlock = 256;

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
                          float base = 10000.0f);
