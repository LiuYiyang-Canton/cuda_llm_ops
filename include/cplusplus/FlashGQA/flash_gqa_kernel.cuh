// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel declarations for FlashGQA.
// ==============================================================================
#pragma once

#include <cuda_bf16.h>

#include "Utils/utils.cuh"

constexpr int WARP_SIZE = 32;

/**
 * @brief Launches the FlashGQA bf16 kernel.
 * @param q Pointer to device query tensor.
 * @param k Pointer to device key tensor.
 * @param v Pointer to device value tensor.
 * @param scale Scaling factor applied to QK scores.
 * @param o Pointer to device output tensor.
 * @param oIntermediate Pointer to device intermediate output buffer.
 * @param batchSize Batch size.
 * @param seqLength Sequence length.
 * @param numQueryHeads Number of query heads.
 * @param numKvHeads Number of KV heads.
 * @param headDim Per-head embedding dimension.
 */
void LaunchFlashGqaBf16Kernel(const __nv_bfloat16* q,
                              const __nv_bfloat16* k,
                              const __nv_bfloat16* v,
                              float scale,
                              __nv_bfloat16* o,
                              float* oIntermediate,
                              int batchSize,
                              int seqLength,
                              int numQueryHeads,
                              int numKvHeads,
                              int headDim);
