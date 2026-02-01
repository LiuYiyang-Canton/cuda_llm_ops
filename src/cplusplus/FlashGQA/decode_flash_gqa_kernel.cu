// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for FlashGQA decoding phase of inference.
// ==============================================================================
#include "FlashGQA/decode_flash_gqa_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;
using bf16_t = __nv_bfloat16;
namespace cg = cooperative_groups;

// --- Kernel Configuration ---
#define FLASH_BLOCK_SIZE 128

#define FRAGMENT_SIZE 16

#define SHARED_MEM_BF16_PADDING 8
#define SHARED_MEM_FP32_PADDING 0

#define NUM_ELEMENTS_PER_LOAD (sizeof(int4) / sizeof(__nv_bfloat16))
#define NUM_ELEMENTS_PER_FP32_LOAD (sizeof(int4) / sizeof(float))

#define MM1_BLOCK_M 16
#define MM1_BLOCK_N FLASH_BLOCK_SIZE

#define MM1_BLOCK_NUM_WARPS_M 1
#define MM1_BLOCK_NUM_WARPS_N 8
#define BLOCK_NUM_WARPS (MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N)

#define MM1_WARP_M (MM1_BLOCK_M / MM1_BLOCK_NUM_WARPS_M)
#define MM1_WARP_N (MM1_BLOCK_N / MM1_BLOCK_NUM_WARPS_N)

#define MM1_WARP_NUM_FRAGMENTS_M (MM1_WARP_M / FRAGMENT_SIZE)
#define MM1_WARP_NUM_FRAGMENTS_N (MM1_WARP_N / FRAGMENT_SIZE)

#define BLOCK_NUM_THREADS (MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N * WARP_SIZE)

#define MM1_K_FRAGMENTS_PER_ITER 2

#define MM1_A_TILE_M MM1_BLOCK_M
#define MM1_A_TILE_K (FRAGMENT_SIZE * MM1_K_FRAGMENTS_PER_ITER)

#define MM1_B_TILE_N MM1_BLOCK_N
#define MM1_B_TILE_K (FRAGMENT_SIZE * MM1_K_FRAGMENTS_PER_ITER)

#define MM1_A_TILE_NUM_THREADS_PER_ROW (MM1_A_TILE_K / NUM_ELEMENTS_PER_LOAD)
#define MM1_A_TILE_NUM_ROWS_PER_WARP (MM1_A_TILE_M / BLOCK_NUM_WARPS)
#define MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM1_A_TILE_NUM_THREADS_PER_ROW)
#define MM1_A_TILE_NUM_LOAD_ITERS CEIL_DIV(MM1_A_TILE_NUM_ROWS_PER_WARP, MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER)

#define MM1_B_TILE_NUM_THREADS_PER_COL (MM1_B_TILE_K / NUM_ELEMENTS_PER_LOAD)
#define MM1_B_TILE_NUM_COLS_PER_WARP (MM1_B_TILE_N / BLOCK_NUM_WARPS)
#define MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER (WARP_SIZE / MM1_B_TILE_NUM_THREADS_PER_COL)
#define MM1_B_TILE_NUM_LOAD_ITERS (MM1_B_TILE_NUM_COLS_PER_WARP / (MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER))

#define MM1_SHARED_MEM_BF16_STRIDE (MM1_A_TILE_K + SHARED_MEM_BF16_PADDING)
#define MM1_SHARED_MEM_FP32_STRIDE (MM1_BLOCK_N + SHARED_MEM_FP32_PADDING)

#define SOFTMAX_SHARED_MEM_BF16_STRIDE (MM1_BLOCK_N + SHARED_MEM_BF16_PADDING)
#define SOFTMAX_NUM_ROWS_PER_WARP (MM1_BLOCK_M / BLOCK_NUM_WARPS)

#define MM2_BLOCK_M 16
#define MM2_BLOCK_N 128

#define MM2_BLOCK_NUM_WARPS_M 1
#define MM2_BLOCK_NUM_WARPS_N 8

#define MM2_WARP_M (MM2_BLOCK_M / MM2_BLOCK_NUM_WARPS_M)
#define MM2_WARP_N (MM2_BLOCK_N / MM2_BLOCK_NUM_WARPS_N)

#define MM2_WARP_NUM_FRAGMENTS_M (MM2_WARP_M / FRAGMENT_SIZE)
#define MM2_WARP_NUM_FRAGMENTS_N (MM2_WARP_N / FRAGMENT_SIZE)

#define MM2_K_FRAGMENTS_PER_ITER 8

#define MM2_A_TILE_M MM2_BLOCK_M
#define MM2_A_TILE_K (FRAGMENT_SIZE * MM2_K_FRAGMENTS_PER_ITER)

#define MM2_B_TILE_K (FRAGMENT_SIZE * MM2_K_FRAGMENTS_PER_ITER)
#define MM2_B_TILE_N MM2_BLOCK_N

#define MM2_B_TILE_NUM_THREADS_PER_COL (MM2_B_TILE_N / NUM_ELEMENTS_PER_LOAD)
#define MM2_B_TILE_NUM_ROWS_PER_WARP (MM2_B_TILE_K / BLOCK_NUM_WARPS)
#define MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM2_B_TILE_NUM_THREADS_PER_COL)
#define MM2_B_TILE_NUM_LOAD_ITERS (MM2_B_TILE_NUM_ROWS_PER_WARP / (MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER))

#define MM2_O_TILE_NUM_THREADS_PER_ROW (MM2_BLOCK_N / NUM_ELEMENTS_PER_FP32_LOAD)
#define MM2_O_TILE_NUM_ROWS_PER_WARP (MM2_BLOCK_M / BLOCK_NUM_WARPS)
#define MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM2_O_TILE_NUM_THREADS_PER_ROW)
#define MM2_O_TILE_NUM_LOAD_ITERS (MM2_O_TILE_NUM_ROWS_PER_WARP / (MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER))

#define MM2_SHARED_MEM_BF16_STRIDE (SOFTMAX_SHARED_MEM_BF16_STRIDE)
#define MM2_V_SHARED_MEM_BF16_STRIDE (MM2_B_TILE_N + SHARED_MEM_BF16_PADDING)
#define MM2_SHARED_MEM_FP32_STRIDE (MM2_BLOCK_N + SHARED_MEM_FP32_PADDING)

#define UPDATE_O_NUM_ROWS_PER_WARP (MM2_BLOCK_M / BLOCK_NUM_WARPS)

/**
 * @brief FlashGQA decoding kernel for bf16 inputs/outputs with fp32 accumulation.
 * @param Q Pointer to query tensor.
 * @param K Pointer to key tensor.
 * @param V Pointer to value tensor.
 * @param scale Scaling factor applied to QK scores.
 * @param O Pointer to output tensor.
 * @param OIntermediate Pointer to intermediate output buffer.
 * @param batchSize Batch size.
 * @param seqLength Sequence length.
 * @param numQueryHeads Number of query heads.
 * @param numKvHeads Number of KV heads.
 * @param headDim Per-head embedding dimension.
 */
__global__ void decode_flash_gqa_bf16_kernel(const bf16_t* __restrict__ Q,
                                            const bf16_t* __restrict__ K,
                                            const bf16_t* __restrict__ V,
                                            float scale,
                                            bf16_t* __restrict__ O,
                                            float* __restrict__ OIntermediate,
                                            int batchSize,
                                            int seqLength,
                                            int numQueryHeads,
                                            int numKvHeads,
                                            int headDim) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batchIdx = by;
    int groupIdx = bx;
    int warpID = tx / WARP_SIZE;
    int laneID = tx % WARP_SIZE;

    if (batchIdx >= batchSize) {
        return;
    }

    const int groupSize = numQueryHeads / numKvHeads;
    const int flashNumBlocks = CEIL_DIV(seqLength, FLASH_BLOCK_SIZE);
    Q = &Q[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * groupSize, 0, numQueryHeads, headDim)];
    K = &K[GET_1D_INDEX_FROM_4D(batchIdx, 0, groupIdx, 0, seqLength, numKvHeads, headDim)];
    V = &V[GET_1D_INDEX_FROM_4D(batchIdx, 0, groupIdx, 0, seqLength, numKvHeads, headDim)];
    O = &O[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * groupSize, 0, numQueryHeads, headDim)];
    OIntermediate = &OIntermediate[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * groupSize, 0, numQueryHeads, headDim)];

    extern __shared__ unsigned char shared_storage[];
    size_t sharedOffset = 0;
    float* rowmax = reinterpret_cast<float*>(shared_storage + sharedOffset);
    sharedOffset += static_cast<size_t>(groupSize) * sizeof(float);
    float* newrowmax = reinterpret_cast<float*>(shared_storage + sharedOffset);
    sharedOffset += static_cast<size_t>(groupSize) * sizeof(float);
    float* rowsum = reinterpret_cast<float*>(shared_storage + sharedOffset);
    sharedOffset += static_cast<size_t>(groupSize) * sizeof(float);
    float* newrowsum = reinterpret_cast<float*>(shared_storage + sharedOffset);
    sharedOffset += static_cast<size_t>(groupSize) * sizeof(float);
    const size_t bf16Alignment = alignof(bf16_t);
    sharedOffset = (sharedOffset + bf16Alignment - 1) / bf16Alignment * bf16Alignment;
    bf16_t* shared_mem = reinterpret_cast<bf16_t*>(shared_storage + sharedOffset);

    if (tx == 0) {
        for (int head = 0; head < groupSize; ++head) {
            rowmax[head] = -INFINITY;
            rowsum[head] = 0.0;
        }
    }

    bf16_t* qs = &shared_mem[0];
    bf16_t* ks = &shared_mem[MM1_A_TILE_M * MM1_SHARED_MEM_BF16_STRIDE];
    for (int flashIter = 0; flashIter < flashNumBlocks; ++flashIter) {
        wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float>
            sFrag[MM1_WARP_NUM_FRAGMENTS_M][MM1_WARP_NUM_FRAGMENTS_N];

    #pragma unroll
        for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                wmma::fill_fragment(sFrag[i][j], 0.0f);
            }
        }

    #pragma unroll
        for (int innerTile = 0; innerTile < headDim; innerTile += MM1_A_TILE_K) {
    #pragma unroll
            for (int iter = 0; iter < MM1_A_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM1_A_TILE_NUM_ROWS_PER_WARP + (laneID / MM1_A_TILE_NUM_THREADS_PER_ROW) +
                          iter * MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER;
                if (row < (warpID + 1) * MM1_A_TILE_NUM_ROWS_PER_WARP) {
                    int col = (laneID % MM1_A_TILE_NUM_THREADS_PER_ROW) * NUM_ELEMENTS_PER_LOAD;
                    *(int4*)&qs[row * MM1_SHARED_MEM_BF16_STRIDE + col] =
                        *(int4*)&Q[row * headDim + innerTile + col];
                }
            }
    #pragma unroll
            for (int iter = 0; iter < MM1_B_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM1_B_TILE_NUM_COLS_PER_WARP + (laneID / MM1_B_TILE_NUM_THREADS_PER_COL) +
                          iter * MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER;
                int col = (laneID % MM1_B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
                *(int4*)&ks[row * MM1_SHARED_MEM_BF16_STRIDE + col] =
                    *(int4*)&K[row * numKvHeads * headDim + innerTile + col];
            }
            __syncthreads();
    #pragma unroll
            for (int innerFrag = 0; innerFrag < MM1_K_FRAGMENTS_PER_ITER; ++innerFrag) {
                wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major>
                    qFrag[MM1_WARP_NUM_FRAGMENTS_M];
                wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::col_major>
                    kFrag[MM1_WARP_NUM_FRAGMENTS_N];
    #pragma unroll
                for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
                    int row = warpID / MM1_BLOCK_NUM_WARPS_N * MM1_WARP_M + i * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(qFrag[i], &qs[row * MM1_SHARED_MEM_BF16_STRIDE + col],
                                           MM1_SHARED_MEM_BF16_STRIDE);
                }
    #pragma unroll
                for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                    int row = warpID % MM1_BLOCK_NUM_WARPS_N * MM1_WARP_N + j * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(kFrag[j], &ks[row * MM1_SHARED_MEM_BF16_STRIDE + col],
                                           MM1_SHARED_MEM_BF16_STRIDE);
                }

    #pragma unroll
                for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
                    for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                        wmma::mma_sync(sFrag[i][j], qFrag[i], kFrag[j], sFrag[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        float* mm1Result = (float*)&shared_mem[MM1_BLOCK_M * SOFTMAX_SHARED_MEM_BF16_STRIDE];
    #pragma unroll
        for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                int row = (warpID / MM1_BLOCK_NUM_WARPS_N) * MM1_WARP_M + i * FRAGMENT_SIZE;
                int col = (warpID % MM1_BLOCK_NUM_WARPS_N) * MM1_WARP_N + j * FRAGMENT_SIZE;
                wmma::store_matrix_sync(&mm1Result[row * (MM1_SHARED_MEM_FP32_STRIDE) + col], sFrag[i][j],
                                        MM1_SHARED_MEM_FP32_STRIDE, wmma::mem_row_major);
            }
        }
        __syncthreads();

        bf16_t* ps = &shared_mem[0];
        auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
        for (int softmaxIter = 0; softmaxIter < SOFTMAX_NUM_ROWS_PER_WARP; ++softmaxIter) {
            int row = warpID * SOFTMAX_NUM_ROWS_PER_WARP + softmaxIter;
            float rowmaxLocal = rowmax[row];
            float rowsumLocal = 0.f;
            if (laneID * 4 >= FLASH_BLOCK_SIZE) {
                break;
            }
            float4 sValue = *(float4*)&mm1Result[row * MM1_SHARED_MEM_FP32_STRIDE + laneID * 4];
            rowmaxLocal = fmaxf(sValue.x * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.y * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.z * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.w * scale, rowmaxLocal);
            newrowmax[row] = cg::reduce(warp, rowmaxLocal, cg::greater<float>());
            __syncwarp();
            rowsumLocal = expf(sValue.x * scale - newrowmax[row]) + expf(sValue.y * scale - newrowmax[row]) +
                          expf(sValue.z * scale - newrowmax[row]) + expf(sValue.w * scale - newrowmax[row]);
            newrowsum[row] = cg::reduce(warp, rowsumLocal, cg::plus<float>());
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4] =
                __float2bfloat16(expf(sValue.x * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 1] =
                __float2bfloat16(expf(sValue.y * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 2] =
                __float2bfloat16(expf(sValue.z * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 3] =
                __float2bfloat16(expf(sValue.w * scale - newrowmax[row]));
        }

        __syncthreads();
        wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float>
            oFrag[MM2_WARP_NUM_FRAGMENTS_M][MM2_WARP_NUM_FRAGMENTS_N];

    #pragma unroll
        for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                wmma::fill_fragment(oFrag[i][j], 0.0f);
            }
        }

        bf16_t* vs = &shared_mem[MM2_A_TILE_M * MM2_SHARED_MEM_BF16_STRIDE];
    #pragma unroll
        for (int innerTile = 0; innerTile < FLASH_BLOCK_SIZE; innerTile += MM2_B_TILE_K) {
    #pragma unroll
            for (int iter = 0; iter < MM2_B_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM2_B_TILE_NUM_ROWS_PER_WARP + (laneID / MM2_B_TILE_NUM_THREADS_PER_COL) +
                          iter * MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER;
                int col = (laneID % MM2_B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
                *(int4*)&vs[row * MM2_V_SHARED_MEM_BF16_STRIDE + col] =
                    *(int4*)&V[(row + innerTile) * numKvHeads * headDim + col];
            }
            __syncthreads();
    #pragma unroll
            for (int innerFrag = 0; innerFrag < MM2_K_FRAGMENTS_PER_ITER; ++innerFrag) {
                wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major>
                    pFrag[MM2_WARP_NUM_FRAGMENTS_M];
                wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major>
                    vFrag[MM2_WARP_NUM_FRAGMENTS_N];
    #pragma unroll
                for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
                    int row = warpID / MM2_BLOCK_NUM_WARPS_N * MM2_WARP_M + i * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(pFrag[i], &ps[row * MM2_SHARED_MEM_BF16_STRIDE + innerTile + col],
                                           MM2_SHARED_MEM_BF16_STRIDE);
                }
    #pragma unroll
                for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                    int row = innerFrag * FRAGMENT_SIZE;
                    int col = warpID % MM2_BLOCK_NUM_WARPS_N * MM2_WARP_N + j * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(vFrag[j], &vs[row * MM2_V_SHARED_MEM_BF16_STRIDE + col],
                                           MM2_V_SHARED_MEM_BF16_STRIDE);
                }

    #pragma unroll
                for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
                    for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                        wmma::mma_sync(oFrag[i][j], pFrag[i], vFrag[j], oFrag[i][j]);
                    }
                }
            }

            __syncthreads();
        }
        float* mm2Result = (float*)&shared_mem[0];
    #pragma unroll
        for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                int row = (warpID / MM2_BLOCK_NUM_WARPS_N) * MM2_WARP_M + i * FRAGMENT_SIZE;
                int col = (warpID % MM2_BLOCK_NUM_WARPS_N) * MM2_WARP_N + j * FRAGMENT_SIZE;
                wmma::store_matrix_sync(&mm2Result[row * (MM2_SHARED_MEM_FP32_STRIDE) + col], oFrag[i][j],
                                        MM2_SHARED_MEM_FP32_STRIDE, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int iter = 0; iter < MM2_O_TILE_NUM_LOAD_ITERS; ++iter) {
            int row = warpID * MM2_O_TILE_NUM_ROWS_PER_WARP + (laneID / MM2_O_TILE_NUM_THREADS_PER_ROW) +
                      iter * MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER;
            int col = (laneID % MM2_O_TILE_NUM_THREADS_PER_ROW) * NUM_ELEMENTS_PER_FP32_LOAD;
            float updateCoef = expf(rowmax[row] - newrowmax[row]);
            if (flashIter > 0) {
                float4 oValueOld = *(float4*)&OIntermediate[row * headDim + col];
                float4 oValueNew = *(float4*)&mm2Result[row * MM2_SHARED_MEM_FP32_STRIDE + col];
                *(float4*)&OIntermediate[row * headDim + col] = {oValueOld.x * updateCoef + oValueNew.x,
                                                                 oValueOld.y * updateCoef + oValueNew.y,
                                                                 oValueOld.z * updateCoef + oValueNew.z,
                                                                 oValueOld.w * updateCoef + oValueNew.w};
            } else {
                *(float4*)&OIntermediate[row * headDim + col] =
                    *(float4*)&mm2Result[row * MM2_SHARED_MEM_FP32_STRIDE + col];
            }
        }
        __syncthreads();
        if (tx < groupSize) {
            float updateCoef = expf(rowmax[tx] - newrowmax[tx]);
            rowsum[tx] = rowsum[tx] * updateCoef + newrowsum[tx];
            rowmax[tx] = newrowmax[tx];
        }
        K += FLASH_BLOCK_SIZE * numKvHeads * headDim;
        V += FLASH_BLOCK_SIZE * numKvHeads * headDim;
    }

    __syncthreads();
    for (int updateOIter = 0; updateOIter < UPDATE_O_NUM_ROWS_PER_WARP; ++updateOIter) {
        int row = warpID * UPDATE_O_NUM_ROWS_PER_WARP + updateOIter;
        if (laneID * 4 >= headDim) {
            break;
        }
        float normalizer = 1.0f / rowsum[row];
        float4 oValue = *(float4*)&OIntermediate[row * MM2_SHARED_MEM_FP32_STRIDE + laneID * 4];
        O[row * headDim + laneID * 4] = __float2bfloat16(oValue.x * normalizer);
        O[row * headDim + laneID * 4 + 1] = __float2bfloat16(oValue.y * normalizer);
        O[row * headDim + laneID * 4 + 2] = __float2bfloat16(oValue.z * normalizer);
        O[row * headDim + laneID * 4 + 3] = __float2bfloat16(oValue.w * normalizer);
    }
}

/**
 * @brief Launches the FlashGQA bf16 decoding kernel.
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
void LaunchFlashGqaBf16Kernel(const bf16_t* q,
                              const bf16_t* k,
                              const bf16_t* v,
                              float scale,
                              bf16_t* o,
                              float* oIntermediate,
                              int batchSize,
                              int seqLength,
                              int numQueryHeads,
                              int numKvHeads,
                              int headDim) {
    if (batchSize <= 0 || seqLength <= 0) {
        return;
    }
    if (numQueryHeads <= 0 || numKvHeads <= 0 || headDim <= 0) {
        return;
    }
    if (numQueryHeads % numKvHeads != 0) {
        return;
    }
    const int groupSize = numQueryHeads / numKvHeads;
    if (groupSize != MM1_BLOCK_M || groupSize != MM2_BLOCK_M) {
        return;
    }
    if (headDim != MM2_BLOCK_N) {
        return;
    }
    if ((headDim % MM1_A_TILE_K) != 0) {
        return;
    }
    if ((seqLength % FLASH_BLOCK_SIZE) != 0) {
        return;
    }
    const dim3 numThreads(BLOCK_NUM_THREADS);
    const dim3 numBlocks(numKvHeads, batchSize);
    size_t tileBytes = std::max((MM1_A_TILE_M + MM1_B_TILE_N) * MM1_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t),
                                MM1_BLOCK_M * SOFTMAX_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t) +
                                    MM1_BLOCK_M * MM1_SHARED_MEM_FP32_STRIDE * sizeof(float));
    tileBytes = std::max(tileBytes,
                         (MM2_A_TILE_M * MM2_SHARED_MEM_BF16_STRIDE +
                          MM2_B_TILE_K * MM2_V_SHARED_MEM_BF16_STRIDE) * sizeof(bf16_t));
    size_t headerBytes = static_cast<size_t>(groupSize) * 4 * sizeof(float);
    const size_t bf16Alignment = alignof(bf16_t);
    headerBytes = (headerBytes + bf16Alignment - 1) / bf16Alignment * bf16Alignment;
    const size_t shmemBytes = headerBytes + tileBytes;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (shmemBytes > deviceProp.sharedMemPerBlock) {
        cudaFuncSetAttribute(decode_flash_gqa_bf16_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmemBytes));
    }

    decode_flash_gqa_bf16_kernel<<<numBlocks, numThreads, shmemBytes>>>(q,
                                                                       k,
                                                                       v,
                                                                       scale,
                                                                       o,
                                                                       oIntermediate,
                                                                       batchSize,
                                                                       seqLength,
                                                                       numQueryHeads,
                                                                       numKvHeads,
                                                                       headDim);
}
