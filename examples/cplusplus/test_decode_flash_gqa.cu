// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for FlashGQA decoding phase of inference kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <omp.h>

using bf16_t = __nv_bfloat16;

// Model and kernel configuration (must match src/cplusplus/FlashGQA/decode_flash_gqa_kernel.cu).
#define NUM_QUERY_HEADS 128
#define NUM_KV_HEADS 8
#define GROUP_SIZE ((NUM_QUERY_HEADS) / (NUM_KV_HEADS))
#define HEAD_DIM 128

#define FLASH_BLOCK_SIZE 128

#define FRAGMENT_SIZE 16

#define SHARED_MEM_BF16_PADDING 8
#define SHARED_MEM_FP32_PADDING 0

#define NUM_ELEMENTS_PER_LOAD (sizeof(int4) / sizeof(__nv_bfloat16))
#define NUM_ELEMENTS_PER_FP32_LOAD (sizeof(int4) / sizeof(float))

#define MM1_BLOCK_M GROUP_SIZE
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

#define MM2_BLOCK_M GROUP_SIZE
#define MM2_BLOCK_N HEAD_DIM

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

constexpr int kFlashBatchSize = 60;
constexpr int kFlashSeqLength = 4096;
constexpr int kFlashNumBlocks = CEIL_DIV(kFlashSeqLength, FLASH_BLOCK_SIZE);

/**
 * @brief Computes RMSE between two bf16 buffers.
 * @param golden Pointer to reference data.
 * @param values Pointer to computed data.
 * @param numElements Number of elements to compare.
 * @return Relative RMSE.
 */
float ComputeBF16RMSE(const bf16_t* __restrict__ golden, const bf16_t* __restrict__ values, size_t numElements) {
    float error = 0.0f;
    float norm = 0.0f;
    for (size_t i = 0; i < numElements; ++i) {
        const float diff = __bfloat162float(golden[i]) - __bfloat162float(values[i]);
        error += diff * diff;
        norm += __bfloat162float(golden[i]) * __bfloat162float(golden[i]);
    }
    return std::sqrt(error) / std::sqrt(norm);
}

/**
 * @brief Computes RMSE between two fp32 buffers.
 * @param golden Pointer to reference data.
 * @param values Pointer to computed data.
 * @param numElements Number of elements to compare.
 * @return Relative RMSE.
 */
float ComputeRMSE(const float* __restrict__ golden, const float* __restrict__ values, size_t numElements) {
    float error = 0.0f;
    float norm = 0.0f;
    for (size_t i = 0; i < numElements; ++i) {
        error += (golden[i] - values[i]) * (golden[i] - values[i]);
        norm += golden[i] * golden[i];
    }
    return std::sqrt(error) / std::sqrt(norm);
}

/**
 * @brief Runs the FlashGQA benchmark and validation.
 * @return Process exit code.
 */
int main() {
    assert(kFlashSeqLength % 4 == 0);
    assert(kFlashSeqLength % FLASH_BLOCK_SIZE == 0);
    assert(MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N == MM2_BLOCK_NUM_WARPS_M * MM2_BLOCK_NUM_WARPS_N);
    assert(SOFTMAX_SHARED_MEM_BF16_STRIDE == MM2_SHARED_MEM_BF16_STRIDE);
    assert(WARP_SIZE * 4 >= FLASH_BLOCK_SIZE);
    assert(WARP_SIZE * 4 >= HEAD_DIM);

    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    bf16_t* QHost = new bf16_t[kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM];
    bf16_t* KHost = new bf16_t[kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM];
    bf16_t* VHost = new bf16_t[kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM];
    for (int i = 0; i < kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM; ++i) {
        QHost[i] = bf16_t(distribution(generator));
    }
    for (int i = 0; i < kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM; ++i) {
        KHost[i] = bf16_t(distribution(generator));
    }
    for (int i = 0; i < kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM; ++i) {
        VHost[i] = bf16_t(distribution(generator));
    }
    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    float* SHostCPU = new float[kFlashBatchSize * NUM_QUERY_HEADS * FLASH_BLOCK_SIZE];
    bf16_t* PHostCPU = new bf16_t[kFlashBatchSize * NUM_QUERY_HEADS * FLASH_BLOCK_SIZE];
    float* OIntermediateHostCPU = new float[kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM];
    bf16_t* OHostCPU = new bf16_t[kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM];
    double t = omp_get_wtime();
#pragma omp parallel for
    for (int batch = 0; batch < kFlashBatchSize; ++batch) {
        for (int head = 0; head < NUM_QUERY_HEADS; ++head) {
            float rowmax = -INFINITY;
            float newrowmax;
            float rowsum = 0.f;
            float newrowsum;
            for (int flashIter = 0; flashIter < kFlashNumBlocks; ++flashIter) {
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    float qkScore = 0.f;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        qkScore += __bfloat162float(QHost[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)]) *
                                   __bfloat162float(KHost[GET_1D_INDEX_FROM_4D(batch, token, head / GROUP_SIZE, d, kFlashSeqLength, NUM_KV_HEADS, HEAD_DIM)]);
                    }
                    SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS,
                                                  FLASH_BLOCK_SIZE)] = qkScore;
                }
                newrowmax = rowmax;
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    newrowmax = fmaxf(newrowmax,
                                      SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE,
                                                                   NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] *
                                          scale);
                }
                newrowsum = 0.f;
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    newrowsum += expf(SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE,
                                                                   NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] *
                                          scale -
                                      newrowmax);
                }
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    PHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS,
                                                  FLASH_BLOCK_SIZE)] =
                        bf16_t(expf(SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE,
                                                                 NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] *
                                        scale -
                                    newrowmax));
                }

                for (int d = 0; d < HEAD_DIM; ++d) {
                    float o = 0.f;
                    for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                        o += __bfloat162float(PHostCPU[GET_1D_INDEX_FROM_3D(batch, head,
                                                                         token - flashIter * FLASH_BLOCK_SIZE,
                                                                         NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)]) *
                             __bfloat162float(VHost[GET_1D_INDEX_FROM_4D(batch, token, head / GROUP_SIZE, d, kFlashSeqLength,
                                                                         NUM_KV_HEADS, HEAD_DIM)]);
                    }
                    if (flashIter == 0) {
                        OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] = o;
                    } else {
                        OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] =
                            OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] *
                                expf(rowmax - newrowmax) +
                            o;
                    }
                }
                rowsum = rowsum * expf(rowmax - newrowmax) + newrowsum;
                rowmax = newrowmax;
            }

            for (int d = 0; d < HEAD_DIM; ++d) {
                OHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] =
                    __float2bfloat16(OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS,
                                                                             HEAD_DIM)] /
                                     rowsum);
            }
        }
    }
    std::cout << "CPU time = " << omp_get_wtime() - t << "s" << std::endl;

    bf16_t* QDevice;
    bf16_t* KDevice;
    bf16_t* VDevice;
    bf16_t* ODevice;
    float* OIntermediateDevice;
    cudaMalloc(&QDevice, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&KDevice, kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&VDevice, kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&ODevice, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&OIntermediateDevice, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(float));
    cudaMemcpy(QDevice, QHost, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(KDevice, KHost, kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(VDevice, VHost, kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t),
               cudaMemcpyHostToDevice);

    float* SDevice;
    cudaMalloc(&SDevice, kFlashBatchSize * NUM_QUERY_HEADS * kFlashSeqLength * sizeof(float));
    bf16_t* PDevice;
    cudaMalloc(&PDevice, kFlashBatchSize * NUM_QUERY_HEADS * kFlashSeqLength * sizeof(bf16_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    size_t shmemBytes =
        std::max((MM1_A_TILE_M + MM1_B_TILE_N) * MM1_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t),
                 MM1_BLOCK_M * SOFTMAX_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t) +
                     MM1_BLOCK_M * MM1_SHARED_MEM_FP32_STRIDE * sizeof(float));
    shmemBytes = std::max(shmemBytes,
                          (MM2_A_TILE_M * MM2_SHARED_MEM_BF16_STRIDE +
                           MM2_B_TILE_K * MM2_V_SHARED_MEM_BF16_STRIDE) * sizeof(bf16_t));
    std::cout << "shared memory usage per block: " << shmemBytes / 1024 << " KB" << std::endl;

    cudaMemset(ODevice, 0, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMemset(OIntermediateDevice, 0, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(float));

    cudaEventRecord(start);
    LaunchFlashGqaBf16Kernel(QDevice,
                             KDevice,
                             VDevice,
                             scale,
                             ODevice,
                             OIntermediateDevice,
                             kFlashBatchSize,
                             kFlashSeqLength,
                             NUM_QUERY_HEADS,
                             NUM_KV_HEADS,
                             HEAD_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "decode_flash_gqa_bf16_kernel elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    float reached_mem_bw =
        (kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t) + kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM *
                                                                           sizeof(bf16_t) +
         kFlashBatchSize * kFlashSeqLength * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t) +
         kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t)) /
        ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "decode_flash_gqa_bf16_kernel reached_mem_bw: " << reached_mem_bw << " TB/s" << std::endl;
    float peakTFLOPS = ((float)kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * kFlashSeqLength) * 2 / 1e12 / (elapsedTime / 1e3);
    std::cout << "decode_flash_gqa_bf16_kernel peak TFLOPS: " << peakTFLOPS << " TFLOPS" << std::endl;

    bf16_t* OHost = new bf16_t[kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM];
    cudaMemcpy(OHost, ODevice, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    float error = ComputeBF16RMSE(OHostCPU, OHost, kFlashBatchSize * NUM_QUERY_HEADS * HEAD_DIM);
    std::cout << "O error = " << error << std::endl;

    delete[] QHost;
    delete[] KHost;
    delete[] VHost;
    delete[] OHost;
    delete[] SHostCPU;
    delete[] PHostCPU;
    delete[] OHostCPU;

    return 0;
}
