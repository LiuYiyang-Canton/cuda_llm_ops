// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for BF16 GLU.
// ==============================================================================
#include "GLU/glu_bf16_kernel.cuh"

#include <algorithm>
#include <mma.h>

using namespace nvcuda;

/**
 * @brief BF16 GLU kernel with fp32 accumulation.
 * @param x Device pointer to input matrix X.
 * @param weight Device pointer to weight matrix.
 * @param result Device pointer to output matrix.
 * @param k Size of the K dimension.
 * @param intermediateSize Size of the intermediate/output dimension.
 */
__global__ void GluBf16Kernel(const Bf16* __restrict__ x,
                              const Bf16* __restrict__ weight,
                              float* __restrict__ result,
                              int k,
                              int intermediateSize) {
    const int threadId = static_cast<int>(threadIdx.x);
    const int blockIdX = static_cast<int>(blockIdx.x);
    const int blockIdY = static_cast<int>(blockIdx.y);

    const int warpId = threadId / glu_config::kWarpSize;
    const int laneId = threadId % glu_config::kWarpSize;
    const int globalOutputRow = blockIdX * BLOCK_M;
    const int globalOutputCol = blockIdY * BLOCK_N;

    extern __shared__ Bf16 sharedMem[][glu_config::kSharedMemStride];
    Bf16* xShared = &sharedMem[0][0];
    Bf16* weight1Shared = &sharedMem[glu_config::kATileM][0];
    Bf16* weight2Shared = &sharedMem[glu_config::kATileM + glu_config::kBTileN][0];

    wmma::fragment<wmma::accumulator, glu_config::kFragmentSize, glu_config::kFragmentSize, glu_config::kFragmentSize, float>
        cFrag[2][glu_config::kWarpNumFragmentsM][glu_config::kWarpNumFragmentsN];

#pragma unroll
    for (int i = 0; i < glu_config::kWarpNumFragmentsM; ++i) {
#pragma unroll
        for (int j = 0; j < glu_config::kWarpNumFragmentsN; ++j) {
            wmma::fill_fragment(cFrag[0][i][j], 0.0f);
            wmma::fill_fragment(cFrag[1][i][j], 0.0f);
        }
    }

#pragma unroll
    for (int kTile = 0; kTile < k; kTile += glu_config::kATileK) {
#pragma unroll
        for (int iter = 0; iter < glu_config::kATileNumLoadIters; ++iter) {
            const int row = warpId * glu_config::kATileNumRowsPerWarp +
                            (laneId / glu_config::kATileNumThreadsPerRow) +
                            iter * glu_config::kATileNumRowsPerWarpPerIter;
            const int col = (laneId % glu_config::kATileNumThreadsPerRow) * glu_config::kNumElementsPerLoad;
            const int globalRow = globalOutputRow + row;
            const int globalCol = kTile + col;
            *reinterpret_cast<int4*>(&xShared[row * glu_config::kSharedMemStride + col]) =
                *reinterpret_cast<const int4*>(&x[globalRow * k + globalCol]);
        }

#pragma unroll
        for (int iter = 0; iter < glu_config::kBTileNumLoadIters; ++iter) {
            const int row = warpId * glu_config::kBTileNumColsPerWarp +
                            (laneId / glu_config::kBTileNumThreadsPerCol) +
                            iter * glu_config::kBTileNumColsPerWarpPerIter;
            const int col = (laneId % glu_config::kBTileNumThreadsPerCol) * glu_config::kNumElementsPerLoad;
            const int globalRow = globalOutputCol + row;
            const int globalCol = kTile + col;
            *reinterpret_cast<int4*>(&weight1Shared[row * glu_config::kSharedMemStride + col]) =
                *reinterpret_cast<const int4*>(&weight[globalRow * k + globalCol]);
            *reinterpret_cast<int4*>(&weight2Shared[row * glu_config::kSharedMemStride + col]) =
                *reinterpret_cast<const int4*>(&weight[(globalRow + intermediateSize) * k + globalCol]);
        }
        __syncthreads();

#pragma unroll
        for (int kFrag = 0; kFrag < glu_config::kKFragmentsPerIter; ++kFrag) {
            wmma::fragment<wmma::matrix_a, glu_config::kFragmentSize, glu_config::kFragmentSize, glu_config::kFragmentSize, Bf16, wmma::row_major>
                aFrag[glu_config::kWarpNumFragmentsM];
            wmma::fragment<wmma::matrix_b, glu_config::kFragmentSize, glu_config::kFragmentSize, glu_config::kFragmentSize, Bf16, wmma::col_major>
                bFrag[2][glu_config::kWarpNumFragmentsN];
#pragma unroll
            for (int i = 0; i < glu_config::kWarpNumFragmentsM; ++i) {
                const int row = warpId / glu_config::kBlockNumWarpsN * glu_config::kWarpM + i * glu_config::kFragmentSize;
                const int col = kFrag * glu_config::kFragmentSize;
                wmma::load_matrix_sync(aFrag[i], &xShared[row * glu_config::kSharedMemStride + col], glu_config::kSharedMemStride);
            }
#pragma unroll
            for (int j = 0; j < glu_config::kWarpNumFragmentsN; ++j) {
                const int row = warpId % glu_config::kBlockNumWarpsN * glu_config::kWarpN + j * glu_config::kFragmentSize;
                const int col = kFrag * glu_config::kFragmentSize;
                wmma::load_matrix_sync(bFrag[0][j], &weight1Shared[row * glu_config::kSharedMemStride + col], glu_config::kSharedMemStride);
                wmma::load_matrix_sync(bFrag[1][j], &weight2Shared[row * glu_config::kSharedMemStride + col], glu_config::kSharedMemStride);
            }

#pragma unroll
            for (int i = 0; i < glu_config::kWarpNumFragmentsM; ++i) {
#pragma unroll
                for (int j = 0; j < glu_config::kWarpNumFragmentsN; ++j) {
                    wmma::mma_sync(cFrag[0][i][j], aFrag[i], bFrag[0][j], cFrag[0][i][j]);
                    wmma::mma_sync(cFrag[1][i][j], aFrag[i], bFrag[1][j], cFrag[1][i][j]);
                }
            }
        }

        __syncthreads();
    }

    float* gemmResult = reinterpret_cast<float*>(&sharedMem[0][0]);
#pragma unroll
    for (int i = 0; i < glu_config::kWarpNumFragmentsM; ++i) {
#pragma unroll
        for (int j = 0; j < glu_config::kWarpNumFragmentsN; ++j) {
            const int row = (warpId / glu_config::kBlockNumWarpsN) * glu_config::kWarpM + i * glu_config::kFragmentSize;
            const int col = (warpId % glu_config::kBlockNumWarpsN) * glu_config::kWarpN + j * glu_config::kFragmentSize;
            wmma::store_matrix_sync(&gemmResult[row * glu_config::kSharedMemCStride + col],
                                    cFrag[0][i][j],
                                    glu_config::kSharedMemCStride,
                                    wmma::mem_row_major);
            wmma::store_matrix_sync(&gemmResult[row * glu_config::kSharedMemCStride + (col + BLOCK_N)],
                                    cFrag[1][i][j],
                                    glu_config::kSharedMemCStride,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = 0; i < glu_config::kCTileNumWriteIters; ++i) {
        const int row = warpId * glu_config::kCTileNumRowsPerWarp +
                        (laneId / glu_config::kCTileNumThreadsPerRow) +
                        i * glu_config::kCTileNumRowsPerWarpPerIter;
        const int col = (laneId % glu_config::kCTileNumThreadsPerRow) * glu_config::kS2GNumElementsPerLoad;
        const float4 result1 = *reinterpret_cast<const float4*>(&gemmResult[row * glu_config::kSharedMemCStride + col]);
        const float4 result2 = *reinterpret_cast<const float4*>(&gemmResult[row * glu_config::kSharedMemCStride + col + BLOCK_N]);
        const float4 finalResult = {
            result1.x * Activation(result2.x),
            result1.y * Activation(result2.y),
            result1.z * Activation(result2.z),
            result1.w * Activation(result2.w)
        };
        *reinterpret_cast<float4*>(&result[(globalOutputRow + row) * intermediateSize + (globalOutputCol + col)]) =
            finalResult;
    }
}

/**
 * @brief Launches the GLU BF16 kernel with configured grid/block sizes.
 * @param x Device pointer to input matrix X.
 * @param weight Device pointer to weight matrix.
 * @param result Device pointer to output matrix.
 * @param batchSize Number of rows in X.
 * @param k Size of the K dimension.
 * @param intermediateSize Size of the intermediate/output dimension.
 */
void LaunchGluBf16Kernel(const Bf16* x,
                         const Bf16* weight,
                         float* result,
                         int batchSize,
                         int k,
                         int intermediateSize) {
    if (batchSize <= 0 || k <= 0 || intermediateSize <= 0) {
        return;
    }
    if (batchSize % BLOCK_M != 0 || intermediateSize % BLOCK_N != 0 || k % A_TILE_K != 0) {
        return;
    }
    const dim3 numThreads(glu_config::kBlockNumThreads);
    const dim3 numBlocks(batchSize / BLOCK_M, intermediateSize / BLOCK_N);

    const size_t sharedMemSizeA = (glu_config::kATileM + glu_config::kBTileN + glu_config::kBTileN) *
                                  glu_config::kSharedMemStride * sizeof(Bf16);
    const size_t sharedMemSizeC = BLOCK_M * glu_config::kSharedMemCStride * sizeof(float);
    const size_t sharedMemSize = std::max(sharedMemSizeA, sharedMemSizeC);

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > static_cast<size_t>(deviceProp.sharedMemPerBlock)) {
        cudaFuncSetAttribute(GluBf16Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sharedMemSize));
    }

    GluBf16Kernel<<<numBlocks, numThreads, sharedMemSize>>>(x, weight, result, k, intermediateSize);
}
