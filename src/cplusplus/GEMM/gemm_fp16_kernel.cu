// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for FP16 GEMM.
// ==============================================================================
#include "GEMM/gemm_fp16_kernel.cuh"

#include <cuda_runtime.h>
#include <mma.h>

namespace {

using Fp16 = half;
using namespace nvcuda;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockNumWarpsM = 2;
constexpr int kBlockNumWarpsN = 2;
constexpr int kBlockNumWarps = kBlockNumWarpsM * kBlockNumWarpsN;
constexpr int kWarpM = kBlockM / kBlockNumWarpsM;
constexpr int kWarpN = kBlockN / kBlockNumWarpsN;
constexpr int kWarpSize = 32;
constexpr int kFragmentSize = 16;
constexpr int kWarpNumFragmentsM = kWarpM / kFragmentSize;
constexpr int kWarpNumFragmentsN = kWarpN / kFragmentSize;
constexpr int kBlockNumThreads = kBlockNumWarpsM * kBlockNumWarpsN * kWarpSize;
constexpr int kKFragmentsPerIter = 4;
constexpr int kATileM = kBlockM;
constexpr int kATileK = kFragmentSize * kKFragmentsPerIter;
constexpr int kBTileN = kBlockN;
constexpr int kBTileK = kFragmentSize * kKFragmentsPerIter;
constexpr int kNumElementsPerLoad = sizeof(int4) / sizeof(Fp16);
constexpr int kATileThreadsPerRow = kATileK / kNumElementsPerLoad;
constexpr int kATileRowsPerWarp = kATileM / kBlockNumWarps;
constexpr int kATileRowsPerWarpPerIter = kWarpSize / kATileThreadsPerRow;
constexpr int kATileLoadIters = kATileRowsPerWarp / kATileRowsPerWarpPerIter;
constexpr int kBTileThreadsPerCol = kBTileK / kNumElementsPerLoad;
constexpr int kBTileColsPerWarp = kBTileN / kBlockNumWarps;
constexpr int kBTileColsPerWarpPerIter = kWarpSize / kBTileThreadsPerCol;
constexpr int kBTileLoadIters = kBTileColsPerWarp / kBTileColsPerWarpPerIter;
constexpr int kSharedMemPadding = 8;
constexpr int kSharedMemStride = kATileK + kSharedMemPadding;

/**
 * @brief CUDA kernel for fp16 GEMM with fp32 accumulation.
 * @param a Pointer to device matrix A (row-major) of shape [m, k].
 * @param b Pointer to device matrix B (column-major) of shape [k, n].
 * @param c Pointer to device output matrix C (row-major) of shape [m, n].
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A and rows in B.
 */
__global__ void GemmFp16Kernel(const Fp16* __restrict__ a,
                               const Fp16* __restrict__ b,
                               float* __restrict__ c,
                               int m,
                               int n,
                               int k) {
    const int thread = static_cast<int>(threadIdx.x);
    const int blockX = static_cast<int>(blockIdx.x);
    const int blockY = static_cast<int>(blockIdx.y);

    const int warpId = thread / kWarpSize;
    const int globalOutputRow = blockX * kBlockM;
    const int globalOutputCol = blockY * kBlockN;

    extern __shared__ Fp16 sharedMem[][kSharedMemStride];
    Fp16* aShared = &sharedMem[0][0];
    Fp16* bShared = &sharedMem[kATileM][0];

    wmma::fragment<wmma::accumulator, kFragmentSize, kFragmentSize, kFragmentSize, float>
        cFrag[kWarpNumFragmentsM][kWarpNumFragmentsN];

#pragma unroll
    for (int i = 0; i < kWarpNumFragmentsM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpNumFragmentsN; ++j) {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
        }
    }

#pragma unroll
    for (int kTile = 0; kTile < k; kTile += kATileK) {
#pragma unroll
        for (int iter = 0; iter < kATileLoadIters; ++iter) {
            const int row = warpId * kATileRowsPerWarp +
                            (thread % kWarpSize) / kATileThreadsPerRow +
                            iter * kATileRowsPerWarpPerIter;
            const int col = (thread % kATileThreadsPerRow) * kNumElementsPerLoad;
            const int globalRow = globalOutputRow + row;
            const int globalCol = kTile + col;
            *reinterpret_cast<int4*>(&aShared[row * kSharedMemStride + col]) =
                *reinterpret_cast<const int4*>(&a[globalRow * k + globalCol]);
        }

#pragma unroll
        for (int iter = 0; iter < kBTileLoadIters; ++iter) {
            const int row = warpId * kBTileColsPerWarp +
                            (thread % kWarpSize) / kBTileThreadsPerCol +
                            iter * kBTileColsPerWarpPerIter;
            const int col = (thread % kBTileThreadsPerCol) * kNumElementsPerLoad;
            const int globalRow = globalOutputCol + row;
            const int globalCol = kTile + col;
            *reinterpret_cast<int4*>(&bShared[row * kSharedMemStride + col]) =
                *reinterpret_cast<const int4*>(&b[globalRow * k + globalCol]);
        }

        __syncthreads();
        int mmaStage = 0;
        wmma::fragment<wmma::matrix_a, kFragmentSize, kFragmentSize, kFragmentSize, Fp16, wmma::row_major>
            aFrag[2][kWarpNumFragmentsM];
        wmma::fragment<wmma::matrix_b, kFragmentSize, kFragmentSize, kFragmentSize, Fp16, wmma::col_major>
            bFrag[2][kWarpNumFragmentsN];

#pragma unroll
        for (int i = 0; i < kWarpNumFragmentsM; ++i) {
            const int row = warpId / kBlockNumWarpsN * kWarpM + i * kFragmentSize;
            wmma::load_matrix_sync(aFrag[mmaStage][i], &aShared[row * kSharedMemStride], kSharedMemStride);
        }
#pragma unroll
        for (int j = 0; j < kWarpNumFragmentsN; ++j) {
            const int row = warpId % kBlockNumWarpsN * kWarpN + j * kFragmentSize;
            wmma::load_matrix_sync(bFrag[mmaStage][j], &bShared[row * kSharedMemStride], kSharedMemStride);
        }
        __syncthreads();

#pragma unroll
        for (int kFrag = 0; kFrag < kKFragmentsPerIter; ++kFrag) {
            mmaStage = 1 - mmaStage;
            const int kNextFrag = kFrag + 1;
            if (kNextFrag < kKFragmentsPerIter) {
#pragma unroll
                for (int i = 0; i < kWarpNumFragmentsM; ++i) {
                    const int row = warpId / kBlockNumWarpsN * kWarpM + i * kFragmentSize;
                    const int col = kNextFrag * kFragmentSize;
                    wmma::load_matrix_sync(aFrag[mmaStage][i],
                                           &aShared[row * kSharedMemStride + col],
                                           kSharedMemStride);
                }
#pragma unroll
                for (int j = 0; j < kWarpNumFragmentsN; ++j) {
                    const int row = warpId % kBlockNumWarpsN * kWarpN + j * kFragmentSize;
                    const int col = kNextFrag * kFragmentSize;
                    wmma::load_matrix_sync(bFrag[mmaStage][j],
                                           &bShared[row * kSharedMemStride + col],
                                           kSharedMemStride);
                }
            }

#pragma unroll
            for (int i = 0; i < kWarpNumFragmentsM; ++i) {
#pragma unroll
                for (int j = 0; j < kWarpNumFragmentsN; ++j) {
                    wmma::mma_sync(cFrag[i][j], aFrag[1 - mmaStage][i], bFrag[1 - mmaStage][j], cFrag[i][j]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kWarpNumFragmentsM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpNumFragmentsN; ++j) {
            const int row = (warpId / kBlockNumWarpsN) * kWarpM + i * kFragmentSize;
            const int col = (warpId % kBlockNumWarpsN) * kWarpN + j * kFragmentSize;
            const int globalRow = globalOutputRow + row;
            const int globalCol = globalOutputCol + col;
            wmma::store_matrix_sync(&c[globalRow * n + globalCol],
                                    cFrag[i][j],
                                    n,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace

/**
 * @brief Launches the fp16 GEMM kernel with fp32 accumulation.
 * @param a Pointer to device matrix A (row-major) of shape [m, k].
 * @param b Pointer to device matrix B (column-major) of shape [k, n].
 * @param c Pointer to device output matrix C (row-major) of shape [m, n].
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A and rows in B.
 */
void LaunchGemmFp16Kernel(const half* a, const half* b, float* c, int m, int n, int k) {
    static_assert(kATileK == kBTileK, "A/B tile K dimensions must match");
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    if ((m % kBlockM) != 0 || (n % kBlockN) != 0) {
        return;
    }
    if ((k % kATileK) != 0) {
        return;
    }

    const dim3 blockDim(kBlockNumThreads);
    const dim3 gridDim(m / kBlockM, n / kBlockN);

    const size_t sharedMemSize = static_cast<size_t>(kATileM + kBTileN) *
                                 static_cast<size_t>(kSharedMemStride) * sizeof(Fp16);

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        cudaFuncSetAttribute(GemmFp16Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sharedMemSize));
    }

    GemmFp16Kernel<<<gridDim, blockDim, sharedMemSize>>>(a, b, c, m, n, k);
}
