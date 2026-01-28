// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for ReduceSum.
// ==============================================================================
#include "ReduceSum/reducesum_kernel.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace {

namespace cg = cooperative_groups;

constexpr int kWarpSize = 32;

/**
 * @brief Reduces rows in a row-major fp32 matrix using vectorized float4 loads.
 * @tparam RowsPerBlock Number of rows reduced per block.
 * @tparam ThreadsPerBlock Number of threads per block.
 * @param matrix Pointer to device input matrix (row-major).
 * @param rowSum Pointer to device output row sums.
 * @param cols Number of columns per row.
 */
template <int RowsPerBlock, int ThreadsPerBlock>
__global__ void ReduceSumFp32Kernel(const float* __restrict__ matrix,
                                    float* __restrict__ rowSum,
                                    int cols) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<kWarpSize>(block);

    constexpr int kWarpsPerBlock = ThreadsPerBlock / kWarpSize;
    static_assert(RowsPerBlock == 4, "Kernel assumes 4 rows per block for float4 vectorization");

    __shared__ float4 warpAccum[kWarpsPerBlock];

    const int thread = static_cast<int>(threadIdx.x);
    const int warpId = thread / kWarpSize;
    const int lane = thread % kWarpSize;
    const int rowBlock = static_cast<int>(blockIdx.x) * RowsPerBlock;
    const float* rowPtr = matrix + static_cast<long long>(rowBlock) * cols;

    float4 local = {0.f, 0.f, 0.f, 0.f};
    for (int col = thread * 4; col < cols; col += static_cast<int>(blockDim.x) * 4) {
        const float4 r0 = *reinterpret_cast<const float4*>(&rowPtr[col]);
        const float4 r1 = *reinterpret_cast<const float4*>(&rowPtr[cols + col]);
        const float4 r2 = *reinterpret_cast<const float4*>(&rowPtr[2 * cols + col]);
        const float4 r3 = *reinterpret_cast<const float4*>(&rowPtr[3 * cols + col]);
        local.x += r0.x + r0.y + r0.z + r0.w;
        local.y += r1.x + r1.y + r1.z + r1.w;
        local.z += r2.x + r2.y + r2.z + r2.w;
        local.w += r3.x + r3.y + r3.z + r3.w;
    }

    local.x = cg::reduce(warp, local.x, cg::plus<float>());
    local.y = cg::reduce(warp, local.y, cg::plus<float>());
    local.z = cg::reduce(warp, local.z, cg::plus<float>());
    local.w = cg::reduce(warp, local.w, cg::plus<float>());

    if (lane == 0) {
        warpAccum[warpId] = local;
    }
    block.sync();

    for (int offset = kWarpsPerBlock / 2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            warpAccum[thread].x += warpAccum[thread + offset].x;
            warpAccum[thread].y += warpAccum[thread + offset].y;
            warpAccum[thread].z += warpAccum[thread + offset].z;
            warpAccum[thread].w += warpAccum[thread + offset].w;
        }
        block.sync();
    }

    if (thread == 0) {
        *reinterpret_cast<float4*>(&rowSum[rowBlock]) = warpAccum[0];
    }
}

}  // namespace

/**
 * @brief Launches the fp32 reduce-sum kernel for a row-major matrix.
 * @param matrix Pointer to device input matrix (row-major).
 * @param rowSum Pointer to device output row-sum vector.
 * @param rows Number of rows in the matrix; must be divisible by kReduceSumRowsPerBlock.
 * @param cols Number of columns in the matrix; must be divisible by 4.
 */
void LaunchReduceSumFp32Kernel(const float* matrix, float* rowSum, int rows, int cols) {
    const dim3 blockDim(kReduceSumThreadsPerBlock);
    const dim3 gridDim(rows / kReduceSumRowsPerBlock);
    ReduceSumFp32Kernel<kReduceSumRowsPerBlock, kReduceSumThreadsPerBlock><<<gridDim, blockDim>>>(
        matrix, rowSum, cols);
}
