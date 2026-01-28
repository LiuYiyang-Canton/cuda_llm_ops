// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for elementwise add.
// ==============================================================================
#include "ElementwiseAdd/elementwiseadd_kernel.cuh"

namespace {

/**
 * @brief Returns ceil(numerator / denominator) for positive integers.
 * @param numerator Dividend value.
 * @param denominator Divisor value; must be > 0.
 * @return Rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Vectorized elementwise add kernel that processes four fp32 values per thread.
 * @param a Pointer to input matrix A in device memory.
 * @param b Pointer to input matrix B in device memory.
 * @param result Pointer to output matrix in device memory.
 * @param n Matrix dimension (square). Must be divisible by 4.
 */
__global__ void ElementwiseAddFp32Kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ result,
                                         int n) {
    const int row = static_cast<int>(blockIdx.y);
    const int rowOffset = row * n;
    const int workPerBlock = static_cast<int>(blockDim.x) * kElementwiseAddWorkPerThread;
    const int blockCol = static_cast<int>(blockIdx.x);
    const int colStart = blockCol * workPerBlock;
    const int tileEnd = (blockCol + 1) * workPerBlock;
    const int colEnd = tileEnd < n ? tileEnd : n;
    for (int col = colStart + static_cast<int>(threadIdx.x) * kElementwiseAddWorkPerThread;
         col < colEnd;
         col += static_cast<int>(blockDim.x) * kElementwiseAddWorkPerThread) {
        const float4 aValue = *reinterpret_cast<const float4*>(&a[rowOffset + col]);
        const float4 bValue = *reinterpret_cast<const float4*>(&b[rowOffset + col]);
        float4 sum = {aValue.x + bValue.x,
                      aValue.y + bValue.y,
                      aValue.z + bValue.z,
                      aValue.w + bValue.w};
        *reinterpret_cast<float4*>(&result[rowOffset + col]) = sum;
    }
}

}  // namespace

/**
 * @brief Launches the fp32 elementwise add kernel.
 * @param a Device pointer to input matrix A.
 * @param b Device pointer to input matrix B.
 * @param result Device pointer to output matrix.
 * @param n Matrix dimension (square). Must be divisible by 4.
 * @param threadsPerBlock Number of threads per block for the kernel launch.
 */
void LaunchElementwiseAddFp32Kernel(const float* a,
                                   const float* b,
                                   float* result,
                                   int n,
                                   int threadsPerBlock) {
    const int workPerBlock = threadsPerBlock * kElementwiseAddWorkPerThread;
    const dim3 gridDim(CeilDiv(n, workPerBlock), n);
    ElementwiseAddFp32Kernel<<<gridDim, threadsPerBlock>>>(a, b, result, n);
}
