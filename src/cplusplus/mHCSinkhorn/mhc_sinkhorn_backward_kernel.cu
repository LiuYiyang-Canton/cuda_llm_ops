// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for mHCSinkhorn backward pass.
// ==============================================================================
#include "mHCSinkhorn/mhc_sinkhorn_backward_kernel.cuh"

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
 * @brief Reduces a float2 across a 2-lane subgroup using packed 64-bit shuffles.
 * @param value Pair of values contributed by the current lane.
 * @return Sum of the pairs across the subgroup.
 */
__device__ __forceinline__ float2 ReduceSum2PackedFloat2(float2 value) {
    constexpr unsigned int kFullMask = 0xFFFFFFFFu;
    unsigned long long packed =
        (static_cast<unsigned long long>(__float_as_uint(value.y)) << 32) |
        static_cast<unsigned long long>(__float_as_uint(value.x));
    unsigned long long shuffled = __shfl_xor_sync(kFullMask, packed, 1, 2);
    float2 other;
    other.x = __uint_as_float(static_cast<unsigned int>(shuffled & 0xffffffffu));
    other.y = __uint_as_float(static_cast<unsigned int>(shuffled >> 32));
    value.x += other.x;
    value.y += other.y;
    return value;
}

/**
 * @brief CUDA kernel that backpropagates Sinkhorn-Knopp using recomputed row/column sums (4x4 specialized).
 *        Each thread processes two rows (8 elements) of a 4x4 matrix.
 * @param input Pointer to X^{(0)} with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param gradOutput Pointer to gradient with respect to output, same shape as output.
 * @param gradX Output pointer for gradient with respect to X^{(0)}, same shape as output.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param epsilon Epsilon added to row/column sums for numerical stability.
 */
template<int kMatrixSize, int kIterations>
__global__ void MhcSinkhornBackwardKernel(
                                       const float* __restrict__ input,
                                       const float* __restrict__ gradOutput,
                                       float* __restrict__ gradX,
                                       int batchSize,
                                       int seqLength,
                                       float epsilon) {
    const int totalSlices = batchSize * seqLength;
    constexpr int kSliceSize = kMatrixSize * kMatrixSize;
    const int sliceInBlock = threadIdx.x / kMhcSinkhornThreadsPerSlice;
    const int laneInSlice = threadIdx.x % kMhcSinkhornThreadsPerSlice;
    const int globalSlice = blockIdx.x * kSlicesPerBlock + sliceInBlock;

    if (globalSlice >= totalSlices) {
        return;
    }

    const int sliceOffset = globalSlice * kSliceSize;

    // Recompute forward pass to restore rowSum and colSum at each iteration.
    __shared__ float4 colSumShared[kIterations][kSlicesPerBlock];
    float2 rowSumLocal[kIterations];

    const float4* input4 = reinterpret_cast<const float4*>(input + sliceOffset);
    const float4* gradOutput4 = reinterpret_cast<const float4*>(gradOutput + sliceOffset);
    float4* gradX4 = reinterpret_cast<float4*>(gradX + sliceOffset);

    const int rowBase = laneInSlice * 2;
    float4 inRow0 = input4[rowBase];
    float4 inRow1 = input4[rowBase + 1];
    float4 gradRow0 = gradOutput4[rowBase];
    float4 gradRow1 = gradOutput4[rowBase + 1];

    #pragma unroll
    for (int iter = 0; iter < kIterations; ++iter) {
        // Row normalization
        const float rowSum0 = inRow0.x + inRow0.y + inRow0.z + inRow0.w + epsilon;
        const float rowSum1 = inRow1.x + inRow1.y + inRow1.z + inRow1.w + epsilon;
        rowSumLocal[iter] = make_float2(rowSum0, rowSum1);

        const float invRowSum0 = 1.0f / rowSum0;
        const float invRowSum1 = 1.0f / rowSum1;
        inRow0.x *= invRowSum0;
        inRow0.y *= invRowSum0;
        inRow0.z *= invRowSum0;
        inRow0.w *= invRowSum0;
        inRow1.x *= invRowSum1;
        inRow1.y *= invRowSum1;
        inRow1.z *= invRowSum1;
        inRow1.w *= invRowSum1;

        // Column normalization
        const float colPartial0 = inRow0.x + inRow1.x;
        const float colPartial1 = inRow0.y + inRow1.y;
        const float colPartial2 = inRow0.z + inRow1.z;
        const float colPartial3 = inRow0.w + inRow1.w;
        const float2 colSum01 = ReduceSum2PackedFloat2(make_float2(colPartial0, colPartial1));
        const float2 colSum23 = ReduceSum2PackedFloat2(make_float2(colPartial2, colPartial3));
        const float colSumX = colSum01.x + epsilon;
        const float colSumY = colSum01.y + epsilon;
        const float colSumZ = colSum23.x + epsilon;
        const float colSumW = colSum23.y + epsilon;

        if (laneInSlice == 0) {
            colSumShared[iter][sliceInBlock] = make_float4(colSumX, colSumY, colSumZ, colSumW);
        }

        const float invColSumX = 1.0f / colSumX;
        const float invColSumY = 1.0f / colSumY;
        const float invColSumZ = 1.0f / colSumZ;
        const float invColSumW = 1.0f / colSumW;
        inRow0.x *= invColSumX;
        inRow0.y *= invColSumY;
        inRow0.z *= invColSumZ;
        inRow0.w *= invColSumW;
        inRow1.x *= invColSumX;
        inRow1.y *= invColSumY;
        inRow1.z *= invColSumZ;
        inRow1.w *= invColSumW;
    }
    __syncwarp();

    #pragma unroll
    for (int rev = 0; rev < kIterations; ++rev) {
        const int iter = kIterations - 1 - rev;
        // Column normalization backward: compute per-column dot and rescale.
        const float2 rowSumPair = rowSumLocal[iter];
        const float rowSum0 = rowSumPair.x;
        const float rowSum1 = rowSumPair.y;
        const float4 colSumVec = colSumShared[iter][sliceInBlock];

        const float colDotPartial0 = gradRow0.x * inRow0.x + gradRow1.x * inRow1.x;
        const float colDotPartial1 = gradRow0.y * inRow0.y + gradRow1.y * inRow1.y;
        const float colDotPartial2 = gradRow0.z * inRow0.z + gradRow1.z * inRow1.z;
        const float colDotPartial3 = gradRow0.w * inRow0.w + gradRow1.w * inRow1.w;
        const float2 colDot01 = ReduceSum2PackedFloat2(make_float2(colDotPartial0, colDotPartial1));
        const float2 colDot23 = ReduceSum2PackedFloat2(make_float2(colDotPartial2, colDotPartial3));

        const float invColSumX = 1.0f / colSumVec.x;
        const float invColSumY = 1.0f / colSumVec.y;
        const float invColSumZ = 1.0f / colSumVec.z;
        const float invColSumW = 1.0f / colSumVec.w;

        gradRow0.x = (gradRow0.x - colDot01.x) * invColSumX;
        gradRow1.x = (gradRow1.x - colDot01.x) * invColSumX;
        gradRow0.y = (gradRow0.y - colDot01.y) * invColSumY;
        gradRow1.y = (gradRow1.y - colDot01.y) * invColSumY;
        gradRow0.z = (gradRow0.z - colDot23.x) * invColSumZ;
        gradRow1.z = (gradRow1.z - colDot23.x) * invColSumZ;
        gradRow0.w = (gradRow0.w - colDot23.y) * invColSumW;
        gradRow1.w = (gradRow1.w - colDot23.y) * invColSumW;

        inRow0.x *= colSumVec.x;
        inRow0.y *= colSumVec.y;
        inRow0.z *= colSumVec.z;
        inRow0.w *= colSumVec.w;
        inRow1.x *= colSumVec.x;
        inRow1.y *= colSumVec.y;
        inRow1.z *= colSumVec.z;
        inRow1.w *= colSumVec.w;

        // Row normalization backward: compute per-row dot and rescale.
        const float rowDot0 =
            gradRow0.x * inRow0.x + gradRow0.y * inRow0.y + gradRow0.z * inRow0.z + gradRow0.w * inRow0.w;
        const float rowDot1 =
            gradRow1.x * inRow1.x + gradRow1.y * inRow1.y + gradRow1.z * inRow1.z + gradRow1.w * inRow1.w;

        const float invRowSum0 = 1.0f / rowSum0;
        const float invRowSum1 = 1.0f / rowSum1;
        gradRow0.x = (gradRow0.x - rowDot0) * invRowSum0;
        gradRow0.y = (gradRow0.y - rowDot0) * invRowSum0;
        gradRow0.z = (gradRow0.z - rowDot0) * invRowSum0;
        gradRow0.w = (gradRow0.w - rowDot0) * invRowSum0;
        gradRow1.x = (gradRow1.x - rowDot1) * invRowSum1;
        gradRow1.y = (gradRow1.y - rowDot1) * invRowSum1;
        gradRow1.z = (gradRow1.z - rowDot1) * invRowSum1;
        gradRow1.w = (gradRow1.w - rowDot1) * invRowSum1;

        inRow0.x *= rowSum0;
        inRow0.y *= rowSum0;
        inRow0.z *= rowSum0;
        inRow0.w *= rowSum0;
        inRow1.x *= rowSum1;
        inRow1.y *= rowSum1;
        inRow1.z *= rowSum1;
        inRow1.w *= rowSum1;
    }

    gradX4[rowBase] = gradRow0;
    gradX4[rowBase + 1] = gradRow1;
}

}  // namespace

/**
 * @brief Launches the Sinkhorn-Knopp backward kernel.
 * @param input Pointer to device input tensor X^{(0)}.
 * @param gradOutput Pointer to device gradient w.r.t. output.
 * @param gradX Pointer to device gradient w.r.t. input.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param epsilon Small constant added to row/column sums.
 */
template<int matrixSize, int iterations>
void LaunchMhcSinkhornBackwardKernel(const float* input,
                                     const float* gradOutput,
                                     float* gradX,
                                     int batchSize,
                                     int seqLength,
                                     float epsilon) {
    constexpr int kThreadsPerBlock = 256;
    if (batchSize <= 0 || seqLength <= 0 || iterations <= 0) {
        return;
    }
    if (matrixSize != 4) {
        return;
    }

    const int totalSlices = batchSize * seqLength;
    const int totalThreads = totalSlices * kMhcSinkhornThreadsPerSlice;
    const dim3 blockDim(kThreadsPerBlock, 1, 1);
    const dim3 gridDim(CeilDiv(totalThreads, kThreadsPerBlock), 1, 1);

    MhcSinkhornBackwardKernel<matrixSize, iterations><<<gridDim, blockDim>>>(
        input, gradOutput, gradX, batchSize, seqLength, epsilon);
}

template void LaunchMhcSinkhornBackwardKernel<4, 20>(const float* input, const float* gradOutput, float* gradX,
                                          int batchSize,
                                          int seqLength,
                                          float epsilon);