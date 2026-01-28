// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for mHCSinkhorn.
// ==============================================================================
#include "mHCSinkhorn/mhc_sinkhorn_kernel.cuh"

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
 * @brief CUDA kernel that applies Sinkhorn-Knopp iterations per (B, S) slice.
 * @param input Pointer to input tensor H with shape (batchSize, seqLength, matrixSize, matrixSize),
 *              stored in row-major order. Also used for output.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param iterations Number of Sinkhorn iterations to run.
 * @param epsilon Small constant to avoid division by zero.
 */
template<int matrixSize>
__global__ void MhcSinkhornKernel(float* input,
                                 int batchSize,
                                 int seqLength,
                                 int iterations,
                                 float epsilon) {
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalSlices = batchSize * seqLength;
    const int sliceSize = matrixSize * matrixSize;

    if (globalThreadId >= totalSlices) {
        return;
    }

    input += globalThreadId * sliceSize;

    // temporary buffer for one row/column for this thread
    float temp[matrixSize];
    for (int iter = 0; iter < iterations; ++iter) {
        // normalize rows
        for (int i = 0; i < matrixSize; ++i) {
            float rowSum = epsilon;
            int idx = i * matrixSize;
            for (int j = 0; j < matrixSize; j += 4) {
                (float4&)temp[j] = (float4&)input[idx + j];
                rowSum += temp[j] + temp[j + 1] + temp[j + 2] + temp[j + 3];
            }
            for (int j = 0; j < matrixSize; j += 4) {
                temp[j] /= rowSum;
                temp[j + 1] /= rowSum;
                temp[j + 2] /= rowSum;
                temp[j + 3] /= rowSum;
                (float4&)input[idx + j] = (float4&)temp[j];
            }
        }

        // normalize columns
        for (int j = 0; j < matrixSize; ++j) {
            float colSum = epsilon;
            for (int i = 0; i < matrixSize; i += 4) {
                int idx0 = (i + 0) * matrixSize + j;
                int idx1 = (i + 1) * matrixSize + j;
                int idx2 = (i + 2) * matrixSize + j;
                int idx3 = (i + 3) * matrixSize + j;
                colSum += input[idx0] + input[idx1] + input[idx2] + input[idx3];
                temp[i + 0] = input[idx0];
                temp[i + 1] = input[idx1];
                temp[i + 2] = input[idx2];
                temp[i + 3] = input[idx3];
            }
            for (int i = 0; i < matrixSize; i += 4) {
                temp[i + 0] /= colSum;
                temp[i + 1] /= colSum;
                temp[i + 2] /= colSum;
                temp[i + 3] /= colSum;
                int idx0 = (i + 0) * matrixSize + j;
                int idx1 = (i + 1) * matrixSize + j;
                int idx2 = (i + 2) * matrixSize + j;
                int idx3 = (i + 3) * matrixSize + j;
                input[idx0] = temp[i + 0];
                input[idx1] = temp[i + 1];
                input[idx2] = temp[i + 2];
                input[idx3] = temp[i + 3];
            }
        }
    }
}

}  // namespace

/**
 * @brief Launches the Sinkhorn-Knopp forward kernel for (B, S, N, N) tensors.
 * @param input Pointer to device tensor H with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param iterations Number of Sinkhorn iterations to run.
 * @param epsilon Small constant to avoid division by zero.
 */
template<int matrixSize>
void LaunchMhcSinkhornKernel(float* input,
                             int batchSize,
                             int seqLength,
                             int iterations,
                             float epsilon) {
    constexpr int kThreadsPerBlock = 256;
    if (batchSize <= 0 || seqLength <= 0 || matrixSize <= 0 || iterations <= 0) {
        return;
    }
    if ((matrixSize % 4) != 0) {
        return;
    }
    const int totalSlices = batchSize * seqLength;
    const int blocks = CeilDiv(totalSlices, kThreadsPerBlock);
    const dim3 gridDim(blocks);
    const dim3 blockDim(kThreadsPerBlock);
    MhcSinkhornKernel<matrixSize><<<gridDim, blockDim>>>(input,
                                             batchSize,
                                             seqLength,
                                             iterations,
                                             epsilon);
}

template void LaunchMhcSinkhornKernel<4>(float* input,
                                          int batchSize,
                                          int seqLength,
                                          int iterations,
                                          float epsilon);
