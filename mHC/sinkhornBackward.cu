// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// Purpose: CUDA Sinkhorn-Knopp backward kernel demo with validation and timing.
// ==============================================================================
// Compilation command: nvcc -O3 --use_fast_math -gencode=arch=compute_120,code=sm_120 mHC/sinkhornBackward.cu mHC/sinkhornBackwardGolden.cu -o sinkhornBackward
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>

#include "sinkhornBackward.cuh"

#define CEIL_DIV(X, Y) (((X) + (Y)-1) / (Y))

constexpr int kMatrixSize = 4;
constexpr int kIterations = 20;
constexpr int kThreadsPerSlice = 4;
constexpr int kThreadsPerBlock = 128;
constexpr int kSlicesPerBlock = kThreadsPerBlock / kThreadsPerSlice;

static_assert(kThreadsPerBlock % kThreadsPerSlice == 0, "kThreadsPerBlock must be divisible by kThreadsPerSlice");

/**
 * @brief Reduces a value across a 4-lane subgroup using warp shuffles.
 * @param value Value contributed by the current lane.
 * @param subgroupMask Active mask for the 4-lane subgroup.
 * @return Sum of the values across the subgroup.
 */
__device__ __forceinline__ float ReduceSum4(float value, unsigned int subgroupMask) {
    value += __shfl_xor_sync(subgroupMask, value, 1);
    value += __shfl_xor_sync(subgroupMask, value, 2);
    return value;
}

/**
 * @brief CUDA kernel that backpropagates Sinkhorn-Knopp using recomputed row/column sums (4x4 specialized).
 *        Each thread processes one row (4 elements) of a 4x4 matrix.
 * @param input Pointer to X^{(0)} with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param gradOutput Pointer to gradient with respect to output, same shape as output.
 * @param gradX Output pointer for gradient with respect to X^{(0)}, same shape as output.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N (must equal kMatrixSize).
 * @param iterations Number of Sinkhorn-Knopp iterations.
 * @param epsilon Epsilon added to row/column sums for numerical stability.
 */
__global__ void mHCSinkhornBackwardKernel(
                                       const float* __restrict__ input,
                                       const float* __restrict__ gradOutput,
                                       float* __restrict__ gradX,
                                       int batchSize,
                                       int seqLength,
                                       int matrixSize,
                                       int iterations,
                                       float epsilon) {
    const int totalSlices = batchSize * seqLength;
    constexpr int kSliceSize = kMatrixSize * kMatrixSize;
    const int sliceInBlock = threadIdx.x / kThreadsPerSlice;
    const int laneInSlice = threadIdx.x % kThreadsPerSlice;
    const int globalSlice = blockIdx.x * kSlicesPerBlock + sliceInBlock;

    if (globalSlice >= totalSlices || matrixSize != kMatrixSize || iterations > kIterations) {
        return;
    }

    const int sliceOffset = globalSlice * kSliceSize;

    // Recompute forward pass to restore rowSum and colSum at each iteration.
    __shared__ float rowSumShared[kIterations][kSlicesPerBlock][4];
    __shared__ float4 colSumShared[kIterations][kSlicesPerBlock];

    const float4* input4 = reinterpret_cast<const float4*>(input + sliceOffset);
    float4 inRow = input4[laneInSlice];

    const float4* gradOutput4 = reinterpret_cast<const float4*>(gradOutput + sliceOffset);
    float4* gradX4 = reinterpret_cast<float4*>(gradX + sliceOffset);

    float4 gradRow = gradOutput4[laneInSlice];

    const unsigned int laneId = threadIdx.x & 31;
    const unsigned int subgroupBase = laneId & ~0x3u;
    const unsigned int subgroupMask = 0xFu << subgroupBase;

    for (int iter = 0; iter < iterations; ++iter) {
        // Row normalization
        const float rowSumVal = inRow.x + inRow.y + inRow.z + inRow.w + epsilon;
        rowSumShared[iter][sliceInBlock][laneInSlice] = rowSumVal;

        inRow.x /= rowSumVal;
        inRow.y /= rowSumVal;
        inRow.z /= rowSumVal;
        inRow.w /= rowSumVal;


        // Column normalization
        const float colSumX = ReduceSum4(inRow.x, subgroupMask) + epsilon;
        const float colSumY = ReduceSum4(inRow.y, subgroupMask) + epsilon;
        const float colSumZ = ReduceSum4(inRow.z, subgroupMask) + epsilon;
        const float colSumW = ReduceSum4(inRow.w, subgroupMask) + epsilon;

        if (laneInSlice == 0) {
            colSumShared[iter][sliceInBlock] = make_float4(colSumX, colSumY, colSumZ, colSumW);
        }
        __syncwarp(subgroupMask);

        inRow.x /= colSumX;
        inRow.y /= colSumY;
        inRow.z /= colSumZ;
        inRow.w /= colSumW;
    }


    for (int iter = iterations - 1; iter >= 0; --iter) {
        // Column normalization backward: compute per-column dot and rescale.
        const float rowSumVal = rowSumShared[iter][sliceInBlock][laneInSlice];
        const float4 colSumVec = colSumShared[iter][sliceInBlock];

        // Column 0
        const float colDotX = ReduceSum4(gradRow.x * inRow.x, subgroupMask);
        gradRow.x = (gradRow.x - colDotX) / colSumVec.x;
        inRow.x *= colSumVec.x;

        // Column 1
        const float colDotY = ReduceSum4(gradRow.y * inRow.y, subgroupMask);
        gradRow.y = (gradRow.y - colDotY) / colSumVec.y;
        inRow.y *= colSumVec.y;

        // Column 2
        const float colDotZ = ReduceSum4(gradRow.z * inRow.z, subgroupMask);
        gradRow.z = (gradRow.z - colDotZ) / colSumVec.z;
        inRow.z *= colSumVec.z;

        // Column 3
        const float colDotW = ReduceSum4(gradRow.w * inRow.w, subgroupMask);
        gradRow.w = (gradRow.w - colDotW) / colSumVec.w;
        inRow.w *= colSumVec.w;

        // Row normalization backward: compute per-row dot and rescale.
        const float rowDot = gradRow.x * inRow.x + gradRow.y * inRow.y + gradRow.z * inRow.z + gradRow.w * inRow.w;

        gradRow.x = (gradRow.x - rowDot) / rowSumVal;
        gradRow.y = (gradRow.y - rowDot) / rowSumVal;
        gradRow.z = (gradRow.z - rowDot) / rowSumVal;
        gradRow.w = (gradRow.w - rowDot) / rowSumVal;

        inRow.x *= rowSumVal;
        inRow.y *= rowSumVal;
        inRow.z *= rowSumVal;
        inRow.w *= rowSumVal;
    }

    gradX4[laneInSlice] = gradRow;
}

/**
 * @brief Computes a flat index for a 4D tensor in row-major order.
 * @param b Batch index in [0, batchSize).
 * @param s Sequence index in [0, seqLength).
 * @param i Row index in [0, matrixSize).
 * @param j Column index in [0, matrixSize).
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @return Flat index into the contiguous buffer.
 */
size_t flatIndex(int b, int s, int i, int j, int seqLength, int matrixSize) {
    return static_cast<size_t>(((b * seqLength + s) * matrixSize + i) * matrixSize + j);
}

/**
 * @brief Checks a CUDA API call result and aborts on error.
 * @param result CUDA error code returned by a runtime API call.
 * @param message Context message describing the failed operation.
 */
void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << message << " (" << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes the root mean squared error (RMSE) between two arrays.
 * @param reference Pointer to reference data.
 * @param candidate Pointer to candidate data.
 * @param elementCount Number of elements to compare.
 * @return RMSE across all elements.
 */
float computeRmse(const float* reference, const float* candidate, size_t elementCount) {
    float sumSquared = 0.0;
    for (size_t idx = 0; idx < elementCount; ++idx) {
        float diff = reference[idx] - candidate[idx];
        sumSquared += diff * diff;
    }
    float meanSquared = sumSquared / elementCount;
    return std::sqrt(meanSquared);
}

/**
 * @brief Computes effective memory bandwidth in GB/s.
 * @param bytesProcessed Total bytes processed per kernel launch.
 * @param elapsedMs Elapsed time in milliseconds per kernel launch.
 * @return Effective bandwidth in GB/s.
 */
double computeEffectiveBandwidthGb(double bytesProcessed, double elapsedMs) {
    const double seconds = elapsedMs * 1.0e-3;
    if (seconds <= 0.0) {
        return 0.0;
    }
    return (bytesProcessed / seconds) / 1.0e9;
}

/**
 * @brief Runs the Sinkhorn forward pass and stores row/column sums per iteration.
 * @param input Pointer to the input matrix batch.
 * @param output Pointer to the output buffer (overwritten in-place).
 * @param rowSum Pointer to row sums stored as (iterations, batchSize, seqLength, matrixSize).
 * @param colSum Pointer to column sums stored as (iterations, batchSize, seqLength, matrixSize).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param epsilon Epsilon added to row/column sums for numerical stability.
 * @param iterations Number of Sinkhorn-Knopp iterations.
 */
void mHCSinkhornForwardGolden(const float* input,
                                 float* output,
                                 float* rowSum,
                                 float* colSum,
                                 int batchSize,
                                 int seqLength,
                                 int matrixSize,
                                 float epsilon,
                                 int iterations) {
    std::copy_n(input, batchSize * seqLength * matrixSize * matrixSize, output);
    for (int b = 0; b < batchSize; ++b) {
        for (int s = 0; s < seqLength; ++s) {
            for (int iter = 0; iter < iterations; ++iter) {
                // Row normalization
                for (int i = 0; i < matrixSize; ++i) {
                    float rowSumVal = 0.0f;
                    for (int j = 0; j < matrixSize; ++j) {
                        rowSumVal += output[flatIndex(b, s, i, j, seqLength, matrixSize)];
                    }
                    rowSumVal += epsilon;
                    rowSum[iter * batchSize * seqLength * matrixSize + b * seqLength * matrixSize + s * matrixSize + i] = rowSumVal;
                    for (int j = 0; j < matrixSize; ++j) {
                        output[flatIndex(b, s, i, j, seqLength, matrixSize)] /= rowSumVal;
                    }
                }

                // Column normalization
                for (int j = 0; j < matrixSize; ++j) {
                    float colSumVal = 0.0f;
                    for (int i = 0; i < matrixSize; ++i) {
                        colSumVal += output[flatIndex(b, s, i, j, seqLength, matrixSize)];
                    }
                    colSumVal += epsilon;
                    colSum[iter * batchSize * seqLength * matrixSize + b * seqLength * matrixSize + s * matrixSize + j] = colSumVal;
                    for (int i = 0; i < matrixSize; ++i) {
                        output[flatIndex(b, s, i, j, seqLength, matrixSize)] /= colSumVal;
                    }
                }
            }
        }
    }
}

/**
 * @brief Entry point that builds sample tensors and runs Sinkhorn backward kernels.
 * @return Exit code (0 on success).
 */
int main() {
    const int batchSize = 2;
    const int seqLength = 128 * 1024;
    constexpr int matrixSize = kMatrixSize;
    const int iterations = kIterations;
    const int warmupRuns = 20;
    const int timedRuns = 20;

    static_assert(iterations % 2 == 0, "iterations must be even");
    static_assert(kMatrixSize == 4, "This implementation only supports matrixSize == 4");

    const size_t totalSize = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength) *
                            static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);

    float* sinkhornInput = new float[totalSize];
    float* sinkhornOutput = new float[totalSize];
    float* gradOutput = new float[totalSize];
    float* gradXGolden = new float[totalSize];
    float* gradXCuda = new float[totalSize];
    float* rowSum = new float[iterations * batchSize * seqLength * matrixSize];
    float* colSum = new float[iterations * batchSize * seqLength * matrixSize];

    // Initialize sinkhornInput with small positive values to seed Sinkhorn.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> inputDist(-1e-3, 1e-3);
    std::uniform_real_distribution<float> gradDist(-100.0f, 100.0f);
    for (size_t idx = 0; idx < totalSize; ++idx) {
        sinkhornInput[idx] = expf(inputDist(rng)) + 1e-3f;
    }

    float epsilon = 1.0e-6f;
    mHCSinkhornForwardGolden(sinkhornInput,
                              sinkhornOutput,
                              rowSum,
                              colSum,
                              batchSize,
                              seqLength,
                              matrixSize,
                              epsilon,
                              iterations);


    for (size_t idx = 0; idx < totalSize; ++idx) {
        gradOutput[idx] = gradDist(rng);
    }

    mHCSinkhornBackwardGolden(sinkhornOutput,
                              gradOutput,
                              gradXGolden,
                              rowSum,
                              colSum,
                              batchSize,
                              seqLength,
                              matrixSize,
                              iterations);
    
    // Allocate device memory and copy data
    float* sinkhornInputDevice;
    float* gradOutputDevice;
    float* gradXDevice;
    checkCuda(cudaMalloc(&sinkhornInputDevice, totalSize * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMalloc(&gradOutputDevice, totalSize * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMalloc(&gradXDevice, totalSize * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMemcpy(sinkhornInputDevice, sinkhornInput, totalSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    checkCuda(cudaMemcpy(gradOutputDevice, gradOutput, totalSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

    const int totalSlices = batchSize * seqLength;
    const int totalThreads = totalSlices * kThreadsPerSlice;
    const int threadsPerBlock = kThreadsPerBlock;
    const dim3 blockDim(threadsPerBlock, 1, 1);
    const dim3 gridDim(CEIL_DIV(totalThreads, threadsPerBlock), 1, 1);

    for (int run = 0; run < warmupRuns; ++run) {
        mHCSinkhornBackwardKernel<<<gridDim, blockDim>>>(sinkhornInputDevice,
                                                         gradOutputDevice,
                                                         gradXDevice,
                                                         batchSize,
                                                         seqLength,
                                                         matrixSize,
                                                         iterations,
                                                         epsilon);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate");

    checkCuda(cudaEventRecord(startEvent), "cudaEventRecord (start)");
    for (int run = 0; run < timedRuns; ++run) {
        mHCSinkhornBackwardKernel<<<gridDim, blockDim>>>(sinkhornInputDevice,
                                                         gradOutputDevice,
                                                         gradXDevice,
                                                         batchSize,
                                                         seqLength,
                                                         matrixSize,
                                                         iterations,
                                                         epsilon);
    }
    checkCuda(cudaEventRecord(stopEvent), "cudaEventRecord (stop)");
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");

    checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy");
    checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy");

    const double bytesProcessed = batchSize * seqLength * matrixSize * matrixSize * 3.0 * sizeof(float);
    const double avgElapsedMs = static_cast<double>(elapsedMs) / static_cast<double>(timedRuns);
    const double effectiveBandwidthGb = computeEffectiveBandwidthGb(bytesProcessed, avgElapsedMs);
    std::cout << "Avg elapsed time (ms): " << avgElapsedMs << '\n';
    std::cout << "Effective bandwidth (GB/s): " << effectiveBandwidthGb << '\n';

    checkCuda(cudaMemcpy(gradOutputDevice, gradOutput, totalSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

    checkCuda(cudaMemcpy(sinkhornInputDevice, sinkhornInput, totalSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    mHCSinkhornBackwardKernel<<<gridDim, blockDim>>>(sinkhornInputDevice,
                                                     gradOutputDevice,
                                                     gradXDevice,
                                                     batchSize,
                                                     seqLength,
                                                     matrixSize,
                                                     iterations,
                                                     epsilon);
    checkCuda(cudaGetLastError(), "mHCSinkhornBackwardKernel launch");

    checkCuda(cudaMemcpy(gradXCuda, gradXDevice, totalSize * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");

    const float rmse = computeRmse(gradXGolden, gradXCuda, totalSize);
    std::cout << "RMSE (gradX): " << rmse << '\n';

    delete[] sinkhornInput;
    delete[] sinkhornOutput;
    delete[] gradOutput;
    delete[] gradXGolden;
    delete[] gradXCuda;
    delete[] rowSum;
    delete[] colSum;

    checkCuda(cudaFree(sinkhornInputDevice), "cudaFree");
    checkCuda(cudaFree(gradOutputDevice), "cudaFree");
    checkCuda(cudaFree(gradXDevice), "cudaFree");

    return 0;
}
