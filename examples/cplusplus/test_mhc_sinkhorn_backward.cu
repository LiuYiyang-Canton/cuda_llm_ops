// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for mHCSinkhorn backward kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace {

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
size_t FlatIndex(int b, int s, int i, int j, int seqLength, int matrixSize) {
    return static_cast<size_t>(((b * seqLength + s) * matrixSize + i) * matrixSize + j);
}

/**
 * @brief Checks a CUDA API call result and aborts on error.
 * @param result CUDA error code returned by a runtime API call.
 * @param message Context message describing the failed operation.
 */
void CheckCuda(cudaError_t result, const char* message) {
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
float ComputeRmse(const float* reference, const float* candidate, size_t elementCount) {
    float sumSquared = 0.0f;
    for (size_t idx = 0; idx < elementCount; ++idx) {
        const float diff = reference[idx] - candidate[idx];
        sumSquared += diff * diff;
    }
    const float meanSquared = sumSquared / static_cast<float>(elementCount);
    return std::sqrt(meanSquared);
}

/**
 * @brief Computes effective memory bandwidth in GB/s.
 * @param bytesProcessed Total bytes processed per kernel launch.
 * @param elapsedMs Elapsed time in milliseconds per kernel launch.
 * @return Effective bandwidth in GB/s.
 */
double ComputeEffectiveBandwidthGb(double bytesProcessed, double elapsedMs) {
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
void MhcSinkhornForwardGolden(const float* input,
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
                for (int i = 0; i < matrixSize; ++i) {
                    float rowSumVal = 0.0f;
                    for (int j = 0; j < matrixSize; ++j) {
                        rowSumVal += output[FlatIndex(b, s, i, j, seqLength, matrixSize)];
                    }
                    rowSumVal += epsilon;
                    rowSum[iter * batchSize * seqLength * matrixSize +
                           b * seqLength * matrixSize + s * matrixSize + i] = rowSumVal;
                    for (int j = 0; j < matrixSize; ++j) {
                        output[FlatIndex(b, s, i, j, seqLength, matrixSize)] /= rowSumVal;
                    }
                }

                for (int j = 0; j < matrixSize; ++j) {
                    float colSumVal = 0.0f;
                    for (int i = 0; i < matrixSize; ++i) {
                        colSumVal += output[FlatIndex(b, s, i, j, seqLength, matrixSize)];
                    }
                    colSumVal += epsilon;
                    colSum[iter * batchSize * seqLength * matrixSize +
                           b * seqLength * matrixSize + s * matrixSize + j] = colSumVal;
                    for (int i = 0; i < matrixSize; ++i) {
                        output[FlatIndex(b, s, i, j, seqLength, matrixSize)] /= colSumVal;
                    }
                }
            }
        }
    }
}

/**
 * @brief Backpropagates column normalization in-place and reconstructs X^{(k+1/2)}.
 * @param xNextInOut Pointer to X^{(k+1)} input, overwritten with X^{(k+1/2)}.
 * @param gradNextInOut Pointer to gradient wrt X^{(k+1)}, overwritten with gradient wrt X^{(k+1/2)}.
 * @param colSum Pointer to column sums c_j^{(k)} (length matrixSize), including epsilon.
 * @param matrixSize Matrix dimension N.
 */
void NormalizeColumnsBackward(float* xNextInOut, float* gradNextInOut, const float* colSum, int matrixSize) {
    for (int j = 0; j < matrixSize; ++j) {
        float sumProd = 0.0f;
        for (int i = 0; i < matrixSize; ++i) {
            sumProd += gradNextInOut[i * matrixSize + j] * xNextInOut[i * matrixSize + j];
        }
        const float denom = colSum[j];
        for (int i = 0; i < matrixSize; ++i) {
            xNextInOut[i * matrixSize + j] = xNextInOut[i * matrixSize + j] * denom;
            gradNextInOut[i * matrixSize + j] = (gradNextInOut[i * matrixSize + j] - sumProd) / denom;
        }
    }
}

/**
 * @brief Backpropagates row normalization in-place and reconstructs X^{(k)}.
 * @param xNextInOut Pointer to X^{(k+1/2)} input, overwritten with X^{(k)}.
 * @param gradNextInOut Pointer to gradient wrt X^{(k+1/2)}, overwritten with gradient wrt X^{(k)}.
 * @param rowSum Pointer to row sums r_i^{(k)} (length matrixSize), including epsilon.
 * @param matrixSize Matrix dimension N.
 */
void NormalizeRowsBackward(float* xNextInOut, float* gradNextInOut, const float* rowSum, int matrixSize) {
    for (int i = 0; i < matrixSize; ++i) {
        float sumProd = 0.0f;
        for (int j = 0; j < matrixSize; ++j) {
            sumProd += gradNextInOut[i * matrixSize + j] * xNextInOut[i * matrixSize + j];
        }

        const float denom = rowSum[i];
        for (int j = 0; j < matrixSize; ++j) {
            xNextInOut[i * matrixSize + j] = xNextInOut[i * matrixSize + j] * denom;
            gradNextInOut[i * matrixSize + j] = (gradNextInOut[i * matrixSize + j] - sumProd) / denom;
        }
    }
}

/**
 * @brief Computes Sinkhorn-Knopp backward using stored row/column sums per iteration.
 * @param output Pointer to X^{(iterations)} with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param gradOutput Pointer to gradient with respect to output, same shape as output.
 * @param gradX Output pointer for gradient with respect to X^{(0)}, same shape as output.
 * @param rowSum Pointer to stored row sums with shape (batchSize, seqLength, iterations, matrixSize).
 * @param colSum Pointer to stored column sums with shape (batchSize, seqLength, iterations, matrixSize).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param iterations Number of Sinkhorn-Knopp iterations.
 */
void MhcSinkhornBackwardGolden(float* output,
                               const float* gradOutput,
                               float* gradX,
                               const float* rowSum,
                               const float* colSum,
                               int batchSize,
                               int seqLength,
                               int matrixSize,
                               int iterations) {
    for (int b = 0; b < batchSize; ++b) {
        for (int s = 0; s < seqLength; ++s) {
            float* xNextInOut = output + ((b * seqLength + s) * matrixSize * matrixSize);
            float* gradXOut = gradX + ((b * seqLength + s) * matrixSize * matrixSize);

            std::copy_n(gradOutput + ((b * seqLength + s) * matrixSize * matrixSize),
                        matrixSize * matrixSize,
                        gradXOut);

            for (int iter = iterations - 1; iter >= 0; --iter) {
                const float* currentColSum = colSum +
                                             (iter * batchSize * seqLength * matrixSize +
                                              b * seqLength * matrixSize + s * matrixSize);
                const float* currentRowSum = rowSum +
                                             (iter * batchSize * seqLength * matrixSize +
                                              b * seqLength * matrixSize + s * matrixSize);

                NormalizeColumnsBackward(xNextInOut, gradXOut, currentColSum, matrixSize);
                NormalizeRowsBackward(xNextInOut, gradXOut, currentRowSum, matrixSize);
            }
        }
    }
}

}  // namespace

/**
 * @brief Entry point that builds sample tensors and runs Sinkhorn backward kernels.
 * @return Exit code (0 on success).
 */
int main() {
    const int batchSize = 2;
    const int seqLength = 128 * 1024;
    constexpr int matrixSize = 4;
    const int iterations = 20;
    const int warmupRuns = 20;
    const int timedRuns = 20;

    static_assert(matrixSize == 4, "This implementation only supports matrixSize == 4");

    const size_t totalSize = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength) *
                             static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);

    std::vector<float> sinkhornInput(totalSize);
    std::vector<float> sinkhornOutput(totalSize);
    std::vector<float> gradOutput(totalSize);
    std::vector<float> gradXGolden(totalSize);
    std::vector<float> gradXCuda(totalSize);
    std::vector<float> rowSum(static_cast<size_t>(iterations) * batchSize * seqLength * matrixSize);
    std::vector<float> colSum(static_cast<size_t>(iterations) * batchSize * seqLength * matrixSize);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> inputDist(-1e-3f, 1e-3f);
    std::uniform_real_distribution<float> gradDist(-100.0f, 100.0f);
    for (size_t idx = 0; idx < totalSize; ++idx) {
        sinkhornInput[idx] = expf(inputDist(rng)) + 1e-3f;
    }

    const float epsilon = 1.0e-6f;
    MhcSinkhornForwardGolden(sinkhornInput.data(),
                             sinkhornOutput.data(),
                             rowSum.data(),
                             colSum.data(),
                             batchSize,
                             seqLength,
                             matrixSize,
                             epsilon,
                             iterations);

    for (size_t idx = 0; idx < totalSize; ++idx) {
        gradOutput[idx] = gradDist(rng);
    }

    MhcSinkhornBackwardGolden(sinkhornOutput.data(),
                              gradOutput.data(),
                              gradXGolden.data(),
                              rowSum.data(),
                              colSum.data(),
                              batchSize,
                              seqLength,
                              matrixSize,
                              iterations);

    float* sinkhornInputDevice = nullptr;
    float* gradOutputDevice = nullptr;
    float* gradXDevice = nullptr;
    CheckCuda(cudaMalloc(&sinkhornInputDevice, totalSize * sizeof(float)), "cudaMalloc sinkhornInputDevice");
    CheckCuda(cudaMalloc(&gradOutputDevice, totalSize * sizeof(float)), "cudaMalloc gradOutputDevice");
    CheckCuda(cudaMalloc(&gradXDevice, totalSize * sizeof(float)), "cudaMalloc gradXDevice");
    CheckCuda(cudaMemcpy(sinkhornInputDevice, sinkhornInput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy sinkhornInput");
    CheckCuda(cudaMemcpy(gradOutputDevice, gradOutput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy gradOutput");

    for (int run = 0; run < warmupRuns; ++run) {
        LaunchMhcSinkhornBackwardKernel<matrixSize, iterations>(sinkhornInputDevice,
                                        gradOutputDevice,
                                        gradXDevice,
                                        batchSize,
                                        seqLength,
                                        epsilon);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    for (int run = 0; run < timedRuns; ++run) {
        LaunchMhcSinkhornBackwardKernel<matrixSize, iterations>(sinkhornInputDevice,
                                        gradOutputDevice,
                                        gradXDevice,
                                        batchSize,
                                        seqLength,
                                        epsilon);
    }
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");

    const double bytesProcessed = static_cast<double>(batchSize) * seqLength * matrixSize * matrixSize * 3.0 *
                                  static_cast<double>(sizeof(float));
    const double avgElapsedMs = static_cast<double>(elapsedMs) / static_cast<double>(timedRuns);
    const double effectiveBandwidthGb = ComputeEffectiveBandwidthGb(bytesProcessed, avgElapsedMs);
    std::cout << "Avg elapsed time (ms): " << avgElapsedMs << '\n';
    std::cout << "Effective bandwidth (GB/s): " << effectiveBandwidthGb << '\n';

    CheckCuda(cudaMemcpy(sinkhornInputDevice, sinkhornInput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy sinkhornInput reset");
    LaunchMhcSinkhornBackwardKernel<matrixSize, iterations>(sinkhornInputDevice,
                                    gradOutputDevice,
                                    gradXDevice,
                                    batchSize,
                                    seqLength,
                                    epsilon);
    CheckCuda(cudaGetLastError(), "MhcSinkhornBackwardKernel launch");

    CheckCuda(cudaMemcpy(gradXCuda.data(), gradXDevice, totalSize * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy gradX");

    const float rmse = ComputeRmse(gradXGolden.data(), gradXCuda.data(), totalSize);
    std::cout << "RMSE (gradX): " << rmse << '\n';

    CheckCuda(cudaFree(sinkhornInputDevice), "cudaFree sinkhornInputDevice");
    CheckCuda(cudaFree(gradOutputDevice), "cudaFree gradOutputDevice");
    CheckCuda(cudaFree(gradXDevice), "cudaFree gradXDevice");

    return 0;
}
