// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for mHCSinkhorn kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

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
    double sumSquared = 0.0;
    for (size_t idx = 0; idx < elementCount; ++idx) {
        const double diff = static_cast<double>(reference[idx]) - static_cast<double>(candidate[idx]);
        sumSquared += diff * diff;
    }
    const double meanSquared = sumSquared / static_cast<double>(elementCount);
    return static_cast<float>(std::sqrt(meanSquared));
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
 * @brief Normalizes each row of a square matrix in-place so rows sum to 1.
 * @param data Pointer to the matrix data of shape (matrixSize, matrixSize).
 * @param matrixSize Matrix dimension N.
 * @param epsilon Small constant to avoid division by zero.
 */
void NormalizeRows(float* data, int matrixSize, float epsilon) {
    for (int i = 0; i < matrixSize; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; ++j) {
            rowSum += static_cast<double>(data[i * matrixSize + j]);
        }
        const double denom = rowSum + epsilon;
        for (int j = 0; j < matrixSize; ++j) {
            data[i * matrixSize + j] = static_cast<float>(data[i * matrixSize + j] / denom);
        }
    }
}

/**
 * @brief Normalizes each column of a square matrix in-place so columns sum to 1.
 * @param data Pointer to the matrix data of shape (matrixSize, matrixSize).
 * @param matrixSize Matrix dimension N.
 * @param epsilon Small constant to avoid division by zero.
 */
void NormalizeCols(float* data, int matrixSize, float epsilon) {
    for (int j = 0; j < matrixSize; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < matrixSize; ++i) {
            colSum += static_cast<double>(data[i * matrixSize + j]);
        }
        const double denom = colSum + epsilon;
        for (int i = 0; i < matrixSize; ++i) {
            data[i * matrixSize + j] = static_cast<float>(data[i * matrixSize + j] / denom);
        }
    }
}

/**
 * @brief Applies Sinkhorn-Knopp iterations to make each (N, N) slice approximately doubly stochastic.
 * @param input Pointer to input tensor H with shape (batchSize, seqLength, matrixSize, matrixSize).
 * @param output Pointer to output tensor with the same shape as input.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param iterations Number of Sinkhorn iterations to run.
 */
void MhcSinkhornCpu(const float* input,
                    float* output,
                    int batchSize,
                    int seqLength,
                    int matrixSize,
                    int iterations) {
    const size_t matrixElemCount = static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
    const size_t sliceSize = matrixElemCount;
    const size_t totalSlices = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength);
    const float epsilon = 1.0e-6f;

    std::copy(input, input + totalSlices * sliceSize, output);

    for (size_t slice = 0; slice < totalSlices; ++slice) {
        float* matrix = output + slice * sliceSize;
        for (int iter = 0; iter < iterations; ++iter) {
            NormalizeRows(matrix, matrixSize, epsilon);
            NormalizeCols(matrix, matrixSize, epsilon);
        }
    }
}

/**
 * @brief Prints row and column sums for a selected slice in the output tensor.
 * @param data Pointer to the output tensor data.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param sampleB Batch index to inspect.
 * @param sampleS Sequence index to inspect.
 */
void PrintRowColSums(const float* data,
                     int batchSize,
                     int seqLength,
                     int matrixSize,
                     int sampleB,
                     int sampleS) {
    (void)batchSize;
    std::cout << "Row sums (b=" << sampleB << ", s=" << sampleS << "): ";
    for (int i = 0; i < matrixSize; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; ++j) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            rowSum += static_cast<double>(data[idx]);
        }
        std::cout << static_cast<float>(rowSum) << (i + 1 == matrixSize ? '\n' : ' ');
    }

    std::cout << "Col sums (b=" << sampleB << ", s=" << sampleS << "): ";
    for (int j = 0; j < matrixSize; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < matrixSize; ++i) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            colSum += static_cast<double>(data[idx]);
        }
        std::cout << static_cast<float>(colSum) << (j + 1 == matrixSize ? '\n' : ' ');
    }
}

}  // namespace

/**
 * @brief Entry point that builds a sample tensor and runs the Sinkhorn CPU/GPU comparison.
 * @return Exit code (0 on success).
 */
int main() {
    const int batchSize = 2;
    const int seqLength = 128 * 1024;
    constexpr int matrixSize = 4;
    const int iterations = 20;
    const float epsilon = 1.0e-6f;
    const int warmupRuns = 20;
    const int timedRuns = 20;

    static_assert(matrixSize % 4 == 0, "matrixSize must be a multiple of 4");

    const size_t totalSize = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength) *
                             static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);

    float* input = new float[totalSize];
    float* cpuOutput = new float[totalSize];
    float* gpuOutput = new float[totalSize];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t idx = 0; idx < totalSize; ++idx) {
        input[idx] = expf(dist(rng)) + 1.0e-3f;
    }

    MhcSinkhornCpu(input, cpuOutput, batchSize, seqLength, matrixSize, iterations);

    float* deviceInput = nullptr;
    CheckCuda(cudaMalloc(&deviceInput, totalSize * sizeof(float)), "cudaMalloc deviceInput");
    CheckCuda(cudaMemcpy(deviceInput, input, totalSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D input");

    LaunchMhcSinkhornKernel<matrixSize>(deviceInput, batchSize, seqLength, iterations, epsilon);
    CheckCuda(cudaGetLastError(), "MhcSinkhornKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    CheckCuda(cudaMemcpy(gpuOutput, deviceInput, totalSize * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H output");

    const int sampleB = 0;
    const int sampleS = 0;
    std::cout << "Sample slice (b=0, s=0) after Sinkhorn:" << std::fixed << std::setprecision(4)
              << '\n';
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            std::cout << cpuOutput[idx] << (j + 1 == matrixSize ? '\n' : ' ');
        }
    }
    PrintRowColSums(cpuOutput, batchSize, seqLength, matrixSize, sampleB, sampleS);

    const float rmse = ComputeRmse(cpuOutput, gpuOutput, totalSize);
    std::cout << "RMSE (CPU vs GPU): " << rmse << '\n';

    CheckCuda(cudaMemcpy(deviceInput, input, totalSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D input (timing)");
    for (int run = 0; run < warmupRuns; ++run) {
        LaunchMhcSinkhornKernel<matrixSize>(deviceInput, batchSize, seqLength, iterations, epsilon);
    }
    CheckCuda(cudaGetLastError(), "MhcSinkhornKernel warmup launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    for (int run = 0; run < timedRuns; ++run) {
        LaunchMhcSinkhornKernel<matrixSize>(deviceInput, batchSize, seqLength, iterations, epsilon);
    }
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");
    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");

    const double avgMs = static_cast<double>(elapsedMs) / static_cast<double>(timedRuns);
    const double avgUs = avgMs * 1.0e3;
    const double bytesPerLaunch = static_cast<double>(batchSize) * static_cast<double>(seqLength) *
                                  static_cast<double>(matrixSize) * static_cast<double>(matrixSize) *
                                  static_cast<double>(sizeof(float));
    const double bandwidthGb = ComputeEffectiveBandwidthGb(bytesPerLaunch, avgMs);
    std::cout << "Kernel time: " << avgUs << " us (avg over " << timedRuns << " runs, "
              << warmupRuns << " warmups)\n";
    std::cout << "Effective bandwidth: " << bandwidthGb << " GB/s\n";

    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");

    delete[] input;
    delete[] cpuOutput;
    delete[] gpuOutput;

    return 0;
}
