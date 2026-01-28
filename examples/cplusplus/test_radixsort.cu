// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for radix sort kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kRadixBits = 8;
constexpr int kRadixHistogramSize = 1 << kRadixBits;
constexpr int kRadixSortBatchSize = 1;
constexpr int kRadixSortArrayLength = 1024 * 1024;
constexpr int kWorkPerThread = 16;
constexpr int kThreadsPerBlock = 256;
constexpr int kWorkPerBlock = kThreadsPerBlock * kWorkPerThread;
constexpr int kNumBlocksPerBatch =
    (kRadixSortArrayLength + kWorkPerBlock - 1) / kWorkPerBlock;

/**
 * @brief Validates CUDA API results and aborts on failure.
 * @param result CUDA status code returned by CUDA runtime.
 * @param message Human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes RMSE between the golden output and computed values.
 * @param golden Reference data on host.
 * @param values Computed data on host.
 * @param count Number of elements.
 * @return RMSE metric.
 */
float ComputeRmse(const float* __restrict__ golden, const float* __restrict__ values, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm + 1e-6));
}

/**
 * @brief CPU reference radix sort (row-wise) to produce golden output.
 * @param input Pointer to input buffer.
 * @param output Pointer to output buffer.
 * @param batchSize Number of rows.
 * @param arrayLength Length of each row.
 */
void SortCpu(const float* __restrict__ input, float* __restrict__ output, int batchSize, int arrayLength) {
    std::vector<float> buffer(arrayLength);
    for (int batch = 0; batch < batchSize; ++batch) {
        const float* rowIn = input + batch * arrayLength;
        float* rowOut = output + batch * arrayLength;
        std::copy(rowIn, rowIn + arrayLength, buffer.begin());
        std::sort(buffer.begin(), buffer.end());
        std::copy(buffer.begin(), buffer.end(), rowOut);
    }
}

}  // namespace

/**
 * @brief Runs the radix sort benchmark and validation.
 * @return Process exit code.
 */
int main() {
    static_assert(kRadixSortArrayLength % 4 == 0, "Array length must be multiples of 4");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t numElements = static_cast<size_t>(kRadixSortBatchSize) * kRadixSortArrayLength;
    const size_t totalBytes = numElements * sizeof(float);

    std::vector<float> hostInput(numElements);
    std::vector<float> hostGolden(numElements);
    std::vector<float> hostResult(numElements, 0.0f);

    std::generate(hostInput.begin(), hostInput.end(), [&]() { return distribution(generator); });

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    SortCpu(hostInput.data(), hostGolden.data(), kRadixSortBatchSize, kRadixSortArrayLength);
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuStop - cpuStart).count();
    std::cout << "CPU radix sort duration: " << cpuDuration << " us" << std::endl;

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    int* deviceHistogram = nullptr;

    CheckCuda(cudaMalloc(&deviceInput, totalBytes), "cudaMalloc deviceInput");
    CheckCuda(cudaMalloc(&deviceOutput, totalBytes), "cudaMalloc deviceOutput");
    CheckCuda(cudaMalloc(&deviceHistogram,
                         static_cast<size_t>(kRadixSortBatchSize) *
                             static_cast<size_t>(kNumBlocksPerBatch) *
                             static_cast<size_t>(kRadixHistogramSize) * sizeof(int)),
              "cudaMalloc deviceHistogram");
    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput to deviceInput");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    for (int i = 0; i < 1000; ++i) {
        LaunchRadixSortFp32Kernel(deviceInput,
                                  deviceOutput,
                                  deviceHistogram,
                                  kRadixSortBatchSize,
                                  kRadixSortArrayLength);
    }

    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput to deviceInput");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    LaunchRadixSortFp32Kernel(deviceInput,
                              deviceOutput,
                              deviceHistogram,
                              kRadixSortBatchSize,
                              kRadixSortArrayLength);
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    std::cout << "GPU radix sort duration: " << elapsedMs * 1000 << " us" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceOutput, totalBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy deviceOutput to hostResult");
    const float error = ComputeRmse(hostGolden.data(), hostResult.data(), numElements);
    std::cout << "Radix sort RMSE = " << error << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCuda(cudaFree(deviceHistogram), "cudaFree deviceHistogram");
    CheckCuda(cudaFree(deviceOutput), "cudaFree deviceOutput");
    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");

    return 0;
}
