// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for bitonic sort kernel.
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

constexpr bool kSortAscending = true;
constexpr int kBitonicBatchSize = 1;
constexpr int kBitonicArrayLength = 1024 * 1024;
constexpr int kWarmupIterations = 1000;

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
 * @return RMSE value.
 */
float ComputeRmse(const float* __restrict__ golden, const float* __restrict__ values, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * @brief CPU reference bitonic sort (row-wise) to produce golden output.
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
 * @brief Runs the bitonic sort benchmark and validation.
 * @return Process exit code.
 */
int main() {
    static_assert(kBitonicArrayLength % 2 == 0, "kBitonicArrayLength must be even");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t numElements = static_cast<size_t>(kBitonicBatchSize) * kBitonicArrayLength;
    const size_t totalBytes = numElements * sizeof(float);

    std::vector<float> hostInput(numElements);
    std::vector<float> hostGolden(numElements);
    std::vector<float> hostResult(numElements, 0.0f);

    std::generate(hostInput.begin(), hostInput.end(), [&]() { return distribution(generator); });

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    SortCpu(hostInput.data(), hostGolden.data(), kBitonicBatchSize, kBitonicArrayLength);
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuStop - cpuStart).count();
    std::cout << "CPU sort duration: " << cpuDuration << " us" << std::endl;

    float* deviceInput = nullptr;
    CheckCuda(cudaMalloc(&deviceInput, totalBytes), "cudaMalloc deviceInput");
    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput to deviceInput");

    for (int i = 0; i < kWarmupIterations; ++i) {
        LaunchBitonicSortFp32Kernel(deviceInput, kBitonicBatchSize, kBitonicArrayLength, kSortAscending);
        CheckCuda(cudaGetLastError(), "BitonicSortFp32Kernel warmup");
    }

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput before run");
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    LaunchBitonicSortFp32Kernel(deviceInput, kBitonicBatchSize, kBitonicArrayLength, kSortAscending);
    CheckCuda(cudaGetLastError(), "BitonicSortFp32Kernel run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    std::cout << "BitonicSortFp32Kernel duration: " << elapsedMs * 1000.0f << " us" << std::endl;
    const double reachedMemBw =
        static_cast<double>(totalBytes) * 2.0 / (static_cast<double>(elapsedMs) / 1000.0) /
        static_cast<double>(1ULL << 40);
    std::cout << "BitonicSortFp32Kernel reached_mem_bw: " << reachedMemBw << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceInput, totalBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy deviceInput to hostResult");
    const float error = ComputeRmse(hostGolden.data(), hostResult.data(), numElements);
    std::cout << "BitonicSortFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");

    return 0;
}
