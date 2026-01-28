// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for BF16 GLU kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

constexpr int kGluBatchSize = 128;
constexpr int kIntermediateSize = 3584;
constexpr int kK = 8192;

/**
 * @brief Computes the RMSE between reference and computed outputs.
 * @param golden Pointer to reference data on the host.
 * @param values Pointer to computed data on the host.
 * @param numElements Number of elements to compare.
 * @return RMSE value.
 */
float ComputeRmse(const float* __restrict__ golden,
                  const float* __restrict__ values,
                  size_t numElements) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < numElements; ++i) {
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * @brief Runs the GLU BF16 kernel test and reports performance.
 * @return Process exit code.
 */
int main() {
    static_assert(A_TILE_K == B_TILE_K, "A_TILE_K must equal B_TILE_K.");
    assert(kGluBatchSize % BLOCK_M == 0 && kIntermediateSize % BLOCK_N == 0 && kK % A_TILE_K == 0);

    const int repeatIters = 1;

    const unsigned seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t xElements = static_cast<size_t>(kGluBatchSize) * kK;
    const size_t weightElements = static_cast<size_t>(kK) * kIntermediateSize * 2;
    const size_t resultElements = static_cast<size_t>(kGluBatchSize) * kIntermediateSize;

    std::vector<Bf16> xHost(xElements);
    std::vector<Bf16> weightHost(weightElements);
    std::vector<float> resultTrue(resultElements);

    for (size_t i = 0; i < xElements; ++i) {
        xHost[i] = static_cast<Bf16>(distribution(generator));
    }
    for (size_t i = 0; i < weightElements; ++i) {
        weightHost[i] = static_cast<Bf16>(distribution(generator));
    }

#pragma omp parallel for
    for (int m = 0; m < kGluBatchSize; ++m) {
        for (int n = 0; n < kIntermediateSize; ++n) {
            float sum = 0.0f;
            float sum2 = 0.0f;
            for (int kIndex = 0; kIndex < kK; ++kIndex) {
                sum += __bfloat162float(xHost[m * kK + kIndex]) *
                       __bfloat162float(weightHost[n * kK + kIndex]);
            }
            for (int kIndex = 0; kIndex < kK; ++kIndex) {
                sum2 += __bfloat162float(xHost[m * kK + kIndex]) *
                        __bfloat162float(weightHost[(n + kIntermediateSize) * kK + kIndex]);
            }
            resultTrue[m * kIntermediateSize + n] = sum * Activation(sum2);
        }
    }

    Bf16* xDevice = nullptr;
    Bf16* weightDevice = nullptr;
    float* resultDevice = nullptr;
    cudaMalloc(&xDevice, xElements * sizeof(Bf16));
    cudaMalloc(&weightDevice, weightElements * sizeof(Bf16));
    cudaMalloc(&resultDevice, resultElements * sizeof(float));
    cudaMemcpy(xDevice, xHost.data(), xElements * sizeof(Bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(weightDevice, weightHost.data(), weightElements * sizeof(Bf16), cudaMemcpyHostToDevice);

    std::vector<float> resultHost(resultElements);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeatIters; ++i) {
        LaunchGluBf16Kernel(xDevice, weightDevice, resultDevice, kGluBatchSize, kK, kIntermediateSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTimeMs = 0.0f;
    cudaEventElapsedTime(&elapsedTimeMs, start, stop);
    elapsedTimeMs /= repeatIters;
    std::cout << "GluBf16Kernel elapsed time: " << elapsedTimeMs * 1000.0f << " us" << std::endl;

    const float peakTflops = (2.0f * kGluBatchSize * kIntermediateSize * 2 * kK +
                              kGluBatchSize * kIntermediateSize * 2.0f) /
                             1e12f / (elapsedTimeMs / 1e3f);
    std::cout << "GluBf16Kernel peak TFLOPS: " << peakTflops << " TFLOPS" << std::endl;

    cudaMemcpy(resultHost.data(), resultDevice, resultElements * sizeof(float), cudaMemcpyDeviceToHost);
    const float error = ComputeRmse(resultTrue.data(), resultHost.data(), resultElements);
    std::cout << "GluBf16Kernel error: " << error << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(xDevice);
    cudaFree(weightDevice);
    cudaFree(resultDevice);

    return 0;
}
