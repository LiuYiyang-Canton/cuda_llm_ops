// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for TopK kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace {

constexpr int kRadixHistogramSize = 256;
constexpr int kTopkBatchSize = 4;
constexpr int kTopkFeatureSize = 128 * 1024;
constexpr int kTopkValue = 2048;

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
 * @brief Comparator for sorting indices by descending value within a row.
 */
struct TopkComparator {
    const float* values;
    int featureSize;
    int row;

    /**
     * @brief Returns true if value at index a is greater than value at index b for the row.
     * @param a Index of first element.
     * @param b Index of second element.
     * @return True if values[a] > values[b].
     */
    bool operator()(int a, int b) const {
        return values[row * featureSize + a] > values[row * featureSize + b];
    }
};

/**
 * @brief Computes golden top-k indices on CPU for validation.
 * @param values Pointer to input values.
 * @param output Pointer to output indices buffer.
 * @param batchSize Batch size.
 * @param featureSize Feature dimension.
 * @param topk Number of top entries to select.
 */
void ComputeTopkGolden(const float* values,
                       int* output,
                       int batchSize,
                       int featureSize,
                       int topk) {
    std::vector<int> indices(static_cast<size_t>(featureSize));
    for (int row = 0; row < batchSize; ++row) {
        std::iota(indices.begin(), indices.end(), 0);
        TopkComparator comparator{values, featureSize, row};
        std::sort(indices.begin(), indices.end(), comparator);
        std::copy_n(indices.begin(), topk, output + row * topk);
        std::sort(output + row * topk, output + (row + 1) * topk);
    }
}

}  // namespace

/**
 * @brief Runs the top-k radix select benchmark and validation.
 * @return Process exit code.
 */
int main() {
    static_assert(kTopkFeatureSize % 4 == 0, "kTopkFeatureSize must be divisible by 4");

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t inputSize = static_cast<size_t>(kTopkBatchSize) * kTopkFeatureSize;
    const size_t outputSize = static_cast<size_t>(kTopkBatchSize) * kTopkValue;

    std::vector<float> hostInput(inputSize);
    for (size_t i = 0; i < inputSize; ++i) {
        hostInput[i] = distribution(generator);
    }

    std::vector<int> result(outputSize);
    std::vector<int> golden(outputSize);

    ComputeTopkGolden(hostInput.data(), golden.data(), kTopkBatchSize, kTopkFeatureSize, kTopkValue);

    float* deviceInput = nullptr;
    bool* deviceTopkMask = nullptr;
    int* deviceHistogram = nullptr;
    int* deviceTargetTopk = nullptr;
    uint32_t* deviceSelectedRadix = nullptr;

    std::vector<int> targetTopk(kTopkBatchSize, kTopkValue);

    CheckCuda(cudaMalloc(&deviceInput, inputSize * sizeof(float)), "cudaMalloc deviceInput");
    CheckCuda(cudaMalloc(&deviceTopkMask, inputSize * sizeof(bool)), "cudaMalloc deviceTopkMask");
    CheckCuda(cudaMalloc(&deviceTargetTopk, kTopkBatchSize * sizeof(int)), "cudaMalloc deviceTargetTopk");
    CheckCuda(cudaMalloc(&deviceHistogram, kTopkBatchSize * kRadixHistogramSize * sizeof(int)),
              "cudaMalloc deviceHistogram");
    CheckCuda(cudaMalloc(&deviceSelectedRadix, kTopkBatchSize * sizeof(uint32_t)),
              "cudaMalloc deviceSelectedRadix");

    CheckCuda(cudaMemcpy(deviceTargetTopk, targetTopk.data(), kTopkBatchSize * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy targetTopk");
    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy input");

    CheckCuda(cudaMemset(deviceHistogram, 0, kTopkBatchSize * kRadixHistogramSize * sizeof(int)),
              "cudaMemset deviceHistogram");
    CheckCuda(cudaMemset(deviceSelectedRadix, 0, kTopkBatchSize * sizeof(uint32_t)),
              "cudaMemset deviceSelectedRadix");
    CheckCuda(cudaMemset(deviceTopkMask, 0, inputSize * sizeof(bool)), "cudaMemset deviceTopkMask");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    for (int i = 0; i < 1000; ++i) {
        LaunchTopkRadixSelectFp32(deviceInput,
                                  deviceTopkMask,
                                  deviceHistogram,
                                  deviceTargetTopk,
                                  deviceSelectedRadix,
                                  kTopkBatchSize,
                                  kTopkFeatureSize);
    }

    CheckCuda(cudaMemcpy(deviceTargetTopk, targetTopk.data(), kTopkBatchSize * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy targetTopk reset");
    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy input reset");
    CheckCuda(cudaMemset(deviceHistogram, 0, kTopkBatchSize * kRadixHistogramSize * sizeof(int)),
              "cudaMemset histogram reset");
    CheckCuda(cudaMemset(deviceSelectedRadix, 0, kTopkBatchSize * sizeof(uint32_t)),
              "cudaMemset selectedRadix reset");
    CheckCuda(cudaMemset(deviceTopkMask, 0, inputSize * sizeof(bool)), "cudaMemset topkMask reset");

    CheckCuda(cudaEventRecord(startEvent, 0), "cudaEventRecord startEvent");
    LaunchTopkRadixSelectFp32(deviceInput,
                              deviceTopkMask,
                              deviceHistogram,
                              deviceTargetTopk,
                              deviceSelectedRadix,
                              kTopkBatchSize,
                              kTopkFeatureSize);
    CheckCuda(cudaEventRecord(stopEvent, 0), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");

    float elapsedTime = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent), "cudaEventElapsedTime");
    std::cout << "Radix select top-k kernel time: " << elapsedTime * 1000 << " us" << std::endl;

    std::vector<unsigned char> topkMaskHost(inputSize);
    CheckCuda(cudaMemcpy(topkMaskHost.data(), deviceTopkMask, inputSize * sizeof(bool), cudaMemcpyDeviceToHost),
              "cudaMemcpy topkMask");

    for (int row = 0; row < kTopkBatchSize; ++row) {
        int count = 0;
        for (int col = 0; col < kTopkFeatureSize; ++col) {
            if (topkMaskHost[static_cast<size_t>(row) * kTopkFeatureSize + col] != 0) {
                result[static_cast<size_t>(row) * kTopkValue + count] = col;
                count++;
                if (count == kTopkValue) {
                    break;
                }
            }
        }
        std::sort(result.begin() + static_cast<size_t>(row) * kTopkValue,
                  result.begin() + static_cast<size_t>(row + 1) * kTopkValue);
    }

    for (int row = 0; row < kTopkBatchSize; ++row) {
        std::vector<int> diff;
        std::set_difference(golden.begin() + static_cast<size_t>(row) * kTopkValue,
                            golden.begin() + static_cast<size_t>(row + 1) * kTopkValue,
                            result.begin() + static_cast<size_t>(row) * kTopkValue,
                            result.begin() + static_cast<size_t>(row + 1) * kTopkValue,
                            std::back_inserter(diff));
        std::set_difference(result.begin() + static_cast<size_t>(row) * kTopkValue,
                            result.begin() + static_cast<size_t>(row + 1) * kTopkValue,
                            golden.begin() + static_cast<size_t>(row) * kTopkValue,
                            golden.begin() + static_cast<size_t>(row + 1) * kTopkValue,
                            std::back_inserter(diff));
        std::cout << "Number of different indices in batch " << row << ": " << diff.size() << std::endl;
    }

    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");
    CheckCuda(cudaFree(deviceTopkMask), "cudaFree deviceTopkMask");
    CheckCuda(cudaFree(deviceHistogram), "cudaFree deviceHistogram");
    CheckCuda(cudaFree(deviceTargetTopk), "cudaFree deviceTargetTopk");
    CheckCuda(cudaFree(deviceSelectedRadix), "cudaFree deviceSelectedRadix");

    return 0;
}
