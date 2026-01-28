// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for LayerNorm kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <random>

constexpr int kLayerNormSeqLength = 4096;
constexpr int kLayerNormHiddenSize = 7168;

/**
 * @brief Computes LayerNorm forward on CPU for validation.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param layernormOut Output tensor with shape (seqLength, hiddenSize).
 * @param seqLength Number of rows in the input.
 * @param hiddenSize Hidden dimension size.
 * @param epsilon Epsilon added to the variance before sqrt.
 */
void LayerNormForwardCpu(const float* x,
                         const float* weight,
                         float* layernormOut,
                         int seqLength,
                         int hiddenSize,
                         float epsilon) {
    const float invHiddenSize = 1.0f / static_cast<float>(hiddenSize);
    for (int row = 0; row < seqLength; ++row) {
        const float* rowPtr = x + static_cast<size_t>(row) * hiddenSize;
        float* outPtr = layernormOut + static_cast<size_t>(row) * hiddenSize;
        float sum = 0.0f;
        for (int col = 0; col < hiddenSize; ++col) {
            sum += rowPtr[col];
        }
        const float avg = sum * invHiddenSize;
        float sumSq = 0.0f;
        for (int col = 0; col < hiddenSize; ++col) {
            const float diff = rowPtr[col] - avg;
            sumSq += diff * diff;
        }
        const float invRms = 1.0f / std::sqrt(sumSq * invHiddenSize + epsilon);
        for (int col = 0; col < hiddenSize; ++col) {
            outPtr[col] = (rowPtr[col] - avg) * invRms * weight[col];
        }
    }
}

/**
 * @brief Fills a buffer with random uniform values in [minValue, maxValue).
 * @param data Output buffer to fill.
 * @param count Number of elements to generate.
 * @param generator Random generator to use.
 * @param minValue Lower bound of the uniform distribution.
 * @param maxValue Upper bound of the uniform distribution.
 */
void FillRandomUniform(float* data,
                       size_t count,
                       std::mt19937& generator,
                       float minValue,
                       float maxValue) {
    std::uniform_real_distribution<float> distribution(minValue, maxValue);
    for (size_t i = 0; i < count; ++i) {
        data[i] = distribution(generator);
    }
}

/**
 * @brief Computes RMSE between reference and computed buffers.
 * @param reference Reference data on host.
 * @param values Computed data on host.
 * @param count Number of fp32 elements to compare.
 * @return RMSE value in fp32.
 */
float ComputeRmse(const float* reference, const float* values, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(reference[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
    }
    if (norm == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * @brief Entry point that runs LayerNorm forward on CPU/GPU and reports metrics.
 * @return 0 on success, non-zero on failure.
 */
int main() {
    assert((kLayerNormHiddenSize % 4) == 0);

    const size_t elementCount = static_cast<size_t>(kLayerNormSeqLength) *
                                static_cast<size_t>(kLayerNormHiddenSize);
    const size_t elementBytes = elementCount * sizeof(float);

    float* x = new float[elementCount];
    float* weight = new float[kLayerNormHiddenSize];
    float* layernormGolden = new float[elementCount];
    float* layernormOut = new float[elementCount];

    const unsigned int seed =
        static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    FillRandomUniform(x, elementCount, generator, 0.0f, 1.0f);
    FillRandomUniform(weight, kLayerNormHiddenSize, generator, 0.0f, 1.0f);

    LayerNormForwardCpu(x, weight, layernormGolden, kLayerNormSeqLength, kLayerNormHiddenSize,
                        kLayerNormEpsilon);

    float* xDevice = nullptr;
    float* weightDevice = nullptr;
    float* layernormOutDevice = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&xDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&weightDevice), kLayerNormHiddenSize * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&layernormOutDevice), elementBytes);
    cudaMemcpy(xDevice, x, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(weightDevice, weight, kLayerNormHiddenSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmupRuns = 1000;
    const int timedRuns = 1;
    for (int i = 0; i < warmupRuns; ++i) {
        LaunchLayerNormFp32Kernel(xDevice, weightDevice, layernormOutDevice, kLayerNormSeqLength,
                                  kLayerNormHiddenSize);
    }

    cudaEventRecord(start);
    for (int i = 0; i < timedRuns; ++i) {
        LaunchLayerNormFp32Kernel(xDevice, weightDevice, layernormOutDevice, kLayerNormSeqLength,
                                  kLayerNormHiddenSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgUs = (elapsedMs * 1000.0f) / static_cast<float>(timedRuns);
    std::cout << "LayerNormFp32Kernel avg time: " << avgUs << " us" << std::endl;
    const double reachedMemBwTb =
        (static_cast<double>(elementBytes) * 2.0) /
        (1024.0 * 1024.0 * 1024.0 * 1024.0) /
        (elapsedMs / 1000.0f);
    std::cout << "LayerNormFp32Kernel reached memory bandwidth: " << reachedMemBwTb << " TB/s"
              << std::endl;

    cudaMemcpy(layernormOut, layernormOutDevice, elementBytes, cudaMemcpyDeviceToHost);

    const float error = ComputeRmse(layernormGolden, layernormOut, elementCount);
    std::cout << "LayerNormFp32Kernel RMSE: " << error << std::endl;

    cudaFree(xDevice);
    cudaFree(weightDevice);
    cudaFree(layernormOutDevice);
    delete[] x;
    delete[] weight;
    delete[] layernormGolden;
    delete[] layernormOut;
    return 0;
}
