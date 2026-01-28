// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for softmax cross-entropy backward kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

constexpr int kSoftmaxCrossEntropyBackwardBatchSize = 32;
constexpr int kSoftmaxCrossEntropyBackwardFeatureSize = 131072;

/**
 * @brief Fills a buffer with random uniform values.
 * @param data Host pointer to output buffer.
 * @param count Number of elements to fill.
 * @param generator RNG generator.
 * @param minValue Minimum value.
 * @param maxValue Maximum value.
 */
void FillRandomUniform(float* data, int count, std::mt19937& generator, float minValue, float maxValue) {
    std::uniform_real_distribution<float> distribution(minValue, maxValue);
    for (int i = 0; i < count; ++i) {
        data[i] = distribution(generator);
    }
}

/**
 * @brief Fills a buffer with random labels.
 * @param data Host pointer to output buffer.
 * @param count Number of labels to fill.
 * @param generator RNG generator.
 * @param numClasses Number of classes.
 */
void FillRandomLabels(int* data, int count, std::mt19937& generator, int numClasses) {
    std::uniform_int_distribution<int> distribution(0, numClasses - 1);
    for (int i = 0; i < count; ++i) {
        data[i] = distribution(generator);
    }
}

/**
 * @brief Reference CPU implementation for softmax cross-entropy backward.
 * @param logits Host pointer to input logits.
 * @param labels Host pointer to label indices.
 * @param gradLogits Host pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void SoftmaxCrossEntropyBackwardCpu(const float* logits, const int* labels,
                                    float* gradLogits, int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowLogits = logits + static_cast<size_t>(row) * featureSize;
        float* rowGrad = gradLogits + static_cast<size_t>(row) * featureSize;

        float maxVal = -INFINITY;
        for (int col = 0; col < featureSize; ++col) {
            const float val = rowLogits[col];
            if (val > maxVal) {
                maxVal = val;
            }
        }

        float sumExp = 0.0f;
        for (int col = 0; col < featureSize; ++col) {
            sumExp += expf(rowLogits[col] - maxVal);
        }

        for (int col = 0; col < featureSize; ++col) {
            rowGrad[col] = expf(rowLogits[col] - maxVal) / sumExp;
        }

        const int label = labels[row];
        rowGrad[label] -= 1.0f;
    }
}

/**
 * @brief Computes RMSE between the golden output and kernel results.
 * @param golden Host pointer to reference data.
 * @param values Host pointer to computed data.
 * @param count Number of fp32 elements to compare.
 * @return RMSE in fp32.
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
 * @brief Entry point: allocates data, runs CPU reference and GPU kernel, and times the result.
 * @return Exit code (0 on success).
 */
int main() {
    static_assert(kSoftmaxCrossEntropyBackwardFeatureSize % 4 == 0,
                  "feature size must be a multiple of 4");
    static_assert(kSoftmaxCrossEntropyBackwardThreadsPerBlock % kSoftmaxCrossEntropyBackwardWarpSize == 0,
                  "threads per block must be a multiple of warp size");

    const unsigned seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    const int totalCount = kSoftmaxCrossEntropyBackwardBatchSize * kSoftmaxCrossEntropyBackwardFeatureSize;
    float* logits = new float[totalCount];
    int* labels = new int[kSoftmaxCrossEntropyBackwardBatchSize];
    float* gradGolden = new float[totalCount];

    FillRandomUniform(logits, totalCount, generator, 0.0f, 1.0f);
    FillRandomLabels(labels, kSoftmaxCrossEntropyBackwardBatchSize, generator,
                     kSoftmaxCrossEntropyBackwardFeatureSize);
    SoftmaxCrossEntropyBackwardCpu(logits, labels, gradGolden,
                                   kSoftmaxCrossEntropyBackwardBatchSize,
                                   kSoftmaxCrossEntropyBackwardFeatureSize);

    float* logitsDevice = nullptr;
    float* gradDevice = nullptr;
    float* rowMaxDevice = nullptr;
    float* rowSumDevice = nullptr;
    int* labelsDevice = nullptr;
    cudaMalloc(&logitsDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&gradDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&rowMaxDevice, kSoftmaxCrossEntropyBackwardBatchSize * sizeof(float));
    cudaMalloc(&rowSumDevice, kSoftmaxCrossEntropyBackwardBatchSize * sizeof(float));
    cudaMalloc(&labelsDevice, kSoftmaxCrossEntropyBackwardBatchSize * sizeof(int));
    cudaMemcpy(logitsDevice, logits, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(labelsDevice, labels, kSoftmaxCrossEntropyBackwardBatchSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedMs = 0.0f;

    cudaMemset(gradDevice, 0, static_cast<size_t>(totalCount) * sizeof(float));
    cudaEventRecord(start);
    LaunchSoftmaxCrossEntropyBackwardRowMaxRowSumFp32Kernel(
        logitsDevice,
        rowMaxDevice,
        rowSumDevice,
        kSoftmaxCrossEntropyBackwardBatchSize,
        kSoftmaxCrossEntropyBackwardFeatureSize);
    LaunchSoftmaxCrossEntropyBackwardFp32Kernel(
        logitsDevice,
        rowMaxDevice,
        rowSumDevice,
        labelsDevice,
        gradDevice,
        kSoftmaxCrossEntropyBackwardBatchSize,
        kSoftmaxCrossEntropyBackwardFeatureSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel duration: " << elapsedMs * 1000.0f
              << " us" << std::endl;

    float* grad = new float[totalCount];
    cudaMemcpy(grad, gradDevice, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyDeviceToHost);
    const float error = ComputeRmse(gradGolden, grad, static_cast<size_t>(totalCount));
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel error: " << error << std::endl;
    const float reachedMemoryBandwidth =
        (static_cast<float>(totalCount) * sizeof(float) * 2) /
        (1024.0f * 1024.0f * 1024.0f * 1024.0f) /
        (elapsedMs / 1000.0f);
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel reached memory bandwidth: "
              << reachedMemoryBandwidth << " TB/s" << std::endl;

    cudaFree(logitsDevice);
    cudaFree(gradDevice);
    cudaFree(rowMaxDevice);
    cudaFree(rowSumDevice);
    cudaFree(labelsDevice);
    delete[] grad;
    delete[] logits;
    delete[] labels;
    delete[] gradGolden;
    return 0;
}
