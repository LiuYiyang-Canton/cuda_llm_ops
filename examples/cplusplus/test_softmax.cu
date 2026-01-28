// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for softmax kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>

constexpr int kSoftmaxBatchSize = 32;
constexpr int kSoftmaxFeatureSize = 131072;

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
 * @brief Reference CPU implementation of softmax.
 * @param input Host pointer to input.
 * @param output Host pointer to output.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void SoftmaxCpu(const float* input, float* output, int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowInput = input + static_cast<size_t>(row) * featureSize;

        float maxVal = -INFINITY;
        for (int col = 0; col < featureSize; ++col) {
            const float val = rowInput[col];
            if (val > maxVal) {
                maxVal = val;
            }
        }

        float sumExp = 0.0f;
        for (int col = 0; col < featureSize; ++col) {
            sumExp += expf(rowInput[col] - maxVal);
        }

        for (int col = 0; col < featureSize; ++col) {
            output[static_cast<size_t>(row) * featureSize + col] = expf(rowInput[col] - maxVal) / sumExp;
        }
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
    static_assert(kSoftmaxFeatureSize % 4 == 0, "feature size must be a multiple of 4");
    static_assert(kSoftmaxThreadsPerBlock % kSoftmaxWarpSize == 0,
                  "threads per block must be a multiple of warp size");

    const unsigned seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    const int totalCount = kSoftmaxBatchSize * kSoftmaxFeatureSize;
    float* x = new float[totalCount];
    float* resultGolden = new float[totalCount];

    FillRandomUniform(x, totalCount, generator, 0.0f, 1.0f);
    SoftmaxCpu(x, resultGolden, kSoftmaxBatchSize, kSoftmaxFeatureSize);

    float* xDevice = nullptr;
    float* resultDevice = nullptr;
    float* rowMaxDevice = nullptr;
    float* rowSumDevice = nullptr;
    cudaMalloc(&xDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&resultDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&rowMaxDevice, kSoftmaxBatchSize * sizeof(float));
    cudaMalloc(&rowSumDevice, kSoftmaxBatchSize * sizeof(float));
    cudaMemcpy(xDevice, x, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedMs = 0.0f;

    const int warmupIters = 1000;
    for (int i = 0; i < warmupIters; ++i) {
        LaunchSoftmaxRowMaxRowSumFp32Kernel(xDevice,
                                            rowMaxDevice,
                                            rowSumDevice,
                                            kSoftmaxBatchSize,
                                            kSoftmaxFeatureSize);
        LaunchSoftmaxFp32Kernel(xDevice,
                                rowMaxDevice,
                                rowSumDevice,
                                resultDevice,
                                kSoftmaxBatchSize,
                                kSoftmaxFeatureSize);
    }

    cudaMemset(resultDevice, 0, static_cast<size_t>(totalCount) * sizeof(float));
    cudaEventRecord(start);
    LaunchSoftmaxRowMaxRowSumFp32Kernel(xDevice,
                                        rowMaxDevice,
                                        rowSumDevice,
                                        kSoftmaxBatchSize,
                                        kSoftmaxFeatureSize);
    LaunchSoftmaxFp32Kernel(xDevice,
                            rowMaxDevice,
                            rowSumDevice,
                            resultDevice,
                            kSoftmaxBatchSize,
                            kSoftmaxFeatureSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << "SoftmaxFp32Kernel duration: " << elapsedMs * 1000.0f << " us" << std::endl;

    float* result = new float[totalCount];
    cudaMemcpy(result, resultDevice, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyDeviceToHost);
    const float error = ComputeRmse(resultGolden, result, static_cast<size_t>(totalCount));
    std::cout << "SoftmaxFp32Kernel error: " << error << std::endl;
    const float reachedMemoryBandwidth =
        (static_cast<float>(totalCount) * sizeof(float) * 2) /
        (1024.0f * 1024.0f * 1024.0f * 1024.0f) /
        (elapsedMs / 1000.0f);
    std::cout << "SoftmaxFp32Kernel reached memory bandwidth: "
              << reachedMemoryBandwidth << " TB/s" << std::endl;

    cudaFree(xDevice);
    cudaFree(resultDevice);
    cudaFree(rowMaxDevice);
    cudaFree(rowSumDevice);
    delete[] result;
    delete[] x;
    delete[] resultGolden;
    return 0;
}
