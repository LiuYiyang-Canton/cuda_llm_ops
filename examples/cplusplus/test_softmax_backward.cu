// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for softmax backward kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cstddef>
#include <cmath>
#include <iostream>
#include <random>

constexpr int kSoftmaxBackwardBatchSize = 32;
constexpr int kSoftmaxBackwardFeatureSize = 131072;

/**
 * @brief Reference CPU implementation for softmax backward.
 * @param gradOutput Host pointer to upstream gradient.
 * @param softmaxOutput Host pointer to softmax output.
 * @param gradX Host pointer to output gradient.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void SoftmaxBackwardCpu(const float* gradOutput, const float* softmaxOutput, float* gradX,
                        int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowGradOutput = gradOutput + static_cast<size_t>(row) * featureSize;
        const float* rowSoftmaxOutput = softmaxOutput + static_cast<size_t>(row) * featureSize;
        float* rowGradX = gradX + static_cast<size_t>(row) * featureSize;

        float dot = 0.0f;
        for (int col = 0; col < featureSize; ++col) {
            dot += rowSoftmaxOutput[col] * rowGradOutput[col];
        }

        for (int col = 0; col < featureSize; ++col) {
            rowGradX[col] = rowSoftmaxOutput[col] * (rowGradOutput[col] - dot);
        }
    }
}

/**
 * @brief Fills a buffer with random normal values.
 * @param data Host pointer to output buffer.
 * @param count Number of elements to fill.
 * @param generator RNG generator.
 * @param mean Normal distribution mean.
 * @param stddev Normal distribution standard deviation.
 */
void FillRandomNormal(float* data, int count, std::mt19937& generator, float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int i = 0; i < count; ++i) {
        data[i] = distribution(generator);
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
 * @brief Entry point: allocates data, runs CPU reference and GPU kernels, and times the result.
 * @return Exit code (0 on success).
 */
int main() {
    static_assert(kSoftmaxBackwardFeatureSize % 4 == 0, "feature size must be divisible by 4");

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    const int totalCount = kSoftmaxBackwardBatchSize * kSoftmaxBackwardFeatureSize;
    float* gradOutput = new float[totalCount];
    float* softmaxOutput = new float[totalCount];
    float* gradXGolden = new float[totalCount];
    float* gradX = new float[totalCount];

    FillRandomNormal(gradOutput, totalCount, generator, 0.0f, 1.0f);
    FillRandomNormal(softmaxOutput, totalCount, generator, 0.0f, 1.0f);

    SoftmaxBackwardCpu(gradOutput, softmaxOutput, gradXGolden,
                       kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);

    float* gradOutputDevice = nullptr;
    float* softmaxOutputDevice = nullptr;
    float* gradXDevice = nullptr;
    float* dotProductDevice = nullptr;
    cudaMalloc(&gradOutputDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&softmaxOutputDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&gradXDevice, static_cast<size_t>(totalCount) * sizeof(float));
    cudaMalloc(&dotProductDevice, kSoftmaxBackwardBatchSize * sizeof(float));
    cudaMemcpy(gradOutputDevice, gradOutput, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(softmaxOutputDevice, softmaxOutput, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dotProductDevice, 0, kSoftmaxBackwardBatchSize * sizeof(float));

    LaunchSoftmaxBackwardDotKernel(gradOutputDevice, softmaxOutputDevice,
                                   gradXDevice, dotProductDevice,
                                   kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);
    LaunchSoftmaxBackwardGradKernel(softmaxOutputDevice, dotProductDevice,
                                    gradXDevice, kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);

    cudaMemcpy(gradX, gradXDevice, static_cast<size_t>(totalCount) * sizeof(float), cudaMemcpyDeviceToHost);

    const float rmse = ComputeRmse(gradXGolden, gradX, static_cast<size_t>(totalCount));
    std::cout << "RMSE: " << rmse << std::endl;

    const int warmupIters = 10;
    const int timedIters = 10;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmupIters; ++i) {
        LaunchSoftmaxBackwardDotKernel(gradOutputDevice, softmaxOutputDevice,
                                       gradXDevice, dotProductDevice,
                                       kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);
        LaunchSoftmaxBackwardGradKernel(softmaxOutputDevice, dotProductDevice,
                                        gradXDevice, kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < timedIters; ++i) {
        LaunchSoftmaxBackwardDotKernel(gradOutputDevice, softmaxOutputDevice,
                                       gradXDevice, dotProductDevice,
                                       kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);
        LaunchSoftmaxBackwardGradKernel(softmaxOutputDevice, dotProductDevice,
                                        gradXDevice, kSoftmaxBackwardBatchSize, kSoftmaxBackwardFeatureSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgMs = elapsedMs / timedIters;
    std::cout << "SoftmaxBackward kernels avg: " << avgMs * 1000.0f
              << " us (" << timedIters << " iters)" << std::endl;

    const size_t numElements = static_cast<size_t>(kSoftmaxBackwardBatchSize) * kSoftmaxBackwardFeatureSize;
    const double bytesPerIter = 3.0 * static_cast<double>(numElements) * sizeof(float);
    const double bandwidthGBs = bytesPerIter / (avgMs / 1000.0) /
                                (1024.0 * 1024.0 * 1024.0);
    std::cout << "SoftmaxBackward effective bandwidth (est.): "
              << bandwidthGBs << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] gradOutput;
    delete[] softmaxOutput;
    delete[] gradX;
    delete[] gradXGolden;

    cudaFree(gradOutputDevice);
    cudaFree(softmaxOutputDevice);
    cudaFree(gradXDevice);
    cudaFree(dotProductDevice);
    return 0;
}
