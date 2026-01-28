// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for softmax cross-entropy kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

constexpr int kSoftmaxCrossEntropyBatchSize = 32;
constexpr int kSoftmaxCrossEntropyFeatureSize = 131072;

/**
 * @brief Reference CPU implementation for softmax cross-entropy.
 * @param logits Host pointer to input logits.
 * @param labels Host pointer to label indices, one element per batch.
 * @param loss Host pointer to per-row loss.
 * @param batchSize Number of rows.
 * @param featureSize Number of columns.
 */
void SoftmaxCrossEntropyCpu(const float* logits, const int* labels,
                            float* loss, int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowLogits = logits + static_cast<size_t>(row) * featureSize;

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

        const int label = labels[row];
        loss[row] = -logits[static_cast<size_t>(row) * featureSize + label] + logf(sumExp) + maxVal;
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
 * @brief Checks a CUDA API call and aborts on failure.
 * @param result CUDA status returned by the call.
 * @param context String describing the call for error reporting.
 */
void CheckCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Entry point: allocates data, runs CPU reference and GPU kernel, and times the result.
 * @return Exit code (0 on success).
 */
int main() {
    static_assert(kSoftmaxCrossEntropyFeatureSize % 4 == 0, "feature size must be a multiple of 4");

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    const int totalCount = kSoftmaxCrossEntropyBatchSize * kSoftmaxCrossEntropyFeatureSize;
    float* logits = new float[totalCount];
    int* labels = new int[kSoftmaxCrossEntropyBatchSize];
    float* lossGolden = new float[kSoftmaxCrossEntropyBatchSize];
    float* loss = new float[kSoftmaxCrossEntropyBatchSize];

    FillRandomNormal(logits, totalCount, generator, 0.0f, 1.0f);
    FillRandomLabels(labels, kSoftmaxCrossEntropyBatchSize, generator, kSoftmaxCrossEntropyFeatureSize);

    SoftmaxCrossEntropyCpu(logits, labels, lossGolden,
                           kSoftmaxCrossEntropyBatchSize, kSoftmaxCrossEntropyFeatureSize);

    float* logitsDevice = nullptr;
    int* labelsDevice = nullptr;
    float* lossDevice = nullptr;
    CheckCuda(cudaMalloc(&logitsDevice, static_cast<size_t>(totalCount) * sizeof(float)), "cudaMalloc(logitsDevice)");
    CheckCuda(cudaMalloc(&labelsDevice, kSoftmaxCrossEntropyBatchSize * sizeof(int)), "cudaMalloc(labelsDevice)");
    CheckCuda(cudaMalloc(&lossDevice, kSoftmaxCrossEntropyBatchSize * sizeof(float)), "cudaMalloc(lossDevice)");
    CheckCuda(cudaMemcpy(logitsDevice, logits, static_cast<size_t>(totalCount) * sizeof(float),
                          cudaMemcpyHostToDevice),
              "cudaMemcpy(logitsDevice)");
    CheckCuda(cudaMemcpy(labelsDevice, labels, kSoftmaxCrossEntropyBatchSize * sizeof(int),
                          cudaMemcpyHostToDevice),
              "cudaMemcpy(labelsDevice)");
    CheckCuda(cudaMemset(lossDevice, 0, kSoftmaxCrossEntropyBatchSize * sizeof(float)), "cudaMemset(lossDevice)");

    LaunchSoftmaxCrossEntropyKernel(logitsDevice, labelsDevice, lossDevice,
                                    kSoftmaxCrossEntropyBatchSize, kSoftmaxCrossEntropyFeatureSize);
    CheckCuda(cudaGetLastError(), "SoftmaxCrossEntropyKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

    CheckCuda(cudaMemcpy(loss, lossDevice, kSoftmaxCrossEntropyBatchSize * sizeof(float),
                          cudaMemcpyDeviceToHost),
              "cudaMemcpy(loss)");
    std::cout << "Copied loss back to host\n";

    const float lossRmse = ComputeRmse(lossGolden, loss, kSoftmaxCrossEntropyBatchSize);
    std::cout << "Loss RMSE: " << lossRmse << std::endl;

    const int timedIters = 1;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");


    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < timedIters; ++i) {
        LaunchSoftmaxCrossEntropyKernel(logitsDevice, labelsDevice, lossDevice,
                                        kSoftmaxCrossEntropyBatchSize, kSoftmaxCrossEntropyFeatureSize);
        CheckCuda(cudaGetLastError(), "SoftmaxCrossEntropyKernel timed launch");
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "cudaEventElapsedTime");
    const float avgMs = elapsedMs / timedIters;
    std::cout << "SoftmaxCrossEntropy kernel avg: " << avgMs * 1000.0f
              << " us (" << timedIters << " iters)" << std::endl;

    const size_t numElements = static_cast<size_t>(kSoftmaxCrossEntropyBatchSize) * kSoftmaxCrossEntropyFeatureSize;
    const double bytesPerIter = static_cast<double>(numElements) * sizeof(float);
    const double bandwidthGBs = bytesPerIter / (avgMs / 1000.0) /
                                (1024.0 * 1024.0 * 1024.0);
    std::cout << "SoftmaxCrossEntropy effective bandwidth (est.): "
              << bandwidthGBs << " GB/s" << std::endl;

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    delete[] logits;
    delete[] labels;
    delete[] lossGolden;
    delete[] loss;

    CheckCuda(cudaFree(logitsDevice), "cudaFree(logitsDevice)");
    CheckCuda(cudaFree(labelsDevice), "cudaFree(labelsDevice)");
    CheckCuda(cudaFree(lossDevice), "cudaFree(lossDevice)");
    return 0;
}
