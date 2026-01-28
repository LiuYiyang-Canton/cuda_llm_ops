// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for RMSNorm backward kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <random>

/**
 * @brief Computes RMSNorm backward on CPU for a batched sequence tensor.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from RMSNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma RMSNorm scale parameter, shape (hiddenDim).
 * @param invRms Saved inverse RMS values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size).
 */
void RMSNormBackwardCpu(float* gradX,
                        float* gradGamma,
                        const float* x,
                        const float* gradOutput,
                        const float* gamma,
                        const float* invRms,
                        int batchSize,
                        int seqLength,
                        int hiddenDim) {
    assert(gradX != nullptr);
    assert(gradGamma != nullptr);
    assert(x != nullptr);
    assert(gradOutput != nullptr);
    assert(gamma != nullptr);
    assert(invRms != nullptr);
    assert(batchSize > 0);
    assert(seqLength > 0);
    assert(hiddenDim > 0);
    assert((hiddenDim % 4) == 0);

    const int tokenCount = batchSize * seqLength;
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);

    for (int token = 0; token < tokenCount; ++token) {
        const float* xToken = x + static_cast<size_t>(token) * hiddenDim;
        const float* gradOutputToken = gradOutput + static_cast<size_t>(token) * hiddenDim;
        float* gradXToken = gradX + static_cast<size_t>(token) * hiddenDim;
        float* gradGammaToken = gradGamma + static_cast<size_t>(token) * hiddenDim;

        float dot = 0.0f;
        for (int i = 0; i < hiddenDim; ++i) {
            const float xVal = xToken[i];
            const float gradVal = gradOutputToken[i] * gamma[i];
            dot += gradVal * xVal;
        }

        const float invRmsValue = invRms[token];

        const float scale = dot * invRmsValue * invRmsValue * invHiddenDim;
        for (int i = 0; i < hiddenDim; ++i) {
            const float xVal = xToken[i];
            const float gradVal = gradOutputToken[i] * gamma[i];
            gradXToken[i] = (gradVal - xVal * scale) * invRmsValue;
            gradGammaToken[i] = gradOutputToken[i] * xVal * invRmsValue;
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
 * @brief Computes per-token inverse RMS values for RMSNorm forward.
 * @param invRms Output inverse RMS buffer of shape (batchSize, seqLength).
 * @param x Input tensor of shape (batchSize, seqLength, hiddenDim).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D.
 * @param epsilon Epsilon added inside the square root.
 */
void ComputeInvRmsFromInput(float* invRms,
                            const float* x,
                            int batchSize,
                            int seqLength,
                            int hiddenDim,
                            float epsilon) {
    assert(invRms != nullptr);
    assert(x != nullptr);
    assert(batchSize > 0);
    assert(seqLength > 0);
    assert(hiddenDim > 0);

    const int tokenCount = batchSize * seqLength;
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);
    for (int token = 0; token < tokenCount; ++token) {
        const float* xToken = x + static_cast<size_t>(token) * hiddenDim;
        float sumSq = 0.0f;
        for (int i = 0; i < hiddenDim; ++i) {
            const float xVal = xToken[i];
            sumSq += xVal * xVal;
        }
        invRms[token] = 1.0f / std::sqrt(sumSq * invHiddenDim + epsilon);
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
 * @brief Entry point that allocates buffers, runs the RMSNorm backward CPU reference, and reports status.
 * @return 0 on success, non-zero on failure.
 */
int main() {
    const int batchSize = 1;
    const int seqLength = 128 * 1024;
    const int hiddenDim = 2048;
    const float epsilon = 1.0e-6f;

    const size_t tokenCount = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength);
    const size_t elementCount = tokenCount * static_cast<size_t>(hiddenDim);
    assert((hiddenDim % 4) == 0);

    const size_t elementBytes = elementCount * sizeof(float);
    const size_t tokenBytes = tokenCount * sizeof(float);

    float* x = new float[elementCount];
    float* gradOutput = new float[elementCount];
    float* gradXGolden = new float[elementCount];
    float* gradXGpu = new float[elementCount];
    float* gradGammaGolden = new float[elementCount];
    float* gradGammaGpu = new float[elementCount];
    float* invRms = new float[tokenCount];
    float* gamma = new float[hiddenDim];

    std::mt19937 generator(1234);
    FillRandomUniform(x, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gradOutput, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gamma, hiddenDim, generator, 0.0f, 1.0f);

    ComputeInvRmsFromInput(invRms, x, batchSize, seqLength, hiddenDim, epsilon);
    RMSNormBackwardCpu(gradXGolden,
                       gradGammaGolden,
                       x,
                       gradOutput,
                       gamma,
                       invRms,
                       batchSize,
                       seqLength,
                       hiddenDim);

    float* xDevice = nullptr;
    float* gradOutputDevice = nullptr;
    float* gradXDevice = nullptr;
    float* gradGammaDevice = nullptr;
    float* invRmsDevice = nullptr;
    float* gammaDevice = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&xDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradOutputDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradXDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradGammaDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&invRmsDevice), tokenBytes);
    cudaMalloc(reinterpret_cast<void**>(&gammaDevice), hiddenDim * sizeof(float));

    cudaMemcpy(xDevice, x, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gradOutputDevice, gradOutput, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(invRmsDevice, invRms, tokenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gammaDevice, gamma, hiddenDim * sizeof(float), cudaMemcpyHostToDevice);

    const int warmupRuns = 20;
    const int timedRuns = 20;
    for (int i = 0; i < warmupRuns; ++i) {
        LaunchRmsNormBackwardKernel(gradXDevice,
                                    gradGammaDevice,
                                    xDevice,
                                    gradOutputDevice,
                                    gammaDevice,
                                    invRmsDevice,
                                    batchSize,
                                    seqLength,
                                    hiddenDim);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < timedRuns; ++i) {
        LaunchRmsNormBackwardKernel(gradXDevice,
                                    gradGammaDevice,
                                    xDevice,
                                    gradOutputDevice,
                                    gammaDevice,
                                    invRmsDevice,
                                    batchSize,
                                    seqLength,
                                    hiddenDim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgUs = (elapsedMs * 1000.0f) / static_cast<float>(timedRuns);
    std::cout << "RMSNormBackwardKernel avg time: " << avgUs << " us" << std::endl;
    const double bytesPerRun = static_cast<double>(elementBytes) * 4.0 +
                               static_cast<double>(tokenBytes);
    const double totalBytes = bytesPerRun * static_cast<double>(timedRuns);
    const double totalSeconds = static_cast<double>(elapsedMs) / 1000.0;
    const double bandwidthGBs = (totalBytes / totalSeconds) / 1.0e9;
    std::cout << "RMSNormBackwardKernel effective bandwidth: " << bandwidthGBs << " GB/s"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(gradXGpu, gradXDevice, elementBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradGammaGpu, gradGammaDevice, elementBytes, cudaMemcpyDeviceToHost);
    const float rmse = ComputeRmse(gradXGolden, gradXGpu, elementCount);
    const float rmseGamma = ComputeRmse(gradGammaGolden, gradGammaGpu, elementCount);
    std::cout << "RMSNormBackward RMSE: " << rmse << std::endl;
    std::cout << "RMSNormBackward dGamma RMSE: " << rmseGamma << std::endl;

    cudaFree(xDevice);
    cudaFree(gradOutputDevice);
    cudaFree(gradXDevice);
    cudaFree(gradGammaDevice);
    cudaFree(invRmsDevice);
    cudaFree(gammaDevice);
    delete[] x;
    delete[] gradOutput;
    delete[] gradXGolden;
    delete[] gradXGpu;
    delete[] gradGammaGolden;
    delete[] gradGammaGpu;
    delete[] invRms;
    delete[] gamma;
    return 0;
}
