// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for LayerNorm backward kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <random>

/**
 * @brief Computes LayerNorm backward on CPU for a batched sequence tensor.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from LayerNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma LayerNorm scale parameter, shape (hiddenDim).
 * @param invStdDev Saved inverse standard deviation values from forward pass, shape (batchSize, seqLength).
 * @param mean Saved mean values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size).
 */
void LayerNormBackwardCpu(float* gradX,
                          float* gradGamma,
                          const float* x,
                          const float* gradOutput,
                          const float* gamma,
                          const float* invStdDev,
                          const float* mean,
                          int batchSize,
                          int seqLength,
                          int hiddenDim) {
    assert(gradX != nullptr);
    assert(gradGamma != nullptr);
    assert(x != nullptr);
    assert(gradOutput != nullptr);
    assert(gamma != nullptr);
    assert(invStdDev != nullptr);
    assert(mean != nullptr);
    assert(batchSize > 0);
    assert(seqLength > 0);
    assert(hiddenDim > 0);

    const int tokenCount = batchSize * seqLength;
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);

    for (int token = 0; token < tokenCount; ++token) {
        const float* xToken = x + static_cast<size_t>(token) * hiddenDim;
        const float* gradOutputToken = gradOutput + static_cast<size_t>(token) * hiddenDim;
        float* gradXToken = gradX + static_cast<size_t>(token) * hiddenDim;
        float* gradGammaToken = gradGamma + static_cast<size_t>(token) * hiddenDim;

        const float meanValue = mean[token];
        const float invStdDevValue = invStdDev[token];
        float sumDhat = 0.0f;
        float sumDhatXhat = 0.0f;
        for (int i = 0; i < hiddenDim; ++i) {
            const float xHat = (xToken[i] - meanValue) * invStdDevValue;
            const float dHat = gradOutputToken[i] * gamma[i];
            sumDhat += dHat;
            sumDhatXhat += dHat * xHat;
        }

        const float meanDhat = sumDhat * invHiddenDim;
        const float meanDhatXhat = sumDhatXhat * invHiddenDim;
        for (int i = 0; i < hiddenDim; ++i) {
            const float xHat = (xToken[i] - meanValue) * invStdDevValue;
            const float dHat = gradOutputToken[i] * gamma[i];
            gradXToken[i] = (dHat - meanDhat - xHat * meanDhatXhat) * invStdDevValue;
            gradGammaToken[i] = gradOutputToken[i] * xHat;
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
 * @brief Computes the root mean square error between two buffers.
 * @param reference Reference buffer.
 * @param actual Buffer to compare against the reference.
 * @param count Number of elements to compare.
 * @return Root mean square error value.
 */
float ComputeRmse(const float* reference, const float* actual, size_t count) {
    assert(reference != nullptr);
    assert(actual != nullptr);
    assert(count > 0);

    double sumSquared = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(reference[i]) - static_cast<double>(actual[i]);
        sumSquared += diff * diff;
    }
    return static_cast<float>(std::sqrt(sumSquared / static_cast<double>(count)));
}

/**
 * @brief Computes per-token mean and inverse standard deviation values for LayerNorm forward.
 * @param mean Output mean buffer of shape (batchSize, seqLength).
 * @param invStdDev Output inverse standard deviation buffer of shape (batchSize, seqLength).
 * @param x Input tensor of shape (batchSize, seqLength, hiddenDim).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D.
 * @param epsilon Epsilon added inside the square root.
 */
void ComputeMeanAndInvStdDevFromInput(float* mean,
                                      float* invStdDev,
                                      const float* x,
                                      int batchSize,
                                      int seqLength,
                                      int hiddenDim,
                                      float epsilon) {
    assert(mean != nullptr);
    assert(invStdDev != nullptr);
    assert(x != nullptr);
    assert(batchSize > 0);
    assert(seqLength > 0);
    assert(hiddenDim > 0);

    const int tokenCount = batchSize * seqLength;
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);
    for (int token = 0; token < tokenCount; ++token) {
        const float* xToken = x + static_cast<size_t>(token) * hiddenDim;
        float meanValue = 0.0f;
        for (int i = 0; i < hiddenDim; ++i) {
            meanValue += xToken[i];
        }
        meanValue *= invHiddenDim;
        mean[token] = meanValue;

        float variance = 0.0f;
        for (int i = 0; i < hiddenDim; ++i) {
            const float diff = xToken[i] - meanValue;
            variance += diff * diff;
        }
        invStdDev[token] = 1.0f / std::sqrt(variance * invHiddenDim + epsilon);
    }
}

/**
 * @brief Entry point that allocates buffers, runs the LayerNorm backward CPU reference, and reports status.
 * @return 0 on success, non-zero on failure.
 */
int main() {
    const int batchSize = 1;
    const int seqLength = 128 * 1024;
    const int hiddenDim = 2048;
    const float epsilon = 1.0e-6f;
    assert((hiddenDim % 4) == 0);

    const size_t tokenCount = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength);
    const size_t elementCount = tokenCount * static_cast<size_t>(hiddenDim);
    const size_t elementBytes = elementCount * sizeof(float);
    const size_t tokenBytes = tokenCount * sizeof(float);

    float* x = new float[elementCount];
    float* gradOutput = new float[elementCount];
    float* gradX = new float[elementCount];
    float* gradGamma = new float[elementCount];
    float* gradXGpu = new float[elementCount];
    float* gradGammaGpu = new float[elementCount];
    float* invStdDev = new float[tokenCount];
    float* mean = new float[tokenCount];
    float* gamma = new float[hiddenDim];

    std::mt19937 generator(1234);
    FillRandomUniform(x, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gradOutput, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gamma, hiddenDim, generator, -1.0f, 1.0f);

    ComputeMeanAndInvStdDevFromInput(mean, invStdDev, x, batchSize, seqLength, hiddenDim, epsilon);
    LayerNormBackwardCpu(gradX,
                         gradGamma,
                         x,
                         gradOutput,
                         gamma,
                         invStdDev,
                         mean,
                         batchSize,
                         seqLength,
                         hiddenDim);

    float* xDevice = nullptr;
    float* gradOutputDevice = nullptr;
    float* gradXDevice = nullptr;
    float* gradGammaDevice = nullptr;
    float* invStdDevDevice = nullptr;
    float* meanDevice = nullptr;
    float* gammaDevice = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&xDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradOutputDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradXDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&gradGammaDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&invStdDevDevice), tokenBytes);
    cudaMalloc(reinterpret_cast<void**>(&meanDevice), tokenBytes);
    cudaMalloc(reinterpret_cast<void**>(&gammaDevice), hiddenDim * sizeof(float));

    cudaMemcpy(xDevice, x, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gradOutputDevice, gradOutput, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(invStdDevDevice, invStdDev, tokenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(meanDevice, mean, tokenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gammaDevice, gamma, hiddenDim * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "LayerNormBackwardCpu completed." << std::endl;

    const int warmupRuns = 20;
    const int timedRuns = 20;
    for (int i = 0; i < warmupRuns; ++i) {
        LaunchLayerNormBackwardKernel(gradXDevice,
                                      gradGammaDevice,
                                      xDevice,
                                      gradOutputDevice,
                                      gammaDevice,
                                      invStdDevDevice,
                                      meanDevice,
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
        LaunchLayerNormBackwardKernel(gradXDevice,
                                      gradGammaDevice,
                                      xDevice,
                                      gradOutputDevice,
                                      gammaDevice,
                                      invStdDevDevice,
                                      meanDevice,
                                      batchSize,
                                      seqLength,
                                      hiddenDim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgUs = (elapsedMs * 1000.0f) / static_cast<float>(timedRuns);
    std::cout << "LayerNormBackwardKernel avg time: " << avgUs << " us" << std::endl;
    const double bytesPerRun = static_cast<double>(elementBytes) * 4.0;
    const double totalBytes = bytesPerRun * static_cast<double>(timedRuns);
    const double totalSeconds = static_cast<double>(elapsedMs) / 1000.0;
    const double bandwidthGBs = (totalBytes / totalSeconds) / 1.0e9;
    std::cout << "LayerNormBackwardKernel effective bandwidth: " << bandwidthGBs << " GB/s"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(gradXGpu, gradXDevice, elementBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradGammaGpu, gradGammaDevice, elementBytes, cudaMemcpyDeviceToHost);

    const float rmseGradX = ComputeRmse(gradX, gradXGpu, elementCount);
    const float rmseGradGamma = ComputeRmse(gradGamma, gradGammaGpu, elementCount);
    std::cout << "RMSE gradX: " << rmseGradX << ", gradGamma: " << rmseGradGamma << std::endl;

    cudaFree(xDevice);
    cudaFree(gradOutputDevice);
    cudaFree(gradXDevice);
    cudaFree(gradGammaDevice);
    cudaFree(invStdDevDevice);
    cudaFree(meanDevice);
    cudaFree(gammaDevice);

    delete[] x;
    delete[] gradOutput;
    delete[] gradX;
    delete[] gradGamma;
    delete[] gradXGpu;
    delete[] gradGammaGpu;
    delete[] invStdDev;
    delete[] mean;
    delete[] gamma;
    return 0;
}
