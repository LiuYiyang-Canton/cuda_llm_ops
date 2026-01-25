// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-25
// Purpose: LayerNorm backward CUDA kernel
// Build: nvcc -std=c++17 -O3 -gencode=arch=compute_120,code=sm_120 layernorm_backward.cu -o layernorm_backward
// ==============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define THREADS_PER_BLOCK 256

/**
 * @brief Reduces a pair of values within a warp using shuffle operations.
 * @param valueA First value to reduce within the warp.
 * @param valueB Second value to reduce within the warp.
 */
__device__ __forceinline__ void WarpReduceSumPair(float& valueA, float& valueB) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        valueA += __shfl_down_sync(0xffffffff, valueA, offset);
        valueB += __shfl_down_sync(0xffffffff, valueB, offset);
    }
}

/**
 * @brief Reduces a pair of values across a block using warp shuffles and shared memory.
 * @param valueA First value to reduce across the block (in/out).
 * @param valueB Second value to reduce across the block (in/out).
 */
__device__ __forceinline__ void BlockReduceSumPair(float& valueA, float& valueB) {
    __shared__ float2 warpSums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int warpId = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + warpSize - 1) / warpSize;

    WarpReduceSumPair(valueA, valueB);
    if (lane == 0) {
        warpSums[warpId] = make_float2(valueA, valueB);
    }
    __syncthreads();

    if (warpId == 0) {
        float2 warpValue = (lane < warpCount) ? warpSums[lane] : make_float2(0.0f, 0.0f);
        float reducedA = warpValue.x;
        float reducedB = warpValue.y;
        WarpReduceSumPair(reducedA, reducedB);
        if (lane == 0) {
            warpSums[0] = make_float2(reducedA, reducedB);
        }
    }
    __syncthreads();

    valueA = warpSums[0].x;
    valueB = warpSums[0].y;
}

/**
 * @brief CUDA kernel that computes LayerNorm backward, where each thread block handles one token.
 * @param gradX Output gradient with respect to x, shape (batchSize, seqLength, hiddenDim).
 * @param gradGamma Output per-token gradient contribution for gamma, shape (batchSize, seqLength, hiddenDim).
 * @param x Input from LayerNorm forward pass, shape (batchSize, seqLength, hiddenDim).
 * @param gradOutput Upstream gradient dy, shape (batchSize, seqLength, hiddenDim).
 * @param gamma LayerNorm scale parameter, shape (hiddenDim).
 * @param invStdDev Saved inverse standard deviation values from forward pass, shape (batchSize, seqLength).
 * @param mean Saved mean values from forward pass, shape (batchSize, seqLength).
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param hiddenDim Hidden dimension D (last dimension size), must be divisible by 4.
 */
__global__ void LayerNormBackwardKernel(float* __restrict__ gradX,
                                        float* __restrict__ gradGamma,
                                        const float* __restrict__ x,
                                        const float* __restrict__ gradOutput,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ invStdDev,
                                        const float* __restrict__ mean,
                                        int batchSize,
                                        int seqLength,
                                        int hiddenDim) {
    const int tokenIndex = blockIdx.x;
    if (tokenIndex >= batchSize * seqLength) {
        return;
    }
    const float invHiddenDim = 1.0f / static_cast<float>(hiddenDim);

    gradX += tokenIndex * hiddenDim;
    gradGamma += tokenIndex * hiddenDim;
    x += tokenIndex * hiddenDim;
    gradOutput += tokenIndex * hiddenDim;
    const float invStdDevValue = invStdDev[tokenIndex];
    const float meanValue = mean[tokenIndex];

    extern __shared__ float sharedData[];
    float* xHatShared = sharedData;
    float* dHatShared = sharedData + hiddenDim;

    // Recompute \hat{x},
    // and compute sum(d \hat{x}) and dot product of d \hat{x} and \hat{x} on the fly
    float sumDHatX = 0.0f;
    float dotProduct = 0.0f;

    for (int i = threadIdx.x * 4; i < hiddenDim; i += blockDim.x * 4) {
        const float4 xVal = *(reinterpret_cast<const float4*>(&x[i]));
        const float4 gradOutputVal = *(reinterpret_cast<const float4*>(&gradOutput[i]));
        const float4 gammaVal = *(reinterpret_cast<const float4*>(&gamma[i]));

        float4 xHatVal;
        xHatVal.x = (xVal.x - meanValue) * invStdDevValue;
        xHatVal.y = (xVal.y - meanValue) * invStdDevValue;
        xHatVal.z = (xVal.z - meanValue) * invStdDevValue;
        xHatVal.w = (xVal.w - meanValue) * invStdDevValue;

        float4 dHatVal;
        dHatVal.x = gradOutputVal.x * gammaVal.x;
        dHatVal.y = gradOutputVal.y * gammaVal.y;
        dHatVal.z = gradOutputVal.z * gammaVal.z;
        dHatVal.w = gradOutputVal.w * gammaVal.w;

        // gradGamma
        float4 gradGammaVal;
        gradGammaVal.x = gradOutputVal.x * xHatVal.x;
        gradGammaVal.y = gradOutputVal.y * xHatVal.y;
        gradGammaVal.z = gradOutputVal.z * xHatVal.z;
        gradGammaVal.w = gradOutputVal.w * xHatVal.w;
        *(reinterpret_cast<float4*>(&gradGamma[i])) = gradGammaVal;

        *(reinterpret_cast<float4*>(&xHatShared[i])) = xHatVal;
        *(reinterpret_cast<float4*>(&dHatShared[i])) = dHatVal;

        sumDHatX += dHatVal.x + dHatVal.y + dHatVal.z + dHatVal.w;
        dotProduct += dHatVal.x * xHatVal.x + dHatVal.y * xHatVal.y +
                      dHatVal.z * xHatVal.z + dHatVal.w * xHatVal.w;
    }

    // Reduce within block
    BlockReduceSumPair(sumDHatX, dotProduct);
    const float meanDhat = sumDHatX * invHiddenDim;
    const float meanDhatXhat = dotProduct * invHiddenDim;

    for (int i = threadIdx.x * 4; i < hiddenDim; i += blockDim.x * 4) {
        const float4 xHatVal = *(reinterpret_cast<const float4*>(&xHatShared[i]));
        const float4 dHatVal = *(reinterpret_cast<const float4*>(&dHatShared[i]));

        float4 gradXVal;
        gradXVal.x = (dHatVal.x - meanDhat - xHatVal.x * meanDhatXhat) * invStdDevValue;
        gradXVal.y = (dHatVal.y - meanDhat - xHatVal.y * meanDhatXhat) * invStdDevValue;
        gradXVal.z = (dHatVal.z - meanDhat - xHatVal.z * meanDhatXhat) * invStdDevValue;
        gradXVal.w = (dHatVal.w - meanDhat - xHatVal.w * meanDhatXhat) * invStdDevValue;
        *(reinterpret_cast<float4*>(&gradX[i])) = gradXVal;
    }
}

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
        const float* xToken = x + token * hiddenDim;
        const float* gradOutputToken = gradOutput + token * hiddenDim;
        float* gradXToken = gradX + token * hiddenDim;
        float* gradGammaToken = gradGamma + token * hiddenDim;

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
        const float* xToken = x + token * hiddenDim;
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
    const int kBatchSize = 1;
    const int kSeqLength = 128 * 1024;
    const int kHiddenDim = 2048;
    const float kEpsilon = 1.0e-6f;
    assert((kHiddenDim % 4) == 0);

    const size_t tokenCount = static_cast<size_t>(kBatchSize) * static_cast<size_t>(kSeqLength);
    const size_t elementCount = tokenCount * static_cast<size_t>(kHiddenDim);
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
    float* gamma = new float[kHiddenDim];

    std::mt19937 generator(1234);
    FillRandomUniform(x, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gradOutput, elementCount, generator, -1.0f, 1.0f);
    FillRandomUniform(gamma, kHiddenDim, generator, -1.0f, 1.0f);

    ComputeMeanAndInvStdDevFromInput(mean, invStdDev, x, kBatchSize, kSeqLength, kHiddenDim, kEpsilon);
    LayerNormBackwardCpu(gradX,
                         gradGamma,
                         x,
                         gradOutput,
                         gamma,
                         invStdDev,
                         mean,
                         kBatchSize,
                         kSeqLength,
                         kHiddenDim);

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
    cudaMalloc(reinterpret_cast<void**>(&gammaDevice), kHiddenDim * sizeof(float));

    cudaMemcpy(xDevice, x, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gradOutputDevice, gradOutput, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(invStdDevDevice, invStdDev, tokenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(meanDevice, mean, tokenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gammaDevice, gamma, kHiddenDim * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "LayerNormBackwardCpu completed." << std::endl;

    const dim3 grid(static_cast<unsigned int>(tokenCount));
    const dim3 block(THREADS_PER_BLOCK);
    const size_t sharedMemBytes = static_cast<size_t>(kHiddenDim) * 2 * sizeof(float);
    const int kWarmupRuns = 20;
    const int kTimedRuns = 20;
    for (int i = 0; i < kWarmupRuns; ++i) {
        LayerNormBackwardKernel<<<grid, block, sharedMemBytes>>>(gradXDevice,
                                                 gradGammaDevice,
                                                 xDevice,
                                                 gradOutputDevice,
                                                 gammaDevice,
                                                 invStdDevDevice,
                                                 meanDevice,
                                                 kBatchSize,
                                                 kSeqLength,
                                                 kHiddenDim);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < kTimedRuns; ++i) {
        LayerNormBackwardKernel<<<grid, block, sharedMemBytes>>>(gradXDevice,
                                                 gradGammaDevice,
                                                 xDevice,
                                                 gradOutputDevice,
                                                 gammaDevice,
                                                 invStdDevDevice,
                                                 meanDevice,
                                                 kBatchSize,
                                                 kSeqLength,
                                                 kHiddenDim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgUs = (elapsedMs * 1000.0f) / static_cast<float>(kTimedRuns);
    std::cout << "LayerNormBackwardKernel avg time: " << avgUs << " us" << std::endl;
    const double bytesPerRun = static_cast<double>(elementBytes) * 4.0;
    const double totalBytes = bytesPerRun * static_cast<double>(kTimedRuns);
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
