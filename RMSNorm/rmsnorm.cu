// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-25
// Purpose: CUDA kernel and CPU reference for RMSNorm forward.
// Build:  nvcc -gencode=arch=compute_120,code=sm_120 -O3 -o rmsnorm rmsnorm.cu
// ==============================================================================

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <random>

namespace cg = cooperative_groups;

constexpr int kSeqLength = 4096;
constexpr int kHiddenSize = 7168;
constexpr float kEpsilon = 1.0e-6f;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kBlockNumWarps = kThreadsPerBlock / kWarpSize;

/**
 * @brief Computes RMSNorm forward for a batch, one hidden-dimension row per block.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param rmsnormOut Output tensor with shape (seqLength, hiddenSize).
 */
__global__ void RMSNormFp32Kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  float* __restrict__ rmsnormOut) {
    auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    const int threadId = threadIdx.x;
    const int laneId = threadId % kWarpSize;
    __shared__ float warpSum[kBlockNumWarps];

    const int rowIndex = blockIdx.x;
    const int rowOffset = rowIndex * kHiddenSize;
    x += rowOffset;

    float sumSq = 0.0f;

#pragma unroll
    for (int i = threadId * 4; i < kHiddenSize; i += blockDim.x * 4) {
        if (i + 3 < kHiddenSize) {
            const float4 temp = *(reinterpret_cast<const float4*>(&x[i]));
            sumSq += temp.x * temp.x + temp.y * temp.y + temp.z * temp.z + temp.w * temp.w;
        } else {
            for (int j = i; j < kHiddenSize; ++j) {
                const float val = x[j];
                sumSq += val * val;
            }
        }
    }

    sumSq = cg::reduce(warp, sumSq, cg::plus<float>());
    if (laneId == 0) {
        warpSum[threadId / kWarpSize] = sumSq;
    }
    __syncthreads();

    for (int offset = 0; offset < kBlockNumWarps; ++offset) {
        if (threadId < offset) {
            warpSum[threadId] += warpSum[offset];
        }
        __syncthreads();
    }

    const float invRms = rsqrtf(warpSum[0] / kHiddenSize + kEpsilon);

#pragma unroll
    for (int i = threadId * 4; i < kHiddenSize; i += blockDim.x * 4) {
        if (i + 3 < kHiddenSize) {
            float4 val = *(reinterpret_cast<const float4*>(&x[i]));
            const float4 w = *(reinterpret_cast<const float4*>(&weight[i]));
            val.x = val.x * invRms * w.x;
            val.y = val.y * invRms * w.y;
            val.z = val.z * invRms * w.z;
            val.w = val.w * invRms * w.w;
            *(reinterpret_cast<float4*>(&rmsnormOut[rowOffset + i])) = val;
        } else {
            for (int j = i; j < kHiddenSize; ++j) {
                rmsnormOut[rowOffset + j] = x[j] * invRms * weight[j];
            }
        }
    }
}

/**
 * @brief Computes RMSNorm forward on CPU for validation.
 * @param x Input tensor with shape (seqLength, hiddenSize).
 * @param weight Weight vector with shape (hiddenSize).
 * @param rmsnormOut Output tensor with shape (seqLength, hiddenSize).
 * @param seqLength Number of rows in the input.
 * @param hiddenSize Hidden dimension size.
 * @param epsilon Epsilon added to the mean square before sqrt.
 */
void RMSNormForwardCpu(const float* x,
                       const float* weight,
                       float* rmsnormOut,
                       int seqLength,
                       int hiddenSize,
                       float epsilon) {
    const float invHiddenSize = 1.0f / static_cast<float>(hiddenSize);
    for (int row = 0; row < seqLength; ++row) {
        const float* rowPtr = x + row * hiddenSize;
        float* outPtr = rmsnormOut + row * hiddenSize;
        float sumSq = 0.0f;
        for (int col = 0; col < hiddenSize; ++col) {
            const float val = rowPtr[col];
            sumSq += val * val;
        }
        const float invRms = 1.0f / std::sqrt(sumSq * invHiddenSize + epsilon);
        for (int col = 0; col < hiddenSize; ++col) {
            outPtr[col] = rowPtr[col] * invRms * weight[col];
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
 * @brief Entry point that runs RMSNorm forward on CPU/GPU and reports metrics.
 * @return 0 on success, non-zero on failure.
 */
int main() {
    assert((kHiddenSize % 4) == 0);

    const size_t elementCount = static_cast<size_t>(kSeqLength) * static_cast<size_t>(kHiddenSize);
    const size_t elementBytes = elementCount * sizeof(float);

    float* x = new float[elementCount];
    float* weight = new float[kHiddenSize];
    float* rmsnormGolden = new float[elementCount];
    float* rmsnormOut = new float[elementCount];

    const unsigned int seed =
        static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    FillRandomUniform(x, elementCount, generator, 0.0f, 1.0f);
    FillRandomUniform(weight, kHiddenSize, generator, 0.0f, 1.0f);

    RMSNormForwardCpu(x, weight, rmsnormGolden, kSeqLength, kHiddenSize, kEpsilon);

    float* xDevice = nullptr;
    float* weightDevice = nullptr;
    float* rmsnormOutDevice = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&xDevice), elementBytes);
    cudaMalloc(reinterpret_cast<void**>(&weightDevice), kHiddenSize * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&rmsnormOutDevice), elementBytes);
    cudaMemcpy(xDevice, x, elementBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(weightDevice, weight, kHiddenSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const dim3 numThreads(kThreadsPerBlock);
    const dim3 numBlocks(kSeqLength);

    const int kWarmupRuns = 1000;
    const int kTimedRuns = 1;
    for (int i = 0; i < kWarmupRuns; ++i) {
        RMSNormFp32Kernel<<<numBlocks, numThreads>>>(xDevice, weightDevice, rmsnormOutDevice);
    }

    cudaEventRecord(start);
    for (int i = 0; i < kTimedRuns; ++i) {
        RMSNormFp32Kernel<<<numBlocks, numThreads>>>(xDevice, weightDevice, rmsnormOutDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    const float avgUs = (elapsedMs * 1000.0f) / static_cast<float>(kTimedRuns);
    std::cout << "RMSNormFp32Kernel avg time: " << avgUs << " us" << std::endl;
    const double reachedMemBwTb =
        (static_cast<double>(elementBytes) * 2.0) / (1024.0 * 1024.0 * 1024.0 * 1024.0) /
        (elapsedMs / 1000.0f);
    std::cout << "RMSNormFp32Kernel reached memory bandwidth: " << reachedMemBwTb << " TB/s"
              << std::endl;

    cudaMemcpy(rmsnormOut, rmsnormOutDevice, elementBytes, cudaMemcpyDeviceToHost);

    const float error = ComputeRmse(rmsnormGolden, rmsnormOut, elementCount);
    std::cout << "RMSNormFp32Kernel RMSE: " << error << std::endl;

    cudaFree(xDevice);
    cudaFree(weightDevice);
    cudaFree(rmsnormOutDevice);
    delete[] x;
    delete[] weight;
    delete[] rmsnormGolden;
    delete[] rmsnormOut;
    return 0;
}
