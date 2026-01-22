// Compilation command: nvcc -O3 --use_fast_math -gencode=arch=compute_120,code=sm_120 Softmax/softmaxCrossEntropyBackward.cu -o softmaxCrossEntropyBackward
/**
 * Softmax cross-entropy backward CUDA example with CPU reference, correctness check, and timing.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

constexpr int kBatchSize = 32;
constexpr int kFeatureSize = 131072;
constexpr int kThreadsPerBlock = 1024;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;

namespace cg = cooperative_groups;

// a struct that holds the max and sum for a tile
// each thread block computes one row's max and sum
struct TileMaxSum {
    float max;
    float sum;
};

// define a reduce operator for TileMaxSum
struct TileMaxSumReduce {
    /**
     * Combines two TileMaxSum values into a single max/sum pair.
     *
     * @param a first value to reduce.
     * @param b second value to reduce.
     * @return reduced max/sum pair.
     */
    __device__ TileMaxSum operator()(const TileMaxSum& a, const TileMaxSum& b) const {
        TileMaxSum result;
        result.max = fmaxf(a.max, b.max);
        result.sum = a.sum * __expf(a.max - result.max) + b.sum * __expf(b.max - result.max);
        return result;
    }
};

/**
 * Kernel that computes per-row max and sum for softmax.
 *
 * @param logits device pointer to input logits.
 * @param rowMax device pointer to output row max values.
 * @param rowSum device pointer to output row sum values.
 */
__global__ void ComputeRowMaxRowSumFp32Kernel(const float* __restrict__ logits,
                                              float* __restrict__ rowMax,
                                              float* __restrict__ rowSum) {
    int warpId = threadIdx.x / kWarpSize;
    int batchIdx = blockIdx.x;
    int tx = threadIdx.x;
    logits += batchIdx * kFeatureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);
    auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    TileMaxSum maxandsum;
    maxandsum.max = -INFINITY;
    maxandsum.sum = 0.0f;
    __shared__ TileMaxSum warpMaxSum[kWarpCount];

    // each thread processes multiple elements
    for (int col = tx * 4; col < kFeatureSize; col += blockDim.x * 4) {
        float4 val = logits4[col / 4];
        TileMaxSum newmaxandsum;
        newmaxandsum.max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
        newmaxandsum.sum = __expf(val.x - newmaxandsum.max) +
                           __expf(val.y - newmaxandsum.max) +
                           __expf(val.z - newmaxandsum.max) +
                           __expf(val.w - newmaxandsum.max);
        maxandsum = TileMaxSumReduce()(maxandsum, newmaxandsum);
    }

    // warp-level reduction
    TileMaxSum warpResult = cg::reduce(warp, maxandsum, TileMaxSumReduce());

    // thread 0 of each warp writes to shared memory
    if (warp.thread_rank() == 0) {
        warpMaxSum[warpId] = warpResult;
    }
    __syncthreads();

    // tree reduction among warps
    for (int offset = kWarpCount / 2; offset > 0; offset /= 2) {
        if (tx < offset) {
            warpMaxSum[tx] = TileMaxSumReduce()(warpMaxSum[tx], warpMaxSum[tx + offset]);
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (tx == 0) {
        rowMax[batchIdx] = warpMaxSum[0].max;
        rowSum[batchIdx] = warpMaxSum[0].sum;
    }
}

/**
 * Kernel that computes softmax-cross-entropy gradient for each row.
 *
 * @param logits device pointer to input logits.
 * @param rowMax device pointer to input row max values.
 * @param rowSum device pointer to input row sum values.
 * @param labels device pointer to label indices.
 * @param gradLogits device pointer to output gradient.
 */
__global__ void SoftmaxCrossEntropyBackwardFp32Kernel(const float* __restrict__ logits,
                                                      const float* __restrict__ rowMax,
                                                      const float* __restrict__ rowSum,
                                                      const int* __restrict__ labels,
                                                      float* __restrict__ gradLogits) {
    int batchIdx = blockIdx.x;
    int tx = threadIdx.x;
    logits += batchIdx * kFeatureSize;
    gradLogits += batchIdx * kFeatureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);
    float4* grad4 = reinterpret_cast<float4*>(gradLogits);
    float maxVal = rowMax[batchIdx];
    float sumVal = rowSum[batchIdx];
    int label = labels[batchIdx];

    // each thread processes multiple elements
    for (int col = tx * 4; col < kFeatureSize; col += blockDim.x * 4) {
        float4 val = logits4[col / 4];
        float4 result;
        result.x = __expf(val.x - maxVal) / sumVal;
        result.y = __expf(val.y - maxVal) / sumVal;
        result.z = __expf(val.z - maxVal) / sumVal;
        result.w = __expf(val.w - maxVal) / sumVal;
        int base = col;
        if (label >= base && label < base + 4) {
            int offset = label - base;
            if (offset == 0) {
                result.x -= 1.0f;
            } else if (offset == 1) {
                result.y -= 1.0f;
            } else if (offset == 2) {
                result.z -= 1.0f;
            } else {
                result.w -= 1.0f;
            }
        }
        grad4[col / 4] = result;
    }
}

/**
 * Fills a buffer with random uniform values.
 *
 * @param data host pointer to output buffer.
 * @param count number of elements to fill.
 * @param generator RNG generator.
 * @param min minimum value.
 * @param max maximum value.
 */
void FillRandomUniform(float* data, int count, std::mt19937& generator, float min, float max) {
    std::uniform_real_distribution<float> distribution(min, max);
    for (int i = 0; i < count; ++i) {
        data[i] = distribution(generator);
    }
}

/**
 * Fills a buffer with random labels.
 *
 * @param data host pointer to output buffer.
 * @param count number of labels to fill.
 * @param generator RNG generator.
 * @param numClasses number of classes.
 */
void FillRandomLabels(int* data, int count, std::mt19937& generator, int numClasses) {
    std::uniform_int_distribution<int> distribution(0, numClasses - 1);
    for (int i = 0; i < count; ++i) {
        data[i] = distribution(generator);
    }
}

/**
 * Reference CPU implementation for softmax cross-entropy backward.
 *
 * @param logits host pointer to input logits.
 * @param labels host pointer to label indices.
 * @param gradLogits host pointer to output gradient.
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
void SoftmaxCrossEntropyBackwardCpu(const float* logits, const int* labels,
                                    float* gradLogits, int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowLogits = logits + row * featureSize;
        float* rowGrad = gradLogits + row * featureSize;

        float maxVal = -INFINITY;
        for (int col = 0; col < featureSize; ++col) {
            float val = rowLogits[col];
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

        int label = labels[row];
        rowGrad[label] -= 1.0f;
    }
}

/**
 * Computes RMSE between the golden output and kernel results.
 *
 * @param golden host pointer to reference data.
 * @param values host pointer to computed data.
 * @param count number of fp32 elements to compare.
 * @return RMSE in fp32.
 */
float ComputeRmse(const float* __restrict__ golden, const float* __restrict__ values, size_t count) {
    float error = 0.0f;
    float norm = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = golden[i] - values[i];
        error += diff * diff;
        norm += golden[i] * golden[i];
    }
    return std::sqrt(error) / std::sqrt(norm);
}

/**
 * Entry point: allocates data, runs CPU reference and GPU kernel, and times the result.
 *
 * @return exit code (0 on success).
 */
int main() {
    static_assert(kFeatureSize % 4 == 0, "feature size must be a multiple of 4");
    static_assert(kThreadsPerBlock % kWarpSize == 0,
                  "threads per block must be a multiple of warp size");

    unsigned seed = static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    const int totalCount = kBatchSize * kFeatureSize;
    float* logits = new float[totalCount];
    int* labels = new int[kBatchSize];
    float* gradGolden = new float[totalCount];

    FillRandomUniform(logits, totalCount, generator, 0.0f, 1.0f);
    FillRandomLabels(labels, kBatchSize, generator, kFeatureSize);
    SoftmaxCrossEntropyBackwardCpu(logits, labels, gradGolden, kBatchSize, kFeatureSize);

    // GPU implementation
    float* logitsDevice;
    float* gradDevice;
    float* rowMaxDevice;
    float* rowSumDevice;
    int* labelsDevice;
    cudaMalloc(&logitsDevice, totalCount * sizeof(float));
    cudaMalloc(&gradDevice, totalCount * sizeof(float));
    cudaMalloc(&rowMaxDevice, kBatchSize * sizeof(float));
    cudaMalloc(&rowSumDevice, kBatchSize * sizeof(float));
    cudaMalloc(&labelsDevice, kBatchSize * sizeof(int));
    cudaMemcpy(logitsDevice, logits, totalCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(labelsDevice, labels, kBatchSize * sizeof(int), cudaMemcpyHostToDevice);

    // For profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedMs;

    dim3 numThreads(kThreadsPerBlock);
    dim3 numBlocks(kBatchSize);

    // warm up
    const int warmupIters = 1000;
    for (int i = 0; i < warmupIters; ++i) {
        ComputeRowMaxRowSumFp32Kernel<<<numBlocks, numThreads>>>(logitsDevice, rowMaxDevice, rowSumDevice);
        SoftmaxCrossEntropyBackwardFp32Kernel<<<numBlocks, numThreads>>>(
            logitsDevice, rowMaxDevice, rowSumDevice, labelsDevice, gradDevice);
    }

    cudaMemset(gradDevice, 0, totalCount * sizeof(float));
    // each block computes one row
    cudaEventRecord(start);
    ComputeRowMaxRowSumFp32Kernel<<<numBlocks, numThreads>>>(logitsDevice, rowMaxDevice, rowSumDevice);
    SoftmaxCrossEntropyBackwardFp32Kernel<<<numBlocks, numThreads>>>(
        logitsDevice, rowMaxDevice, rowSumDevice, labelsDevice, gradDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel duration: " << elapsedMs * 1000.0f
              << " us" << std::endl;

    // copy back the result
    float* grad = new float[totalCount];
    cudaMemcpy(grad, gradDevice, totalCount * sizeof(float), cudaMemcpyDeviceToHost);
    float error = ComputeRmse(gradGolden, grad, totalCount);
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel error: " << error << std::endl;
    float reachedMemoryBandwidth =
        (static_cast<float>(totalCount) * sizeof(float) * 2) /
        (1024.0f * 1024.0f * 1024.0f * 1024.0f) /
        (elapsedMs / 1000.0f);
    std::cout << "SoftmaxCrossEntropyBackwardFp32Kernel reached memory bandwidth: "
              << reachedMemoryBandwidth << " TB/s" << std::endl;

    // Free memory
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
