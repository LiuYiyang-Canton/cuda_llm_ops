// Compilation command: nvcc -O3  --use_fast_math -gencode=arch=compute_120,code=sm_120 Softmax/softmaxCrossEntropy.cu -o softmaxCrossEntropy
/**
 * Softmax cross-entropy CUDA example with CPU reference, correctness check, and timing.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

constexpr int kBatchSize = 32;
constexpr int kFeatureSize = 131072;
constexpr int kThreadsPerBlock = 1024;

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
 * Kernel for fused softmax + cross-entropy.
 *
 * @param logits device pointer to input logits.
 * @param labels device pointer to label indices.
 * @param loss device pointer to per-row loss, storing sum of exponentials temporarily.
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
__global__ void SoftmaxCrossEntropyKernel(const float* __restrict__ logits, const int* __restrict__ labels,
                                          float* __restrict__  loss, int batchSize, int featureSize) {
    int row = blockIdx.x;
    if (row >= batchSize) {
        return;
    }
    logits += row * featureSize;
    const float4* logits4 = reinterpret_cast<const float4*>(logits);

    // Compute max for numerical stability
    TileMaxSum maxandsum;
    maxandsum.max = -INFINITY;
    maxandsum.sum = 0.0f;
    for (int col = threadIdx.x * 4; col < featureSize; col += blockDim.x * 4) {
        float4 localLogits = logits4[col / 4];
        TileMaxSum newmaxandsum;
        newmaxandsum.max = fmaxf(fmaxf(localLogits.x, localLogits.y),
                                 fmaxf(localLogits.z, localLogits.w));
        newmaxandsum.sum = __expf(localLogits.x - newmaxandsum.max) +
                          __expf(localLogits.y - newmaxandsum.max) +
                          __expf(localLogits.z - newmaxandsum.max) +
                          __expf(localLogits.w - newmaxandsum.max);
        maxandsum = TileMaxSumReduce()(maxandsum, newmaxandsum);
    }
    auto block = cg::tiled_partition<kThreadsPerBlock>(cg::this_thread_block());
    TileMaxSum blockResult = cg::reduce(block, maxandsum, TileMaxSumReduce());

    if (threadIdx.x == 0) {
        int label = labels[row];
        loss[row] = -logits[label] + __logf(blockResult.sum) + blockResult.max;
    }
}

/**
 * Reference CPU implementation for softmax cross-entropy.
 *
 * @param logits host pointer to input logits.
 * @param labels host pointer to label indices, one element per batch.
 * @param loss host pointer to per-row loss.
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
void SoftmaxCrossEntropyCpu(const float* logits, const int* labels,
                            float* loss, int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowLogits = logits + row * featureSize;

        float maxVal = -INFINITY;
        for (int col = 0; col < featureSize; ++col) {
            float val = rowLogits[col];
            if (val > maxVal) {
                maxVal = val;
            }
        }

        float sumExp = 0.0f;
        for (int col = 0; col < featureSize; ++col) {
            float expVal = expf(rowLogits[col] - maxVal);
            sumExp += expVal;
        }

        int label = labels[row];
        loss[row] = -logits[row * featureSize + label] + logf(sumExp) + maxVal;
    }
}

/**
 * Fills a buffer with random normal values.
 *
 * @param data host pointer to output buffer.
 * @param count number of elements to fill.
 * @param generator RNG generator.
 * @param mean normal distribution mean.
 * @param stddev normal distribution standard deviation.
 */
void FillRandomNormal(float* data, int count, std::mt19937& generator, float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
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
 * Computes RMSE between the golden output and kernel results.
 *
 * @param golden host pointer to reference data.
 * @param values host pointer to computed data.
 * @param count number of fp32 elements to compare.
 * @return RMSE in fp32.
 */
float ComputeRmse(const float* __restrict__ golden, const float* __restrict__ values, size_t count) {
    double error = 0.0;  // accumulate squared error in fp64 for stability.
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * Checks a CUDA API call and aborts on failure.
 *
 * @param result CUDA status returned by the call.
 * @param context string describing the call for error reporting.
 */
void CheckCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Entry point: allocates data, runs CPU reference and GPU kernel, and times the result.
 *
 * @return exit code (0 on success).
 */
int main() {
    static_assert(kFeatureSize % 4 == 0, "feature size must be a multiple of 4");

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    const int totalCount = kBatchSize * kFeatureSize;
    float* logits = new float[totalCount];
    int* labels = new int[kBatchSize];
    float* lossGolden = new float[kBatchSize];
    float* loss = new float[kBatchSize];

    FillRandomNormal(logits, totalCount, generator, 0.0f, 1.0f);
    FillRandomLabels(labels, kBatchSize, generator, kFeatureSize);

    SoftmaxCrossEntropyCpu(logits, labels, lossGolden,
                           kBatchSize, kFeatureSize);

    // Allocate device memory and copy data
    float* logitsDevice;
    int* labelsDevice;
    float* lossDevice;
    CheckCuda(cudaMalloc(&logitsDevice, totalCount * sizeof(float)), "cudaMalloc(logitsDevice)");
    CheckCuda(cudaMalloc(&labelsDevice, kBatchSize * sizeof(int)), "cudaMalloc(labelsDevice)");
    CheckCuda(cudaMalloc(&lossDevice, kBatchSize * sizeof(float)), "cudaMalloc(lossDevice)");
    CheckCuda(cudaMemcpy(logitsDevice, logits, totalCount * sizeof(float),
                          cudaMemcpyHostToDevice),
              "cudaMemcpy(logitsDevice)");
    CheckCuda(cudaMemcpy(labelsDevice, labels, kBatchSize * sizeof(int),
                          cudaMemcpyHostToDevice),
              "cudaMemcpy(labelsDevice)");
    CheckCuda(cudaMemset(lossDevice, 0, kBatchSize * sizeof(float)), "cudaMemset(lossDevice)");


    SoftmaxCrossEntropyKernel<<<kBatchSize, kThreadsPerBlock>>>(
        logitsDevice, labelsDevice, lossDevice, kBatchSize, kFeatureSize);
    CheckCuda(cudaGetLastError(), "SoftmaxCrossEntropyKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

    CheckCuda(cudaMemcpy(loss, lossDevice, kBatchSize * sizeof(float),
                          cudaMemcpyDeviceToHost),
              "cudaMemcpy(loss)");
    std::cout << "Copied loss back to host\n";

    float lossRmse = ComputeRmse(lossGolden, loss, kBatchSize);
    std::cout << "Loss RMSE: " << lossRmse << std::endl;

    const int warmupIters = 10;
    const int timedIters = 10;

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmupIters; ++i) {
        SoftmaxCrossEntropyKernel<<<kBatchSize, kThreadsPerBlock>>>(
            logitsDevice, labelsDevice, lossDevice, kBatchSize, kFeatureSize);
        CheckCuda(cudaGetLastError(), "SoftmaxCrossEntropyKernel warmup launch");
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < timedIters; ++i) {
        SoftmaxCrossEntropyKernel<<<kBatchSize, kThreadsPerBlock>>>(
            logitsDevice, labelsDevice, lossDevice, kBatchSize, kFeatureSize);
        CheckCuda(cudaGetLastError(), "SoftmaxCrossEntropyKernel timed launch");
    }
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "cudaEventElapsedTime");
    float avgMs = elapsedMs / timedIters;
    std::cout << "SoftmaxCrossEntropy kernel avg: " << avgMs * 1000.0f
              << " us (" << timedIters << " iters)" << std::endl;

    size_t numElements = static_cast<size_t>(kBatchSize) * kFeatureSize;
    double bytesPerIter = static_cast<double>(numElements) * sizeof(float);
    double bandwidthGBs = bytesPerIter / (avgMs / 1000.0) /
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
