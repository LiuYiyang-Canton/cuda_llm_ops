// Compilation command: nvcc -O3  -gencode=arch=compute_120,code=sm_120 Softmax/softmaxBackward.cu -o softmaxBackward
/**
 * Softmax backward CUDA example with CPU reference, correctness check, and timing.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define BATCH_SIZE 32
#define FEATURE_SIZE 131072
#define THREADS_PER_BLOCK 256
#define WORK_PER_THREAD 8
#define CeilDiv(a, b) (((a) + (b) - 1) / (b))

namespace cg = cooperative_groups;

/**
 * Computes per-row dot(s, g) and writes s*g into a temporary buffer.
 *
 * 2D grid: blockIdx.y selects the row, blockIdx.x tiles the feature dimension.
 *
 * @param gradOutput device pointer to upstream gradient.
 * @param softmaxOutput device pointer to softmax output.
 * @param gradX device pointer to s*g buffer.
 * @param dotProduct device pointer to per-row dot(s, g) accumulator.
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
__global__ void SoftmaxBackwardDotKernel(const float* gradOutput, const float* softmaxOutput,
                                         float* gradX, float* dotProduct,
                                         int batchSize, int featureSize) {
    int row = blockIdx.y;
    if (row >= batchSize) {
        return;
    }

    gradOutput += row * featureSize;
    softmaxOutput += row * featureSize;
    gradX += row * featureSize;
    int blockStart = blockIdx.x * blockDim.x * WORK_PER_THREAD;
    int blockEnd = blockStart + blockDim.x * WORK_PER_THREAD;
    if (blockEnd > featureSize) {
        blockEnd = featureSize;
    }

    float dot = 0.0f;

    int threadStart = blockStart + threadIdx.x * 4;
    for (int col = threadStart; col < blockEnd; col += blockDim.x * 4) {
        float4 g = *((float4*)&gradOutput[col]);
        float4 s = *((float4*)&softmaxOutput[col]);
        float4 sg = {g.x * s.x, g.y * s.y, g.z * s.z, g.w * s.w};
        dot += sg.x + sg.y + sg.z + sg.w;
        *((float4*)&gradX[col]) = sg;
    }

    auto block = cg::tiled_partition<THREADS_PER_BLOCK>(cg::this_thread_block());
    float blockSum = cg::reduce(block, dot, cg::plus<float>());
    if (block.thread_rank() == 0) {
        atomicAdd(&dotProduct[row], blockSum);
    }
}

/**
 * Computes final gradient: gradX = s*g - s*dot(s, g).
 *
 * @param softmaxOutput device pointer to softmax output.
 * @param dotProduct device pointer to per-row dot(s, g).
 * @param gradX device pointer to output gradient (in-place over s*g buffer).
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
__global__ void SoftmaxBackwardGradKernel(const float* softmaxOutput,
                                          const float* dotProduct, float* gradX,
                                          int batchSize, int featureSize) {

    int row = blockIdx.y;
    if (row >= batchSize) {
        return;
    }

    softmaxOutput += row * featureSize;
    gradX += row * featureSize;
    float dot = dotProduct[row];
    int blockStart = blockIdx.x * blockDim.x * WORK_PER_THREAD;
    int blockEnd = blockStart + blockDim.x * WORK_PER_THREAD;
    if (blockEnd > featureSize) {
        blockEnd = featureSize;
    }

    int threadStart = blockStart + threadIdx.x * 4;
    for (int col = threadStart; col < blockEnd; col += blockDim.x * 4) {
        float4 s = *((float4*)&softmaxOutput[col]);
        float4 sg = *((float4*)&gradX[col]);
        float4 gradxLocal = {sg.x - s.x * dot,
                             sg.y - s.y * dot,
                             sg.z - s.z * dot,
                             sg.w - s.w * dot};

        // Write back the results
        *((float4*)&gradX[col]) = gradxLocal;
    }
}

/**
 * Reference CPU implementation for softmax backward.
 *
 * @param gradOutput host pointer to upstream gradient.
 * @param softmaxOutput host pointer to softmax output.
 * @param gradX host pointer to output gradient.
 * @param batchSize number of rows.
 * @param featureSize number of columns.
 */
void SoftmaxBackwardCpu(const float* gradOutput, const float* softmaxOutput, float* gradX,
                        int batchSize, int featureSize) {
    for (int row = 0; row < batchSize; ++row) {
        const float* rowGradOutput = gradOutput + row * featureSize;
        const float* rowSoftmaxOutput = softmaxOutput + row * featureSize;
        float* rowGradX = gradX + row * featureSize;

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
 * Entry point: allocates data, runs CPU reference and GPU kernels, and times the result.
 *
 * @return exit code (0 on success).
 */
int main() {
    static_assert(FEATURE_SIZE % 4 == 0);

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    int totalCount = BATCH_SIZE * FEATURE_SIZE;
    float* gradOutput = new float[totalCount];
    float* softmaxOutput = new float[totalCount];
    float* gradXGolden = new float[totalCount];
    float* gradX = new float[totalCount];

    FillRandomNormal(gradOutput, totalCount, generator, 0.0f, 1.0f);
    FillRandomNormal(softmaxOutput, totalCount, generator, 0.0f, 1.0f);

    SoftmaxBackwardCpu(gradOutput, softmaxOutput, gradXGolden,
                       BATCH_SIZE, FEATURE_SIZE);
    
    // Allocate device memory and copy data
    float* gradOutputDevice;
    float* softmaxOutputDevice;
    float* gradXDevice;
    float* dotProductDevice; // Intermediate result for dot product
    cudaMalloc(&gradOutputDevice, totalCount * sizeof(float));
    cudaMalloc(&softmaxOutputDevice, totalCount * sizeof(float));
    cudaMalloc(&gradXDevice, totalCount * sizeof(float));
    cudaMalloc(&dotProductDevice, BATCH_SIZE * sizeof(float));
    cudaMemcpy(gradOutputDevice, gradOutput, totalCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(softmaxOutputDevice, softmaxOutput, totalCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dotProductDevice, 0, BATCH_SIZE * sizeof(float));

    int blocksPerRow = CeilDiv(FEATURE_SIZE, THREADS_PER_BLOCK * WORK_PER_THREAD);
    dim3 gridDim(blocksPerRow, BATCH_SIZE);
    dim3 blockDim(THREADS_PER_BLOCK);

    // Use gradXDevice as a temporary s*g buffer before final gradient writeback.
    SoftmaxBackwardDotKernel<<<gridDim, blockDim>>>(gradOutputDevice, softmaxOutputDevice,
                                                    gradXDevice, dotProductDevice,
                                                    BATCH_SIZE, FEATURE_SIZE);
    SoftmaxBackwardGradKernel<<<gridDim, blockDim>>>(softmaxOutputDevice, dotProductDevice,
                                                     gradXDevice, BATCH_SIZE, FEATURE_SIZE);
    
    cudaMemcpy(gradX, gradXDevice, totalCount * sizeof(float), cudaMemcpyDeviceToHost);

    float rmse = ComputeRmse(gradXGolden, gradX, totalCount);
    std::cout << "RMSE: " << rmse << std::endl;

    const int warmupIters = 10;
    const int timedIters = 10;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmupIters; ++i) {
        SoftmaxBackwardDotKernel<<<gridDim, blockDim>>>(gradOutputDevice, softmaxOutputDevice,
                                                        gradXDevice, dotProductDevice,
                                                        BATCH_SIZE, FEATURE_SIZE);
        SoftmaxBackwardGradKernel<<<gridDim, blockDim>>>(softmaxOutputDevice, dotProductDevice,
                                                         gradXDevice, BATCH_SIZE, FEATURE_SIZE);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < timedIters; ++i) {
        SoftmaxBackwardDotKernel<<<gridDim, blockDim>>>(gradOutputDevice, softmaxOutputDevice,
                                                        gradXDevice, dotProductDevice,
                                                        BATCH_SIZE, FEATURE_SIZE);
        SoftmaxBackwardGradKernel<<<gridDim, blockDim>>>(softmaxOutputDevice, dotProductDevice,
                                                         gradXDevice, BATCH_SIZE, FEATURE_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    float avgMs = elapsedMs / timedIters;
    std::cout << "SoftmaxBackward kernels avg: " << avgMs * 1000.0f
              << " us (" << timedIters << " iters)" << std::endl;

    size_t numElements = static_cast<size_t>(BATCH_SIZE) * FEATURE_SIZE;
    double bytesPerIter = 3.0 * static_cast<double>(numElements) * sizeof(float);
    double bandwidthGBs = bytesPerIter / (avgMs / 1000.0) /
                          (1024.0 * 1024.0 * 1024.0);
    std::cout << "SoftmaxBackward effective bandwidth (est.): "
              << bandwidthGBs << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    delete[] gradOutput;
    delete[] softmaxOutput;
    delete[] gradX;

    // Free device memory
    cudaFree(gradOutputDevice);
    cudaFree(softmaxOutputDevice);
    cudaFree(gradXDevice);
    cudaFree(dotProductDevice);
    return 0;
}
