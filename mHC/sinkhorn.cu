// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-22
// Brief:  CPU reference + CUDA kernel comparison for Sinkhorn-Knopp on (B, S, N, N) tensors.
// ==============================================================================
// Compilation command: nvcc -O3 --use_fast_math -gencode=arch=compute_120,code=sm_120 mHC/sinkhorn.cu mHC/sinkhornGolden.cu -o sinkhorn
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>

#include <cuda_runtime.h>

#include "sinkhorn.cuh"

#define CEIL_DIV(X, Y) (((X) + (Y)-1) / (Y))

constexpr int kMatrixSize = 4;

/**
 * @brief CUDA kernel that applies Sinkhorn-Knopp iterations per (B, S) slice.
 * @param input Pointer to input tensor H with shape (batchSize, seqLength, matrixSize, matrixSize),
 *              stored in row-major order. Also used for output.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param iterations Number of Sinkhorn iterations to run.
 * @param epsilon Small constant to avoid division by zero.
 */
__global__ void MHCSinkhornKernel(float* input,
                                  int batchSize,
                                  int seqLength,
                                  int iterations,
                                  float epsilon) {
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalSlices = batchSize * seqLength;
    const int sliceSize = kMatrixSize * kMatrixSize;

    if (globalThreadId >= totalSlices) {
        return;
    }

    input += globalThreadId * sliceSize;

    // temporary buffer for one row/column for this thread
    float temp[kMatrixSize];
    for (int iter = 0; iter < iterations; ++iter) {
        // normalize rows
        for (int i = 0; i < kMatrixSize; ++i) {
            float rowSum = epsilon;
            int idx = i * kMatrixSize;
            for (int j = 0; j < kMatrixSize; j += 4) {
                (float4&)temp[j] = (float4&)input[idx + j];
                rowSum += temp[j] + temp[j + 1] + temp[j + 2] + temp[j + 3];
            }
            for (int j = 0; j < kMatrixSize; j += 4) {
                temp[j] /= rowSum;
                temp[j + 1] /= rowSum;
                temp[j + 2] /= rowSum;
                temp[j + 3] /= rowSum;
                (float4&)input[idx + j] = (float4&)temp[j];
            }
        }

        // normalize columns
        for (int j = 0; j < kMatrixSize; ++j) {
            float colSum = epsilon;
            for (int i = 0; i < kMatrixSize; i += 4) {
                int idx0 = (i + 0) * kMatrixSize + j;
                int idx1 = (i + 1) * kMatrixSize + j;
                int idx2 = (i + 2) * kMatrixSize + j;
                int idx3 = (i + 3) * kMatrixSize + j;
                colSum += input[idx0] + input[idx1] + input[idx2] + input[idx3];
                temp[i + 0] = input[idx0];
                temp[i + 1] = input[idx1];
                temp[i + 2] = input[idx2];
                temp[i + 3] = input[idx3];
            }
            for (int i = 0; i < kMatrixSize; i += 4) {
                temp[i + 0] /= colSum;
                temp[i + 1] /= colSum;
                temp[i + 2] /= colSum;
                temp[i + 3] /= colSum;
                int idx0 = (i + 0) * kMatrixSize + j;
                int idx1 = (i + 1) * kMatrixSize + j;
                int idx2 = (i + 2) * kMatrixSize + j;
                int idx3 = (i + 3) * kMatrixSize + j;
                input[idx0] = temp[i + 0];
                input[idx1] = temp[i + 1];
                input[idx2] = temp[i + 2];
                input[idx3] = temp[i + 3];
            }
        }
    }

}

/**
 * @brief Computes a flat index for a 4D tensor in row-major order.
 * @param b Batch index in [0, batchSize).
 * @param s Sequence index in [0, seqLength).
 * @param i Row index in [0, matrixSize).
 * @param j Column index in [0, matrixSize).
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @return Flat index into the contiguous buffer.
 */
size_t FlatIndex(int b, int s, int i, int j, int seqLength, int matrixSize) {
    return static_cast<size_t>(((b * seqLength + s) * matrixSize + i) * matrixSize + j);
}

/**
 * @brief Checks a CUDA API call result and aborts on error.
 * @param result CUDA error code returned by a runtime API call.
 * @param message Context message describing the failed operation.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << message << " (" << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes the root mean squared error (RMSE) between two arrays.
 * @param reference Pointer to reference data.
 * @param candidate Pointer to candidate data.
 * @param elementCount Number of elements to compare.
 * @return RMSE across all elements.
 */
float ComputeRmse(const float* reference, const float* candidate, size_t elementCount) {
    double sumSquared = 0.0;
    for (size_t idx = 0; idx < elementCount; ++idx) {
        double diff = static_cast<double>(reference[idx]) - static_cast<double>(candidate[idx]);
        sumSquared += diff * diff;
    }
    double meanSquared = sumSquared / static_cast<double>(elementCount);
    return static_cast<float>(std::sqrt(meanSquared));
}

/**
 * @brief Computes effective memory bandwidth in GB/s.
 * @param bytesProcessed Total bytes processed per kernel launch.
 * @param elapsedMs Elapsed time in milliseconds per kernel launch.
 * @return Effective bandwidth in GB/s.
 */
double ComputeEffectiveBandwidthGb(double bytesProcessed, double elapsedMs) {
    const double seconds = elapsedMs * 1.0e-3;
    if (seconds <= 0.0) {
        return 0.0;
    }
    return (bytesProcessed / seconds) / 1.0e9;
}

/**
 * @brief Prints row and column sums for a selected slice in the output tensor.
 * @param data Pointer to the output tensor data.
 * @param batchSize Batch size B.
 * @param seqLength Sequence length S.
 * @param matrixSize Matrix dimension N.
 * @param sampleB Batch index to inspect.
 * @param sampleS Sequence index to inspect.
 */
void PrintRowColSums(const float* data,
    int batchSize,
    int seqLength,
    int matrixSize,
    int sampleB,
    int sampleS) {
    (void)batchSize;
    std::cout << "Row sums (b=" << sampleB << ", s=" << sampleS << "): ";
    for (int i = 0; i < matrixSize; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < matrixSize; ++j) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            rowSum += static_cast<double>(data[idx]);
        }
        std::cout << static_cast<float>(rowSum) << (i + 1 == matrixSize ? '\n' : ' ');
    }

    std::cout << "Col sums (b=" << sampleB << ", s=" << sampleS << "): ";
    for (int j = 0; j < matrixSize; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < matrixSize; ++i) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            colSum += static_cast<double>(data[idx]);
        }
        std::cout << static_cast<float>(colSum) << (j + 1 == matrixSize ? '\n' : ' ');
    }
}

/**
 * @brief Entry point that builds a sample tensor and runs the Sinkhorn CPU reference.
 * @return Exit code (0 on success).
 */
int main() {
    const int batchSize = 2;
    const int seqLength = 128 * 1024;
    constexpr int matrixSize = kMatrixSize;
    const int iterations = 20;
    const float epsilon = 1.0e-6f;
    const int warmupRuns = 20;
    const int timedRuns = 20;

    static_assert(kMatrixSize % 4 == 0, "kMatrixSize must be a multiple of 4");

    const size_t totalSize = static_cast<size_t>(batchSize) * static_cast<size_t>(seqLength) *
        static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);

    float* input = new float[totalSize];
    float* cpuOutput = new float[totalSize];
    float* gpuOutput = new float[totalSize];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t idx = 0; idx < totalSize; ++idx) {
        input[idx] = expf(dist(rng)) + 1.0e-3f;
    }

    MHCSinkhornCpu(input, cpuOutput, batchSize, seqLength, matrixSize, iterations);

    float* deviceInput = nullptr;
    CheckCuda(cudaMalloc(&deviceInput, totalSize * sizeof(float)), "cudaMalloc deviceInput");
    CheckCuda(cudaMemcpy(deviceInput, input, totalSize * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D input");

    const int threadsPerBlock = 256;
    const int totalSlices = batchSize * seqLength;
    const int blocks = CEIL_DIV(totalSlices, threadsPerBlock);
    MHCSinkhornKernel<<<blocks, threadsPerBlock>>>(deviceInput,
        batchSize,
        seqLength,
        iterations,
        epsilon);
    CheckCuda(cudaGetLastError(), "MHCSinkhornKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    CheckCuda(cudaMemcpy(gpuOutput, deviceInput, totalSize * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H output");

    const int sampleB = 0;
    const int sampleS = 0;
    std::cout << "Sample slice (b=0, s=0) after Sinkhorn:" << std::fixed << std::setprecision(4)
        << '\n';
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            const size_t idx = FlatIndex(sampleB, sampleS, i, j, seqLength, matrixSize);
            std::cout << cpuOutput[idx] << (j + 1 == matrixSize ? '\n' : ' ');
        }
    }
    PrintRowColSums(cpuOutput, batchSize, seqLength, matrixSize, sampleB, sampleS);

    const float rmse = ComputeRmse(cpuOutput, gpuOutput, totalSize);
    std::cout << "RMSE (CPU vs GPU): " << rmse << '\n';

    CheckCuda(cudaMemcpy(deviceInput, input, totalSize * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D input (timing)");
    for (int run = 0; run < warmupRuns; ++run) {
        MHCSinkhornKernel<<<blocks, threadsPerBlock>>>(deviceInput,
            batchSize,
            seqLength,
            iterations,
            epsilon);
    }
    CheckCuda(cudaGetLastError(), "MHCSinkhornKernel warmup launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    for (int run = 0; run < timedRuns; ++run) {
        MHCSinkhornKernel<<<blocks, threadsPerBlock>>>(deviceInput,
            batchSize,
            seqLength,
            iterations,
            epsilon);
    }
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");
    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");

    const double avgMs = static_cast<double>(elapsedMs) / static_cast<double>(timedRuns);
    const double avgUs = avgMs * 1.0e3;
    const double bytesPerLaunch = static_cast<double>(batchSize) * static_cast<double>(seqLength) *
        static_cast<double>(matrixSize) * static_cast<double>(matrixSize) *
        static_cast<double>(sizeof(float));
    const double bandwidthGb = ComputeEffectiveBandwidthGb(bytesPerLaunch, avgMs);
    std::cout << "Kernel time: " << avgUs << " us (avg over " << timedRuns << " runs, "
        << warmupRuns << " warmups)\n";
    std::cout << "Effective bandwidth: " << bandwidthGb << " GB/s\n";

    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");

    delete[] input;
    delete[] cpuOutput;
    delete[] gpuOutput;
    return 0;
}
