// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for ReduceSum kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

/**
 * @brief Validates CUDA API results and aborts on failure.
 * @param result CUDA status code returned by the runtime.
 * @param message Human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Validates cuBLAS API results and aborts on failure.
 * @param status cuBLAS status code returned by the library.
 * @param message Human-readable context for diagnostics.
 */
void CheckCublas(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << message << ": cuBLAS error " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes RMSE between reference and computed results.
 * @param golden Pointer to reference data on the host.
 * @param values Pointer to computed data on the host.
 * @param count Number of elements to compare.
 * @return Relative RMSE value.
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
 * @brief Runs a cuBLAS SGEMV to compute row sums.
 * @param handle cuBLAS handle.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param input Pointer to device matrix (row-major).
 * @param ones Pointer to device vector of ones.
 * @param output Pointer to device output vector.
 * @return cuBLAS status code.
 */
cublasStatus_t RunCublasSgemv(cublasHandle_t handle,
                              int rows,
                              int cols,
                              const float* input,
                              const float* ones,
                              float* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemv(handle,
                       CUBLAS_OP_T,
                       cols,
                       rows,
                       &alpha,
                       input,
                       cols,
                       ones,
                       1,
                       &beta,
                       output,
                       1);
}

}  // namespace

/**
 * @brief Runs the ReduceSum kernel benchmark and validation.
 * @return Process exit code.
 */
int main() {
    constexpr int kRows = 4096;
    constexpr int kCols = 4096;
    static_assert(kRows % kReduceSumRowsPerBlock == 0, "Rows must be divisible by kReduceSumRowsPerBlock");
    static_assert(kCols % 4 == 0, "Columns must align to float4 vector width");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t matrixElements = static_cast<size_t>(kRows) * kCols;
    const size_t matrixBytes = matrixElements * sizeof(float);
    const size_t rowsBytes = static_cast<size_t>(kRows) * sizeof(float);

    std::vector<float> hostInput(matrixElements);
    std::vector<float> hostReference(kRows, 0.0f);
    std::vector<float> hostResult(kRows, 0.0f);

    std::generate(hostInput.begin(), hostInput.end(), [&]() { return distribution(generator); });
    for (int row = 0; row < kRows; ++row) {
        const float* rowPtr = &hostInput[static_cast<size_t>(row) * kCols];
        hostReference[row] = std::accumulate(rowPtr, rowPtr + kCols, 0.0f);
    }

    float* deviceInput = nullptr;
    float* deviceRowSum = nullptr;
    CheckCuda(cudaMalloc(&deviceInput, matrixBytes), "cudaMalloc deviceInput");
    CheckCuda(cudaMalloc(&deviceRowSum, rowsBytes), "cudaMalloc deviceRowSum");
    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), matrixBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput");

    std::vector<float> hostOnes(kCols, 1.0f);
    float* deviceOnes = nullptr;
    CheckCuda(cudaMalloc(&deviceOnes, kCols * sizeof(float)), "cudaMalloc deviceOnes");
    CheckCuda(cudaMemcpy(deviceOnes, hostOnes.data(), kCols * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy deviceOnes");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate");

    constexpr int kWarmupIterations = 10000;
    for (int i = 0; i < kWarmupIterations; ++i) {
        CheckCublas(RunCublasSgemv(handle, kRows, kCols, deviceInput, deviceOnes, deviceRowSum),
                    "cublasSgemv warmup");
    }

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord cublas start");
    CheckCublas(RunCublasSgemv(handle, kRows, kCols, deviceInput, deviceOnes, deviceRowSum),
                "cublasSgemv run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord cublas stop");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize cublas stop");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime cublas");
    std::cout << "cublasSgemv duration: " << elapsedMs * 1000.0f << " us" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceRowSum, rowsBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy cublas result");
    float error = ComputeRmse(hostReference.data(), hostResult.data(), kRows);
    std::cout << "cublasSgemv error = " << error << std::endl;
    const double bytesProcessed = static_cast<double>(matrixBytes + rowsBytes);
    const double elapsedSeconds = static_cast<double>(elapsedMs) / 1000.0;
    const double cublasBandwidth = bytesProcessed / elapsedSeconds / static_cast<double>(1ULL << 40);
    std::cout << "cublasSgemv reached_mem_bw: " << cublasBandwidth << " TB/s" << std::endl;

    for (int i = 0; i < kWarmupIterations; ++i) {
        LaunchReduceSumFp32Kernel(deviceInput, deviceRowSum, kRows, kCols);
        CheckCuda(cudaGetLastError(), "ReduceSumFp32Kernel warmup");
    }
    CheckCuda(cudaMemset(deviceRowSum, 0, rowsBytes), "cudaMemset deviceRowSum");

    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord kernel start");
    LaunchReduceSumFp32Kernel(deviceInput, deviceRowSum, kRows, kCols);
    CheckCuda(cudaGetLastError(), "ReduceSumFp32Kernel run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime kernel");
    std::cout << "ReduceSumFp32Kernel duration: " << elapsedMs * 1000.0f << " us" << std::endl;

    const double kernelBandwidth = bytesProcessed / (static_cast<double>(elapsedMs) / 1000.0) /
                                   static_cast<double>(1ULL << 40);
    std::cout << "ReduceSumFp32Kernel reached_mem_bw: " << kernelBandwidth << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceRowSum, rowsBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy kernel result");
    error = ComputeRmse(hostReference.data(), hostResult.data(), kRows);
    std::cout << "ReduceSumFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCublas(cublasDestroy(handle), "cublasDestroy");

    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");
    CheckCuda(cudaFree(deviceRowSum), "cudaFree deviceRowSum");
    CheckCuda(cudaFree(deviceOnes), "cudaFree deviceOnes");

    return 0;
}
