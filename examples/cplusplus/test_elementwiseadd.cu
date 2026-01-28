// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for elementwise add kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

/**
 * @brief Validates CUDA API results and aborts if an error occurs.
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
 * @param status Status code returned by cuBLAS.
 * @param message Human-readable context for diagnostics.
 */
void CheckCublas(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << message << ": cuBLAS error " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes the RMSE between the golden output and computed values.
 * @param golden Pointer to reference data on the host.
 * @param values Pointer to computed data on the host.
 * @param count Number of fp32 elements to compare.
 * @return RMSE value in fp32.
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

}  // namespace

/**
 * @brief Runs the elementwise add benchmark and validation.
 * @return Process exit code.
 */
int main() {
    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    constexpr int kMatrixDim = 4096;
    static_assert(kMatrixDim % kElementwiseAddWorkPerThread == 0, "Matrix dim must be divisible by 4.");

    const size_t matrixElements = static_cast<size_t>(kMatrixDim) * kMatrixDim;
    const size_t matrixBytes = matrixElements * sizeof(float);

    std::vector<float> hostA(matrixElements);
    std::vector<float> hostB(matrixElements);
    std::vector<float> hostReference(matrixElements);
    std::vector<float> hostResult(matrixElements, 0.0f);

    std::generate(hostA.begin(), hostA.end(), [&]() { return distribution(generator); });
    std::generate(hostB.begin(), hostB.end(), [&]() { return distribution(generator); });
    std::transform(hostA.begin(), hostA.end(), hostB.begin(), hostReference.begin(), std::plus<float>());

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceResult = nullptr;

    CheckCuda(cudaMalloc(&deviceA, matrixBytes), "cudaMalloc deviceA");
    CheckCuda(cudaMalloc(&deviceB, matrixBytes), "cudaMalloc deviceB");
    CheckCuda(cudaMalloc(&deviceResult, matrixBytes), "cudaMalloc deviceResult");

    CheckCuda(cudaMemcpy(deviceA, hostA.data(), matrixBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostA to deviceA");
    CheckCuda(cudaMemcpy(deviceB, hostB.data(), matrixBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostB to deviceB");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate");

    auto runCublasSgeam = [&]() {
        return cublasSgeam(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           kMatrixDim,
                           kMatrixDim,
                           &alpha,
                           deviceA,
                           kMatrixDim,
                           &beta,
                           deviceB,
                           kMatrixDim,
                           deviceResult,
                           kMatrixDim);
    };

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    CheckCublas(runCublasSgeam(), "cublasSgeam run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime cublas");
    std::cout << "cublasSgeam duration: " << elapsedMs * 1000.0f << " us" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceResult, matrixBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy result cublas");
    float error = ComputeRmse(hostReference.data(), hostResult.data(), matrixElements);
    std::cout << "cublasSgeam error = " << error << std::endl;
    const double bytesProcessed = static_cast<double>(matrixBytes) * 3.0;
    const double elapsedSeconds = static_cast<double>(elapsedMs) / 1000.0;
    const double cublasBandwidth = bytesProcessed / elapsedSeconds / static_cast<double>(1ULL << 40);
    std::cout << "cublasSgeam reached_mem_bw: " << cublasBandwidth << " TB/s" << std::endl;

    constexpr int kThreadsPerBlock = 256;

    auto launchKernel = [&]() {
        LaunchElementwiseAddFp32Kernel(deviceA, deviceB, deviceResult, kMatrixDim, kThreadsPerBlock);
    };

    CheckCuda(cudaMemset(deviceResult, 0, matrixBytes), "cudaMemset deviceResult");

    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord kernel start");
    launchKernel();
    CheckCuda(cudaGetLastError(), "ElementwiseAddFp32Kernel run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime kernel");
    std::cout << "ElementwiseAddFp32Kernel duration: " << elapsedMs * 1000.0f << " us" << std::endl;
    const double kernelBandwidth = bytesProcessed / (static_cast<double>(elapsedMs) / 1000.0) /
                                   static_cast<double>(1ULL << 40);
    std::cout << "ElementwiseAddFp32Kernel reached_mem_bw: " << kernelBandwidth << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(hostResult.data(), deviceResult, matrixBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy result kernel");
    error = ComputeRmse(hostReference.data(), hostResult.data(), matrixElements);
    std::cout << "ElementwiseAddFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCublas(cublasDestroy(handle), "cublasDestroy");

    CheckCuda(cudaFree(deviceA), "cudaFree deviceA");
    CheckCuda(cudaFree(deviceB), "cudaFree deviceB");
    CheckCuda(cudaFree(deviceResult), "cudaFree deviceResult");

    return 0;
}
