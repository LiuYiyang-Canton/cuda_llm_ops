// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for FP16 GEMM kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

namespace {

using Fp16 = half;

constexpr int kGemmFp16M = 4096;
constexpr int kGemmFp16N = 4096;
constexpr int kGemmFp16K = 4096;

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
 * @brief Computes RMSE between a reference and computed matrix.
 * @param golden Pointer to reference matrix on host.
 * @param values Pointer to computed matrix on host.
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
 * @brief Runs a cuBLAS GEMM with fp16 inputs and fp32 accumulation.
 * @param handle cuBLAS handle.
 * @param a Pointer to device matrix A (row-major).
 * @param b Pointer to device matrix B (column-major).
 * @param c Pointer to device matrix C (row-major).
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A and rows in B.
 * @return cuBLAS status code.
 */
cublasStatus_t RunCublasGemm(cublasHandle_t handle,
                             const Fp16* a,
                             const Fp16* b,
                             float* c,
                             int m,
                             int n,
                             int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmEx(handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        b,
                        CUDA_R_16F,
                        k,
                        a,
                        CUDA_R_16F,
                        k,
                        &beta,
                        c,
                        CUDA_R_32F,
                        n,
                        CUBLAS_COMPUTE_32F_FAST_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

/**
 * @brief Runs the GEMM kernel benchmark and validation.
 * @return Process exit code.
 */
int main() {
    static_assert(kGemmFp16M % 128 == 0 && kGemmFp16N % 128 == 0, "Matrix sizes must be multiples of 128");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t aElements = static_cast<size_t>(kGemmFp16M) * kGemmFp16K;
    const size_t bElements = static_cast<size_t>(kGemmFp16K) * kGemmFp16N;
    const size_t cElements = static_cast<size_t>(kGemmFp16M) * kGemmFp16N;

    std::vector<Fp16> aHost(aElements);
    std::vector<Fp16> bHost(bElements);
    std::vector<float> cReference(cElements);

    for (size_t i = 0; i < aElements; ++i) {
        aHost[i] = static_cast<Fp16>(distribution(generator));
    }
    for (size_t i = 0; i < bElements; ++i) {
        bHost[i] = static_cast<Fp16>(distribution(generator));
    }

#pragma omp parallel for
    for (int m = 0; m < kGemmFp16M; ++m) {
        for (int n = 0; n < kGemmFp16N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < kGemmFp16K; ++k) {
                sum += __half2float(aHost[static_cast<size_t>(m) * kGemmFp16K + k]) *
                       __half2float(bHost[static_cast<size_t>(n) * kGemmFp16K + k]);
            }
            cReference[static_cast<size_t>(m) * kGemmFp16N + n] = sum;
        }
    }

    Fp16* aDevice = nullptr;
    Fp16* bDevice = nullptr;
    float* cDevice = nullptr;
    CheckCuda(cudaMalloc(&aDevice, aElements * sizeof(Fp16)), "cudaMalloc aDevice");
    CheckCuda(cudaMalloc(&bDevice, bElements * sizeof(Fp16)), "cudaMalloc bDevice");
    CheckCuda(cudaMalloc(&cDevice, cElements * sizeof(float)), "cudaMalloc cDevice");

    CheckCuda(cudaMemcpy(aDevice, aHost.data(), aElements * sizeof(Fp16), cudaMemcpyHostToDevice),
              "cudaMemcpy aHost to aDevice");
    CheckCuda(cudaMemcpy(bDevice, bHost.data(), bElements * sizeof(Fp16), cudaMemcpyHostToDevice),
              "cudaMemcpy bHost to bDevice");

    std::vector<float> cHost(cElements);

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate");

    constexpr int kWarmupIterations = 10;
    for (int i = 0; i < kWarmupIterations; ++i) {
        CheckCublas(RunCublasGemm(handle,
                                  aDevice,
                                  bDevice,
                                  cDevice,
                                  kGemmFp16M,
                                  kGemmFp16N,
                                  kGemmFp16K),
                    "cublasGemmEx warmup");
    }

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord startEvent");
    for (int i = 0; i < kWarmupIterations; ++i) {
        CheckCublas(RunCublasGemm(handle,
                                  aDevice,
                                  bDevice,
                                  cDevice,
                                  kGemmFp16M,
                                  kGemmFp16N,
                                  kGemmFp16K),
                    "cublasGemmEx run");
    }
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord stopEvent");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    elapsedMs /= static_cast<float>(kWarmupIterations);

    std::cout << "cuBLAS gemmEx elapsed time: " << elapsedMs * 1000.0f << " us" << std::endl;
    float peakTflops = static_cast<float>(2.0 * kGemmFp16M) * static_cast<float>(kGemmFp16N) *
                       static_cast<float>(kGemmFp16K) / 1e12f / (elapsedMs / 1e3f);
    std::cout << "cuBLAS gemmEx peak TFLOPS: " << peakTflops << " TFLOPS" << std::endl;

    CheckCuda(cudaMemcpy(cHost.data(), cDevice, cElements * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy cDevice to cHost");
    float error = ComputeRmse(cReference.data(), cHost.data(), cElements);
    std::cout << "cuBLAS gemmEx error: " << error << std::endl;

    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord kernel start");
    for (int i = 0; i < kWarmupIterations; ++i) {
        LaunchGemmFp16Kernel(aDevice, bDevice, cDevice, kGemmFp16M, kGemmFp16N, kGemmFp16K);
    }
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime kernel");
    elapsedMs /= static_cast<float>(kWarmupIterations);

    std::cout << "GemmFp16Kernel elapsed time: " << elapsedMs * 1000.0f << " us" << std::endl;
    peakTflops = static_cast<float>(2.0 * kGemmFp16M) * static_cast<float>(kGemmFp16N) *
                 static_cast<float>(kGemmFp16K) / 1e12f / (elapsedMs / 1e3f);
    std::cout << "GemmFp16Kernel peak TFLOPS: " << peakTflops << " TFLOPS" << std::endl;

    CheckCuda(cudaMemcpy(cHost.data(), cDevice, cElements * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy kernel result");
    error = ComputeRmse(cReference.data(), cHost.data(), cElements);
    std::cout << "GemmFp16Kernel error: " << error << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCublas(cublasDestroy(handle), "cublasDestroy");

    CheckCuda(cudaFree(aDevice), "cudaFree aDevice");
    CheckCuda(cudaFree(bDevice), "cudaFree bDevice");
    CheckCuda(cudaFree(cDevice), "cudaFree cDevice");

    return 0;
}
