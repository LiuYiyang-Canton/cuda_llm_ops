// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA test for RoPE kernel.
// ==============================================================================

#include "cuda_llm_ops.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Bf16 = __nv_bfloat16;

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
 * @brief Computes the RMSE between golden and computed bf16 buffers.
 * @param golden Pointer to reference data on host.
 * @param values Pointer to computed data on host.
 * @param count Number of bf16 elements to compare.
 * @return RMSE metric in fp32.
 */
float ComputeBf16Rmse(const Bf16* __restrict__ golden, const Bf16* __restrict__ values, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double ref = static_cast<double>(__bfloat162float(golden[i]));
        const double got = static_cast<double>(__bfloat162float(values[i]));
        const double diff = ref - got;
        error += diff * diff;
        norm += ref * ref;
    }
    if (norm == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * @brief Reference CPU rotary position embedding.
 * @param input Input tensor on host.
 * @param output Output tensor on host.
 * @param batchSize Batch size.
 * @param seqLen Sequence length.
 * @param hiddenDim Hidden dimension (must be even).
 * @param base RoPE base frequency.
 */
void ComputeCpuRope(const Bf16* __restrict__ input,
                    Bf16* __restrict__ output,
                    int batchSize,
                    int seqLen,
                    int hiddenDim,
                    float base = 10000.0f) {
    assert(hiddenDim % 2 == 0);
    const int halfDim = hiddenDim / 2;

    std::vector<float> invFreq(halfDim);
    const float hiddenInv = 1.0f / static_cast<float>(hiddenDim);
    for (int i = 0; i < halfDim; ++i) {
        invFreq[i] = std::pow(base, -2.0f * static_cast<float>(i) * hiddenInv);
    }

    for (int batch = 0; batch < batchSize; ++batch) {
        for (int token = 0; token < seqLen; ++token) {
            const size_t baseIndex = (static_cast<size_t>(batch) * seqLen + token) * hiddenDim;
            for (int i = 0; i < halfDim; ++i) {
                const float angle = static_cast<float>(token) * invFreq[i];
                const float cosValue = std::cos(angle);
                const float sinValue = std::sin(angle);
                const size_t evenIndex = baseIndex + static_cast<size_t>(2 * i);
                const size_t oddIndex = evenIndex + 1;
                const float even = __bfloat162float(input[evenIndex]);
                const float odd = __bfloat162float(input[oddIndex]);
                output[evenIndex] = __float2bfloat16(even * cosValue - odd * sinValue);
                output[oddIndex] = __float2bfloat16(even * sinValue + odd * cosValue);
            }
        }
    }
}

}  // namespace

/**
 * @brief Runs the RoPE kernel benchmark and validation.
 * @return Process exit code.
 */
int main() {
    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    constexpr int kBatchSize = 1;
    constexpr int kSeqLen = 512 * 1024;
    constexpr int kRopeHiddenDim = 128;
    static_assert(kRopeHiddenDim % 2 == 0, "kRopeHiddenDim must be even for RoPE.");

    const size_t totalElements = static_cast<size_t>(kBatchSize) * kSeqLen * kRopeHiddenDim;
    const size_t totalBytes = totalElements * sizeof(Bf16);

    std::vector<Bf16> hostInput(totalElements);
    std::generate(hostInput.begin(), hostInput.end(), [&]() {
        return static_cast<Bf16>(distribution(generator));
    });

    std::vector<Bf16> hostReference(totalElements);
    const auto cpuStart = std::chrono::steady_clock::now();
    ComputeCpuRope(hostInput.data(), hostReference.data(), kBatchSize, kSeqLen, kRopeHiddenDim);
    const auto cpuEnd = std::chrono::steady_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU RoPE duration: " << cpuMs * 1000.0 << " us" << std::endl;

    Bf16* deviceInput = nullptr;
    Bf16* deviceOutput = nullptr;
    CheckCuda(cudaMalloc(&deviceInput, totalBytes), "cudaMalloc deviceInput");
    CheckCuda(cudaMalloc(&deviceOutput, totalBytes), "cudaMalloc deviceOutput");

    CheckCuda(cudaMemcpy(deviceInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy hostInput to deviceInput");
    CheckCuda(cudaMemset(deviceOutput, 0, totalBytes), "cudaMemset deviceOutput");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CheckCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    CheckCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    constexpr int kWarmupIterations = 100;
    for (int i = 0; i < kWarmupIterations; ++i) {
        LaunchRopeBf16Kernel(deviceInput, deviceOutput, kBatchSize, kSeqLen, kRopeHiddenDim);
    }
    CheckCuda(cudaGetLastError(), "RopeKernel warmup");

    CheckCuda(cudaMemset(deviceOutput, 0, totalBytes), "cudaMemset deviceOutput before timing");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventRecord(startEvent), "cudaEventRecord kernel start");
    for (int i = 0; i < 10; ++i) {
        LaunchRopeBf16Kernel(deviceInput, deviceOutput, kBatchSize, kSeqLen, kRopeHiddenDim);
    }
    CheckCuda(cudaGetLastError(), "RopeKernel run");
    CheckCuda(cudaEventRecord(stopEvent), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime kernel");
    elapsedMs /= 10.0f;
    std::cout << "RopeKernel duration: " << elapsedMs * 1000.0f << " us" << std::endl;

    const double bytesProcessed = static_cast<double>(totalBytes) * 2.0;
    const double kernelBandwidth = bytesProcessed / (static_cast<double>(elapsedMs) / 1000.0) /
                                   static_cast<double>(1ULL << 40);
    std::cout << "RopeKernel reached_mem_bw: " << kernelBandwidth << " TB/s" << std::endl;

    std::vector<Bf16> hostOutput(totalElements);
    CheckCuda(cudaMemcpy(hostOutput.data(), deviceOutput, totalBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy deviceOutput to host");
    const float error = ComputeBf16Rmse(hostReference.data(), hostOutput.data(), totalElements);
    std::cout << "RopeKernel error = " << std::scientific << std::setprecision(4) << error
              << std::defaultfloat << std::endl;

    CheckCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    CheckCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    CheckCuda(cudaFree(deviceInput), "cudaFree deviceInput");
    CheckCuda(cudaFree(deviceOutput), "cudaFree deviceOutput");

    return 0;
}
