// compilation: nvcc -o elementwiseadd.o -gencode=arch=compute_120,code=sm_120 -lcublas elementwiseadd.cu

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

/**
 * Returns ceil(numerator / denominator) for positive integers.
 *
 * @param numerator int value representing dividend.
 * @param denominator int value representing divisor (must be > 0).
 * @return int rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    // Add denominator - 1 to perform integer ceil division safely.
    return (numerator + denominator - 1) / denominator;
}

/**
 * Validates CUDA API results and aborts if an error occurs.
 *
 * @param result cudaError_t status code returned by CUDA runtime.
 * @param message const char* human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        // Stop execution immediately so downstream code never sees invalid state.
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Validates cuBLAS API results and aborts on failure.
 *
 * @param status cublasStatus_t result from cuBLAS call.
 * @param message const char* human-readable description.
 */
void CheckCublas(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        // cuBLAS exposes integer error codes so include that for debugging.
        std::cerr << message << ": cuBLAS error " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Vectorized elementwise add kernel that processes 4 fp32 values per thread.
 *
 * @tparam WorkPerBlock compile-time constant describing how many fp32 elements a block handles.
 * @param a const float* pointer to input matrix A on device memory.
 * @param b const float* pointer to input matrix B on device memory.
 * @param result float* pointer to output matrix on device memory.
 * @param n int column dimension of the square matrix.
 */
template <int WorkPerBlock>
__global__ void ElementwiseAddFp32Kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ result,
                                         int n) {
    const int row = blockIdx.y;
    const int row_offset = row * n;  // row-major offset into the matrix.
    const int block_col = static_cast<int>(blockIdx.x);
    const int col_start = block_col * WorkPerBlock;
    const int tile_end = (block_col + 1) * WorkPerBlock;
    const int col_end = tile_end < n ? tile_end : n;  // clamp for the last partial tile.
    for (int col = col_start + threadIdx.x * 4; col < col_end; col += blockDim.x * 4) {
        // Vectorized load/store improves memory bandwidth utilization.
        const float4 a_value = *reinterpret_cast<const float4*>(&a[row_offset + col]);
        const float4 b_value = *reinterpret_cast<const float4*>(&b[row_offset + col]);
        float4 sum = {a_value.x + b_value.x,
                      a_value.y + b_value.y,
                      a_value.z + b_value.z,
                      a_value.w + b_value.w};
        *reinterpret_cast<float4*>(&result[row_offset + col]) = sum;
    }
}

/**
 * Computes the RMSE between the golden output and kernel results.
 *
 * @param golden const float* pointer to reference data on host.
 * @param values const float* pointer to computed data on host.
 * @param count size_t number of fp32 elements to compare.
 * @return float RMSE metric in fp32.
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

}  // namespace

/**
 * Entry point: initializes data, runs cuBLAS and custom kernels, and reports metrics.
 *
 * @return int standard process exit code.
 */
int main() {
    // Random distribution for host input tensors.
    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    constexpr int kMatrixDim = 4096;
    assert(kMatrixDim % 4 == 0);

    const size_t matrix_elements = static_cast<size_t>(kMatrixDim) * kMatrixDim;
    const size_t matrix_bytes = matrix_elements * sizeof(float);

    // Host buffers store golden reference and kernel output for validation.
    std::vector<float> host_a(matrix_elements);
    std::vector<float> host_b(matrix_elements);
    std::vector<float> host_reference(matrix_elements);
    std::vector<float> host_result(matrix_elements, 0.0f);

    std::generate(host_a.begin(), host_a.end(), [&]() { return distribution(generator); });
    std::generate(host_b.begin(), host_b.end(), [&]() { return distribution(generator); });
    std::transform(host_a.begin(), host_a.end(), host_b.begin(), host_reference.begin(), std::plus<float>());

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_result = nullptr;

    // Allocate device resources once and reuse for both implementations.
    CheckCuda(cudaMalloc(&device_a, matrix_bytes), "cudaMalloc device_a");
    CheckCuda(cudaMalloc(&device_b, matrix_bytes), "cudaMalloc device_b");
    CheckCuda(cudaMalloc(&device_result, matrix_bytes), "cudaMalloc device_result");

    CheckCuda(cudaMemcpy(device_a, host_a.data(), matrix_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_a to device_a");
    CheckCuda(cudaMemcpy(device_b, host_b.data(), matrix_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_b to device_b");

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    // Events provide microsecond-level timing for kernels and cuBLAS calls.
    CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start_event");
    CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop_event");

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate");

    auto run_cublas_sgeam = [&]() {
        return cublasSgeam(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           kMatrixDim,
                           kMatrixDim,
                           &alpha,
                           device_a,
                           kMatrixDim,
                           &beta,
                           device_b,
                           kMatrixDim,
                           device_result,
                           kMatrixDim);
    };

    constexpr int kWarmupIterations = 1000;
    for (int i = 0; i < kWarmupIterations; ++i) {
        // Warm-up hides first-launch overhead when timing the actual run.
        CheckCublas(run_cublas_sgeam(), "cublasSgeam warmup");
    }

    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord start_event");
    CheckCublas(run_cublas_sgeam(), "cublasSgeam run");
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord stop_event");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop_event");
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime cublas");
    std::cout << "cublasSgeam duration: " << elapsed_ms * 1000.0f << " us" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), device_result, matrix_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy result cublas");
    float error = ComputeRmse(host_reference.data(), host_result.data(), matrix_elements);
    std::cout << "cublasSgeam error = " << error << std::endl;
    const double bytes_processed = static_cast<double>(matrix_bytes) * 3.0;
    const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1000.0;
    const double cublas_bw = bytes_processed / elapsed_seconds / static_cast<double>(1ULL << 40);
    std::cout << "cublasSgeam reached_mem_bw: " << cublas_bw << " TB/s" << std::endl;

    constexpr int kThreadsPerBlock = 256;
    constexpr int kWorkPerThread = 4;
    constexpr int kWorkPerBlock = kThreadsPerBlock * kWorkPerThread;
    const dim3 num_blocks(CeilDiv(kMatrixDim, kWorkPerBlock), kMatrixDim);

    auto launch_kernel = [&]() {
        ElementwiseAddFp32Kernel<kWorkPerBlock><<<num_blocks, kThreadsPerBlock>>>(device_a, device_b, device_result, kMatrixDim);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        launch_kernel();
        CheckCuda(cudaGetLastError(), "ElementwiseAddFp32Kernel warmup");
    }
    CheckCuda(cudaMemset(device_result, 0, matrix_bytes), "cudaMemset device_result");

    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord kernel start");
    launch_kernel();
    CheckCuda(cudaGetLastError(), "ElementwiseAddFp32Kernel run");
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime kernel");
    std::cout << "ElementwiseAddFp32Kernel duration: " << elapsed_ms * 1000.0f << " us" << std::endl;
    const double kernel_bw = bytes_processed / (static_cast<double>(elapsed_ms) / 1000.0) / static_cast<double>(1ULL << 40);
    std::cout << "ElementwiseAddFp32Kernel reached_mem_bw: " << kernel_bw << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), device_result, matrix_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy result kernel");
    error = ComputeRmse(host_reference.data(), host_result.data(), matrix_elements);
    std::cout << "ElementwiseAddFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(start_event), "cudaEventDestroy start_event");
    CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop_event");
    CheckCublas(cublasDestroy(handle), "cublasDestroy");

    CheckCuda(cudaFree(device_a), "cudaFree device_a");
    CheckCuda(cudaFree(device_b), "cudaFree device_b");
    CheckCuda(cudaFree(device_result), "cudaFree device_result");

    return 0;
}
