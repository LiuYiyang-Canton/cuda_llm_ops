// compilation: nvcc -o reducesum.o -gencode=arch=compute_120,code=sm_120 -lcublas --use_fast_math reducesum.cu

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace cg = cooperative_groups;

/**
 * Validates CUDA return codes and aborts on failure.
 *
 * @param result cudaError_t returned by CUDA runtime API.
 * @param message const char* description for debugging output.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Validates cuBLAS status codes and exits on error.
 *
 * @param status cublasStatus_t from cuBLAS calls.
 * @param message const char* diagnostic string.
 */
void CheckCublas(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << message << ": cuBLAS error " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Computes relative RMSE between reference and computed outputs.
 *
 * @param golden const float* pointer to reference vector.
 * @param approx const float* pointer to approximated vector.
 * @param count size_t number of elements.
 * @return float RMSE error metric.
 */
float ComputeRmse(const float* __restrict__ golden, const float* __restrict__ approx, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(approx[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

constexpr int kWarpSize = 32;

/**
 * Each block reduces four rows of the matrix using vectorized loads.
 *
 * @tparam RowsPerBlock compile-time constant describing how many rows each block sums.
 * @tparam ThreadsPerBlock compile-time constant for blockDim.x.
 * @param matrix const float* pointer to input matrix (row-major).
 * @param rowsum float* pointer to row-sum output vector.
 * @param cols int number of columns per row.
 */
template <int RowsPerBlock, int ThreadsPerBlock>
__global__ void ReduceSumFp32Kernel(const float* __restrict__ matrix,
                                    float* __restrict__ rowsum,
                                    int cols) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<kWarpSize>(block);

    constexpr int kWarpsPerBlock = ThreadsPerBlock / kWarpSize;
    static_assert(RowsPerBlock == 4, "Kernel assumes 4 rows per block for float4 vectorization");

    __shared__ float4 warp_accum[kWarpsPerBlock];

    const int thread = threadIdx.x;
    const int warp_id = thread / kWarpSize;
    const int lane = thread % kWarpSize;
    const int row_block = blockIdx.x * RowsPerBlock;
    const float* row_ptr = matrix + static_cast<long long>(row_block) * cols;

    float4 local = {0.f, 0.f, 0.f, 0.f};
    // Each thread sums 4 columns across 4 consecutive rows using float4 vectorization.
    for (int col = thread * 4; col < cols; col += blockDim.x * 4) {
        const float4 r0 = *reinterpret_cast<const float4*>(&row_ptr[col]);
        const float4 r1 = *reinterpret_cast<const float4*>(&row_ptr[cols + col]);
        const float4 r2 = *reinterpret_cast<const float4*>(&row_ptr[2 * cols + col]);
        const float4 r3 = *reinterpret_cast<const float4*>(&row_ptr[3 * cols + col]);
        local.x += r0.x + r0.y + r0.z + r0.w;
        local.y += r1.x + r1.y + r1.z + r1.w;
        local.z += r2.x + r2.y + r2.z + r2.w;
        local.w += r3.x + r3.y + r3.z + r3.w;
    }

    // Warp-level partial reductions.
    local.x = cg::reduce(warp, local.x, cg::plus<float>());
    local.y = cg::reduce(warp, local.y, cg::plus<float>());
    local.z = cg::reduce(warp, local.z, cg::plus<float>());
    local.w = cg::reduce(warp, local.w, cg::plus<float>());

    if (lane == 0) {
        warp_accum[warp_id] = local;
    }
    block.sync();

    // Tree reduction across warps within the block.
    for (int offset = kWarpsPerBlock / 2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            warp_accum[thread].x += warp_accum[thread + offset].x;
            warp_accum[thread].y += warp_accum[thread + offset].y;
            warp_accum[thread].z += warp_accum[thread + offset].z;
            warp_accum[thread].w += warp_accum[thread + offset].w;
        }
        block.sync();
    }

    if (thread == 0) {
        *reinterpret_cast<float4*>(&rowsum[row_block]) = warp_accum[0];
    }
}

}  // namespace

int main() {
    constexpr int kRows = 4096;
    constexpr int kCols = 4096;
    constexpr int kRowsPerBlock = 4;
    constexpr int kThreadsPerBlock = 256;
    static_assert(kRows % kRowsPerBlock == 0, "Rows must be divisible by kRowsPerBlock");
    static_assert(kCols % 4 == 0, "Columns must align to float4 vector width");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t matrix_elements = static_cast<size_t>(kRows) * kCols;
    const size_t matrix_bytes = matrix_elements * sizeof(float);
    const size_t rows_bytes = static_cast<size_t>(kRows) * sizeof(float);

    std::vector<float> host_input(matrix_elements);
    std::vector<float> host_reference(kRows, 0.0f);
    std::vector<float> host_result(kRows, 0.0f);

    std::generate(host_input.begin(), host_input.end(), [&]() { return distribution(generator); });
    for (int row = 0; row < kRows; ++row) {
        const float* row_ptr = &host_input[static_cast<size_t>(row) * kCols];
        host_reference[row] = std::accumulate(row_ptr, row_ptr + kCols, 0.0f);
    }

    float* device_input = nullptr;
    float* device_rowsum = nullptr;
    CheckCuda(cudaMalloc(&device_input, matrix_bytes), "cudaMalloc device_input");
    CheckCuda(cudaMalloc(&device_rowsum, rows_bytes), "cudaMalloc device_rowsum");
    CheckCuda(cudaMemcpy(device_input, host_input.data(), matrix_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input");

    std::vector<float> host_ones(kCols, 1.0f);
    float* device_ones = nullptr;
    CheckCuda(cudaMalloc(&device_ones, kCols * sizeof(float)), "cudaMalloc device_ones");
    CheckCuda(cudaMemcpy(device_ones, host_ones.data(), kCols * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy device_ones");

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start_event");
    CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop_event");

    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto run_cublas = [&]() {
        return cublasSgemv(handle,
                           CUBLAS_OP_T,
                           kCols,
                           kRows,
                           &alpha,
                           device_input,
                           kCols,
                           device_ones,
                           1,
                           &beta,
                           device_rowsum,
                           1);
    };

    constexpr int kWarmupIterations = 10000;
    for (int i = 0; i < kWarmupIterations; ++i) {
        CheckCublas(run_cublas(), "cublasSgemv warmup");
    }

    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord cublas start");
    CheckCublas(run_cublas(), "cublasSgemv run");
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord cublas stop");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize cublas stop");
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime cublas");
    std::cout << "cublasSgemv duration: " << elapsed_ms * 1000.0f << " us" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), device_rowsum, rows_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy cublas result");
    float error = ComputeRmse(host_reference.data(), host_result.data(), kRows);
    std::cout << "cublasSgemv error = " << error << std::endl;
    const double bytes_processed = static_cast<double>(matrix_bytes + rows_bytes);
    const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1000.0;
    const double cublas_bw = bytes_processed / elapsed_seconds / static_cast<double>(1ULL << 40);
    std::cout << "cublasSgemv reached_mem_bw: " << cublas_bw << " TB/s" << std::endl;

    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(kRows / kRowsPerBlock);
    auto launch_kernel = [&]() {
        ReduceSumFp32Kernel<kRowsPerBlock, kThreadsPerBlock><<<grid_dim, block_dim>>>(
            device_input, device_rowsum, kCols);
    };
    for (int i = 0; i < kWarmupIterations; ++i) {
        launch_kernel();
        CheckCuda(cudaGetLastError(), "ReduceSumFp32Kernel warmup");
    }
    CheckCuda(cudaMemset(device_rowsum, 0, rows_bytes), "cudaMemset device_rowsum");

    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord kernel start");
    launch_kernel();
    CheckCuda(cudaGetLastError(), "ReduceSumFp32Kernel run");
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime kernel");
    std::cout << "ReduceSumFp32Kernel duration: " << elapsed_ms * 1000.0f << " us" << std::endl;

    const double kernel_bw = bytes_processed / (static_cast<double>(elapsed_ms) / 1000.0) / static_cast<double>(1ULL << 40);
    std::cout << "ReduceSumFp32Kernel reached_mem_bw: " << kernel_bw << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), device_rowsum, rows_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy kernel result");
    error = ComputeRmse(host_reference.data(), host_result.data(), kRows);
    std::cout << "ReduceSumFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(start_event), "cudaEventDestroy start_event");
    CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop_event");
    CheckCublas(cublasDestroy(handle), "cublasDestroy");

    CheckCuda(cudaFree(device_input), "cudaFree device_input");
    CheckCuda(cudaFree(device_rowsum), "cudaFree device_rowsum");
    CheckCuda(cudaFree(device_ones), "cudaFree device_ones");
    return 0;
}
