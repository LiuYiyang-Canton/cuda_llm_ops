// compilation: nvcc -o bitonicsort.o -gencode=arch=compute_120,code=sm_120 -O3 bitonicsort.cu

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kBatchSize = 1;
constexpr int kArrayLength = 1024 * 1024;
constexpr int kThreadsPerBlock = 256;
constexpr int kWorkPerThread = 4;
constexpr int kLocalWorkPerThread = 4;
constexpr int kVectorWidth = 4;
constexpr int kPadding = 4;  // Must be a multiple of kVectorWidth for float4 access.
constexpr int kWarpSize = 32;

static_assert(kArrayLength > 0 && (kArrayLength & (kArrayLength - 1)) == 0,
              "kArrayLength must be a positive power of two for bitonic sort");
static_assert(kLocalWorkPerThread % kVectorWidth == 0, "Each thread must handle a multiple of kVectorWidth elements");

constexpr int kLocalSize = kThreadsPerBlock * kLocalWorkPerThread;
constexpr int kLocalPad = (kLocalSize / kWarpSize) * kPadding;  // Pad to reduce shared-memory bank conflicts.
constexpr bool kSortAscending = true;
constexpr int kWarmupIterations = 1000;

/**
 * Returns ceil(numerator / denominator) for positive integers.
 *
 * @param numerator dividend.
 * @param denominator divisor, must be > 0.
 * @return integer quotient rounded up.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * Validates CUDA API results and aborts on failure.
 *
 * @param result cudaError_t status code returned by CUDA runtime.
 * @param message const char* human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Computes RMSE between the golden output and computed values.
 *
 * @param golden const float* reference data on host.
 * @param values const float* computed data on host.
 * @param count size_t number of elements.
 * @return float RMSE metric.
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
 * CPU reference bitonic sort (row-wise) to produce golden output.
 *
 * @param input const float* pointer to input buffer.
 * @param output float* pointer to output buffer.
 * @param batch_size int number of rows.
 * @param array_length int length of each row.
 */
void SortCpu(const float* __restrict__ input, float* __restrict__ output, int batch_size, int array_length) {
    std::vector<float> buffer(array_length);
    for (int batch = 0; batch < batch_size; ++batch) {
        const float* row_in = input + batch * array_length;
        float* row_out = output + batch * array_length;
        std::copy(row_in, row_in + array_length, buffer.begin());
        std::sort(buffer.begin(), buffer.end());
        std::copy(buffer.begin(), buffer.end(), row_out);
    }
}

/**
 * Sorts a local segment of one row using shared memory.
 *
 * Grid: dim3 grid(num_segments_per_row, batch_size).
 * One block = one segment of length kLocalSize in one row.
 */
__global__ void BitonicBlockSortShared(float* data, bool ascending) {
    extern __shared__ float shared[];  // Padded shared memory to reduce bank conflicts.

    const int row = static_cast<int>(blockIdx.y);
    const int block_segment = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int segment_start = block_segment * kLocalSize;

    float* row_data = data + row * kArrayLength;

#pragma unroll
    for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
        const int local_index = thread * kLocalWorkPerThread + offset;
        const int global_index = segment_start + local_index;
        const int padded_index = local_index + (local_index / kWarpSize) * kPadding;
        *reinterpret_cast<float4*>(&shared[padded_index]) =
            *reinterpret_cast<const float4*>(&row_data[global_index]);
    }
    __syncthreads();

    // Run full bitonic sort for this segment in shared memory.
    for (int k = 2; k <= kLocalSize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= kVectorWidth) {
#pragma unroll
                for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
                    const int local_index = thread * kLocalWorkPerThread + offset;
                    const int global_index = segment_start + local_index;
                    const int ixj_global = global_index ^ j;
                    const int ixj_local = ixj_global - segment_start;

                    if (ixj_global > global_index && ixj_local >= 0 && ixj_local < kLocalSize) {
                        const int padded_index = local_index + (local_index / kWarpSize) * kPadding;
                        const int padded_index_j = ixj_local + (ixj_local / kWarpSize) * kPadding;
                        float4 value_i = *reinterpret_cast<float4*>(&shared[padded_index]);
                        float4 value_j = *reinterpret_cast<float4*>(&shared[padded_index_j]);

                        const bool sort_up = ascending ? ((global_index & k) == 0) : ((global_index & k) != 0);

                        if ((value_i.x > value_j.x) == sort_up) {
                            const float temp = value_i.x;
                            value_i.x = value_j.x;
                            value_j.x = temp;
                        }
                        if ((value_i.y > value_j.y) == sort_up) {
                            const float temp = value_i.y;
                            value_i.y = value_j.y;
                            value_j.y = temp;
                        }
                        if ((value_i.z > value_j.z) == sort_up) {
                            const float temp = value_i.z;
                            value_i.z = value_j.z;
                            value_j.z = temp;
                        }
                        if ((value_i.w > value_j.w) == sort_up) {
                            const float temp = value_i.w;
                            value_i.w = value_j.w;
                            value_j.w = temp;
                        }
                        *reinterpret_cast<float4*>(&shared[padded_index]) = value_i;
                        *reinterpret_cast<float4*>(&shared[padded_index_j]) = value_j;
                    }
                    __syncthreads();
                }
            } else {
                for (int offset = 0; offset < kLocalWorkPerThread; ++offset) {
                    const int local_index = thread * kLocalWorkPerThread + offset;
                    const int global_index = segment_start + local_index;
                    const int ixj_global = global_index ^ j;
                    const int ixj_local = ixj_global - segment_start;

                    if (ixj_global > global_index && ixj_local >= 0 && ixj_local < kLocalSize) {
                        const int padded_index = local_index + (local_index / kWarpSize) * kPadding;
                        const int padded_index_j = ixj_local + (ixj_local / kWarpSize) * kPadding;
                        const float value_i = shared[padded_index];
                        const float value_j = shared[padded_index_j];

                        const bool sort_up = ascending ? ((global_index & k) == 0) : ((global_index & k) != 0);

                        if ((value_i > value_j) == sort_up) {
                            shared[padded_index] = value_j;
                            shared[padded_index_j] = value_i;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

#pragma unroll
    for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
        const int local_index = thread * kLocalWorkPerThread + offset;
        const int global_index = segment_start + local_index;
        const int padded_index = local_index + (local_index / kWarpSize) * kPadding;
        *reinterpret_cast<float4*>(&row_data[global_index]) =
            *reinterpret_cast<float4*>(&shared[padded_index]);
    }
}

/**
 * Bitonic merge network on global memory for each row.
 *
 * @param data pointer to [batch_size, kArrayLength] row-major tensor.
 * @param j current stride in the merge network.
 * @param k current subsequence length in the merge network.
 * @param ascending whether the sort order is ascending.
 */
__global__ void BitonicSortFp32Kernel(float* data, int j, int k, bool ascending) {
    const int row = static_cast<int>(blockIdx.y);
    const int base_index = (static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x)) * kWorkPerThread;

    float* row_data = data + row * kArrayLength;

    if (row >= kBatchSize || base_index >= kArrayLength) {
        return;
    }

    if (j >= kVectorWidth) {
#pragma unroll
        for (int offset = 0; offset < kWorkPerThread; offset += kVectorWidth) {
            const int index = base_index + offset;
            if (index >= kArrayLength) {
                break;
            }

            const int index_j = index ^ j;

            if (index_j > index && index_j < kArrayLength) {
                float4 value_i = *reinterpret_cast<float4*>(&row_data[index]);
                float4 value_j = *reinterpret_cast<float4*>(&row_data[index_j]);

                const bool sort_up = ascending ? ((index & k) == 0) : ((index & k) != 0);

                if ((value_i.x > value_j.x) == sort_up) {
                    const float temp = value_i.x;
                    value_i.x = value_j.x;
                    value_j.x = temp;
                }
                if ((value_i.y > value_j.y) == sort_up) {
                    const float temp = value_i.y;
                    value_i.y = value_j.y;
                    value_j.y = temp;
                }
                if ((value_i.z > value_j.z) == sort_up) {
                    const float temp = value_i.z;
                    value_i.z = value_j.z;
                    value_j.z = temp;
                }
                if ((value_i.w > value_j.w) == sort_up) {
                    const float temp = value_i.w;
                    value_i.w = value_j.w;
                    value_j.w = temp;
                }
                *reinterpret_cast<float4*>(&row_data[index]) = value_i;
                *reinterpret_cast<float4*>(&row_data[index_j]) = value_j;
            }
        }
        return;
    }

#pragma unroll
    for (int offset = 0; offset < kWorkPerThread; ++offset) {
        const int index = base_index + offset;
        if (index >= kArrayLength) {
            break;
        }

        const int index_j = index ^ j;

        if (index_j > index && index_j < kArrayLength) {
            const float value_i = row_data[index];
            const float value_j = row_data[index_j];

            const bool sort_up = ascending ? ((index & k) == 0) : ((index & k) != 0);

            if ((value_i > value_j) == sort_up) {
                row_data[index] = value_j;
                row_data[index_j] = value_i;
            }
        }
    }
}

}  // namespace

int main() {
    static_assert(kArrayLength % kLocalSize == 0, "kLocalSize must evenly divide kArrayLength");

    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    const size_t num_elements = static_cast<size_t>(kBatchSize) * kArrayLength;
    const size_t total_bytes = num_elements * sizeof(float);

    std::vector<float> host_input(num_elements);
    std::vector<float> host_golden(num_elements);
    std::vector<float> host_result(num_elements, 0.0f);

    std::generate(host_input.begin(), host_input.end(), [&]() { return distribution(generator); });

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    SortCpu(host_input.data(), host_golden.data(), kBatchSize, kArrayLength);
    const auto cpu_stop = std::chrono::high_resolution_clock::now();
    const auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
    std::cout << "CPU sort duration: " << cpu_duration << " us" << std::endl;

    float* device_input = nullptr;

    CheckCuda(cudaMalloc(&device_input, total_bytes), "cudaMalloc device_input");
    CheckCuda(cudaMemcpy(device_input, host_input.data(), total_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input to device_input");

    const dim3 threads_local(kThreadsPerBlock);
    const dim3 blocks_local(kArrayLength / kLocalSize, kBatchSize);
    const dim3 threads(kThreadsPerBlock);
    const dim3 blocks(CeilDiv(kArrayLength, kThreadsPerBlock * kWorkPerThread), kBatchSize);

    const size_t shared_memory_bytes = (kLocalSize + kLocalPad) * sizeof(float);

    cudaDeviceProp device_properties{};
    CheckCuda(cudaGetDeviceProperties(&device_properties, 0), "cudaGetDeviceProperties");
    if (shared_memory_bytes > device_properties.sharedMemPerBlock) {
        CheckCuda(cudaFuncSetAttribute(BitonicBlockSortShared,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(shared_memory_bytes)),
                  "cudaFuncSetAttribute BitonicBlockSortShared");
    }

    // Warm-up to hide first-launch overhead in timing.
    for (int i = 0; i < kWarmupIterations; ++i) {
        BitonicBlockSortShared<<<blocks_local, threads_local, shared_memory_bytes>>>(device_input, kSortAscending);
        CheckCuda(cudaGetLastError(), "BitonicBlockSortShared warmup");
        for (int k = kLocalSize << 1; k <= kArrayLength; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                BitonicSortFp32Kernel<<<blocks, threads>>>(device_input, j, k, kSortAscending);
                CheckCuda(cudaGetLastError(), "BitonicSortFp32Kernel warmup");
            }
        }
    }

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start_event");
    CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop_event");

    CheckCuda(cudaMemcpy(device_input, host_input.data(), total_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input before run");
    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord start_event");
    BitonicBlockSortShared<<<blocks_local, threads_local, shared_memory_bytes>>>(device_input, kSortAscending);
    CheckCuda(cudaGetLastError(), "BitonicBlockSortShared run");
    for (int k = (kLocalSize << 1); k <= kArrayLength; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            BitonicSortFp32Kernel<<<blocks, threads>>>(device_input, j, k, kSortAscending);
            CheckCuda(cudaGetLastError(), "BitonicSortFp32Kernel run");
        }
    }
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord stop_event");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop_event");

    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
    std::cout << "BitonicSortFp32Kernel duration: " << elapsed_ms * 1000.0f << " us" << std::endl;
    const double reached_mem_bw =
        static_cast<double>(total_bytes) * 2.0 / (static_cast<double>(elapsed_ms) / 1000.0) /
        static_cast<double>(1ULL << 40);
    std::cout << "BitonicSortFp32Kernel reached_mem_bw: " << reached_mem_bw << " TB/s" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), device_input, total_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device_input to host_result");
    const float error = ComputeRmse(host_golden.data(), host_result.data(), num_elements);
    std::cout << "BitonicSortFp32Kernel error = " << error << std::endl;

    CheckCuda(cudaEventDestroy(start_event), "cudaEventDestroy start_event");
    CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop_event");
    CheckCuda(cudaFree(device_input), "cudaFree device_input");

    return 0;
}
