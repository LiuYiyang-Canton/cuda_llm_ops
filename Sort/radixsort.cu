// This file implements an LSD (least-significant-digit first) radix sort for 32-bit floats:
// each pass processes 8 bits using a three-step pipeline of per-block histogramming, a global
// prefix-sum across blocks, and a scatter into the correct bucket. On the first pass, floats
// are remapped to sortable uint32 keys (so negative numbers compare correctly); after the final
// pass the keys are converted back to floats. Arrays are tiled per block with shared memory
// histograms for locality and warp-level coordination to reduce contention.
// Compilation command: nvcc -o radixsort.o -gencode=arch=compute_120,code=sm_120 -O3 radixsort.cu

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kBatchSize = 1;
constexpr int kArrayLength = 1024 * 1024;

#define RADIX_BITS 8
#define RADIX_HISTOGRAM_SIZE (1 << RADIX_BITS)
#define RADIX_PASSES (32 / RADIX_BITS)
#define WORK_PER_THREAD 16
#define THREADS_PER_BLOCK 256
#define WORK_PER_BLOCK (THREADS_PER_BLOCK * WORK_PER_THREAD)
#define NUM_BLOCKS_PER_BATCH CeilDiv(kArrayLength, WORK_PER_BLOCK)

#define CeilDiv(a, b) ((a) + (b) - 1) / (b)

#define WARP_SIZE 32
#define NUM_WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)

/**
 * Transforms raw float bits into a monotonic sortable key.
 *
 * @param bits Raw IEEE754 bits from a float.
 * @return uint32_t Key where lexicographic order matches float ordering.
 */
__device__ __forceinline__ uint32_t FloatToSortKey(uint32_t bits) {
    const uint32_t sign = bits & 0x80000000u;
    // For positive numbers, flip the sign bit to 1; for negative, flip all bits.
    return bits ^ (sign ? 0xFFFFFFFFu : 0x80000000u);
}

/**
 * Converts a sortable key produced by FloatToSortKey back to the original float.
 *
 * @param key_bits Encoded sortable key.
 * @return float Reconstructed floating-point value.
 */
__device__ __forceinline__ float SortKeyToFloat(uint32_t key_bits) {
    // Inverse of FloatToSortKey: sign==1 implies original positive; sign==0 implies original negative.
    const uint32_t sign = key_bits & 0x80000000u;
    const uint32_t raw = sign ? (key_bits ^ 0x80000000u) : (key_bits ^ 0xFFFFFFFFu);
    return __uint_as_float(raw);
}

/**
 * Builds per-block histograms for a single radix pass.
 *
 * @param input float* Input data (in-place keys for pass 0).
 * @param histogram int* Output histogram buffer shaped [batch, blocks, RADIX_HISTOGRAM_SIZE].
 * @param iterPass int Zero-based radix pass index.
 */
__global__ void RadixSortBuildHistogram(
    float* __restrict__ input,
    int* __restrict__ histogram,
    int iterPass
) {
    // Calculate global thread index
    int batchIdx = blockIdx.y;
    const int chunkStart = blockIdx.x * WORK_PER_BLOCK;
    const int chunkStop = (blockIdx.x + 1) * WORK_PER_BLOCK;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Advance pointers to the start of this batch.
    input += batchIdx * kArrayLength;
    histogram += batchIdx * NUM_BLOCKS_PER_BATCH * RADIX_HISTOGRAM_SIZE + bx * RADIX_HISTOGRAM_SIZE;

    // Initialize local histogram
    __shared__ int sharedHistogram[RADIX_HISTOGRAM_SIZE];
    if (tx < RADIX_HISTOGRAM_SIZE) {
        sharedHistogram[tx] = 0;
    }
    __syncthreads();

    // Number of bits to right shift
    const int shiftBits = iterPass * RADIX_BITS;
    const uint32_t mask = (1U << RADIX_BITS) - 1;
    // Loop over the elements in the chunk and build histogram
    if (iterPass == 0) {
        for (int i = chunkStart + tx * 4; i < chunkStop; i += blockDim.x * 4) {
            if (i >= kArrayLength) {
                break;
            }
            // If it is the first pass, we need to convert the float to uint32
            // For positive floats, we flip the sign bit to 1; for negative floats, flip all bits.
            const uint4 uval_f = *reinterpret_cast<const uint4*>(&input[i]);
            uint4 uval{};
            uval.x = FloatToSortKey(uval_f.x);
            uval.y = FloatToSortKey(uval_f.y);
            uval.z = FloatToSortKey(uval_f.z);
            uval.w = FloatToSortKey(uval_f.w);

            // Write back the converted value
            *reinterpret_cast<uint4*>(&input[i]) = uval;

            atomicAdd(&sharedHistogram[(uval.x >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.y >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.z >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.w >> shiftBits) & mask], 1);
        }
    } else {
        for (int i = chunkStart + tx * 4; i < chunkStop; i += blockDim.x * 4) {
            if (i >= kArrayLength) {
                break;
            }
            const uint4 uval = *reinterpret_cast<const uint4*>(&input[i]);
            atomicAdd(&sharedHistogram[(uval.x >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.y >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.z >> shiftBits) & mask], 1);
            atomicAdd(&sharedHistogram[(uval.w >> shiftBits) & mask], 1);
        }
    }

    __syncthreads();
    if (tx < RADIX_HISTOGRAM_SIZE) {
        histogram[tx] = sharedHistogram[tx];
    }
}

/**
 * Computes an exclusive prefix sum over per-block histograms in-place.
 *
 * Layout: histogram is shaped [batch, blocks_per_row, RADIX_HISTOGRAM_SIZE].
 * Launch configuration: grid.x == 1, grid.y == batch_size, block.x >= RADIX_HISTOGRAM_SIZE.
 * 
 * This could be further improved using Blleloch's up/down sweep algorithm. We skip it on
 * laptop GPU fornow.
 */
__global__ void RadixSortPrefixSum(int* histogram) {
    const int batchIdx = static_cast<int>(blockIdx.y);
    const int bucket = static_cast<int>(threadIdx.x);
    const int tx = threadIdx.x;

    // Scratch for bucket totals.
    __shared__ int bucket_totals[RADIX_HISTOGRAM_SIZE];

    if (bucket >= RADIX_HISTOGRAM_SIZE || blockIdx.x != 0) {
        return;
    }

    histogram += batchIdx * NUM_BLOCKS_PER_BATCH * RADIX_HISTOGRAM_SIZE;

    // 1) Exclusive scan per bucket across blocks; also accumulate total count for this bucket.
    int running_total = 0;
    for (int block = 0; block < NUM_BLOCKS_PER_BATCH; ++block) {
        const int count = histogram[block * RADIX_HISTOGRAM_SIZE + tx];
        histogram[block * RADIX_HISTOGRAM_SIZE + tx] =  running_total;
        running_total += count;
    }
    bucket_totals[tx] = running_total;
    __syncthreads();

    // 2) Exclusive scan across buckets to get a global base for each bucket.
    int bucket_base = 0;
    for (int b = 0; b < tx; ++b) {
        bucket_base += bucket_totals[b];
    }
    __syncthreads();

    // 3) Add bucket_base to every entry of this bucket.
    for (int block = 0; block < NUM_BLOCKS_PER_BATCH; ++block) {
        histogram[block * RADIX_HISTOGRAM_SIZE + tx] += bucket_base;
    }
}

/**
 * Scatters elements into the correct output position for the current radix pass.
 *
 * @param input const float* Source buffer (keys for intermediate passes).
 * @param output float* Destination buffer.
 * @param histogram int* Exclusive prefix sums per block and bucket.
 * @param iterPass int Zero-based radix pass index.
 */
__global__ void RadixSortScatter(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ histogram,
    int iterPass
) {
    int batchIdx = blockIdx.y;
    const int chunkStart = blockIdx.x * WORK_PER_BLOCK;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Advance pointers to the start of this batch.
    input += batchIdx * kArrayLength;
    output += batchIdx * kArrayLength;
    histogram += batchIdx * NUM_BLOCKS_PER_BATCH * RADIX_HISTOGRAM_SIZE + bx * RADIX_HISTOGRAM_SIZE;

    // Read global histogram into shared memory
    __shared__ int sharedHistogram[RADIX_HISTOGRAM_SIZE];
    if (tx < RADIX_HISTOGRAM_SIZE) {
        sharedHistogram[tx] = histogram[tx];
    }
    __syncthreads();

    // One 256-bin histogram per warp (8 warps in a 256-thread block).
    __shared__ int warpStart[NUM_WARPS_PER_BLOCK][RADIX_HISTOGRAM_SIZE];

    const int warpID = threadIdx.x / WARP_SIZE;
    const int laneID = threadIdx.x % WARP_SIZE;

    // Zero shared histograms
    for (int i = 0; i < NUM_WARPS_PER_BLOCK; ++i) {
        if (tx < RADIX_HISTOGRAM_SIZE) {
            warpStart[i][tx] = 0;
        }
    }
    __syncthreads();

    const int shiftBits = iterPass * RADIX_BITS;
    const uint32_t bucketMask = (1U << RADIX_BITS) - 1;

    // First pass: per-warp bin counts
    for (int i = chunkStart + warpID * WARP_SIZE * WORK_PER_THREAD + laneID * 4;
            i < chunkStart + (warpID + 1) * WARP_SIZE * WORK_PER_THREAD; i += WARP_SIZE * 4) {
        if (i >= kArrayLength) {
            break;
        }
        const uint4 uval = *reinterpret_cast<const uint4*>(&input[i]);

        const uint32_t b0 = (uval.x >> shiftBits) & bucketMask;
        const uint32_t b1 = (uval.y >> shiftBits) & bucketMask;
        const uint32_t b2 = (uval.z >> shiftBits) & bucketMask;
        const uint32_t b3 = (uval.w >> shiftBits) & bucketMask;

        atomicAdd(&warpStart[warpID][b0], 1);
        atomicAdd(&warpStart[warpID][b1], 1);
        atomicAdd(&warpStart[warpID][b2], 1);
        atomicAdd(&warpStart[warpID][b3], 1);
    }
    __syncthreads();

    // Prefix-sum along the warp dimension (exclusive), per bin.
    if (tx < RADIX_HISTOGRAM_SIZE) {
        int running = 0;
        // For this bin (threadIdx.x), walk warps 0..7
        #pragma unroll
        for (int w = 0; w < NUM_WARPS_PER_BLOCK; ++w) {
            const int count = warpStart[w][threadIdx.x];
            warpStart[w][threadIdx.x] = running; // store exclusive prefix
            running += count;
        }
    }
    __syncthreads();

    for (int i = chunkStart + warpID * WARP_SIZE * WORK_PER_THREAD + laneID;
            i < chunkStart + (warpID + 1) * WARP_SIZE * WORK_PER_THREAD; i += WARP_SIZE) {
        if (i >= kArrayLength) {
            break;
        }
        const uint uval = *reinterpret_cast<const uint*>(&input[i]);
        const uint b = (uval >> shiftBits) & bucketMask;

        // Mask of lanes in this warp whose bucket matches mine.
        const unsigned activeMask = __activemask();
        const unsigned sameBucketMask = __match_any_sync(activeMask, b);  // mask of lanes whose b matches mine

        const int local_in_warp = __popc(sameBucketMask & ((1u << laneID) - 1)); // lanes before me with same bucket
        const int warp_offset   = warpStart[warpID][b];                           // prefix across warps
        const int block_base    = sharedHistogram[b];                             // block start for this bin
        const int pos           = block_base + warp_offset + local_in_warp;

        if (iterPass < RADIX_PASSES - 1) {
            *(uint32_t*)&output[pos] = uval;
        } else {
            output[pos] = SortKeyToFloat(uval);
        }

        // Advance this warp’s running offset for this bucket by the number of lanes that wrote.
        const int total_in_ballot = __popc(sameBucketMask);
        const int leader          = __ffs(sameBucketMask) - 1;
        if (laneID == leader) {
            warpStart[warpID][b] += total_in_ballot;
        }
        __syncwarp();
    }
}

/**
 * Validates CUDA API results and aborts on failure.
 *
 * @param result cudaError_t status code returned by CUDA runtime.
 * @param message const char* human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    // Fail fast so later operations do not run on invalid state.
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
        // Accumulate squared error and a scale factor for normalization.
        const double diff = static_cast<double>(golden[i]) - static_cast<double>(values[i]);
        error += diff * diff;
        norm += static_cast<double>(golden[i]) * static_cast<double>(golden[i]);
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm + 1e-6));
}

/**
 * CPU reference radix sort (row-wise) to produce golden output.
 * Uses std::sort for correctness validation only.
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
        // Copy one row, sort it on CPU, and write it back to the output buffer.
        std::copy(row_in, row_in + array_length, buffer.begin());
        std::sort(buffer.begin(), buffer.end());
        std::copy(buffer.begin(), buffer.end(), row_out);
    }
}

}  // namespace

/**
 * Entrypoint: generates data, runs GPU radix sort, and compares to CPU reference.
 *
 * @return int Exit status.
 */
int main() {
    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static_assert(kArrayLength % 4 == 0, "Array length must be multiples of 4");

    const size_t num_elements = static_cast<size_t>(kBatchSize) * kArrayLength;
    const size_t total_bytes = num_elements * sizeof(float);

    std::vector<float> host_input(num_elements);
    std::vector<float> host_golden(num_elements);
    std::vector<float> host_result(num_elements, 0.0f);

    // Generate random input and prepare CPU reference result.
    std::generate(host_input.begin(), host_input.end(), [&]() { return distribution(generator); });

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    SortCpu(host_input.data(), host_golden.data(), kBatchSize, kArrayLength);
    const auto cpu_stop = std::chrono::high_resolution_clock::now();
    const auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
    std::cout << "CPU radix sort duration: " << cpu_duration << " us" << std::endl;

    float* device_input = nullptr;
    float* device_output = nullptr;
    int* device_histogram = nullptr;

    // Allocate device buffers and stage input data to the GPU.
    CheckCuda(cudaMalloc(&device_input, total_bytes), "cudaMalloc device_input");
    CheckCuda(cudaMalloc(&device_output, total_bytes), "cudaMalloc device_output");
    CheckCuda(cudaMalloc(&device_histogram,
                         static_cast<size_t>(kBatchSize) * NUM_BLOCKS_PER_BATCH * RADIX_HISTOGRAM_SIZE *
                             sizeof(int)),
              "cudaMalloc device_histogram");
    CheckCuda(cudaMemcpy(device_input, host_input.data(), total_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input to device_input");

    // Create CUDA events for timing.
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start_event");
    CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop_event");

    const dim3 block_dim(THREADS_PER_BLOCK);
    const dim3 grid_dim(NUM_BLOCKS_PER_BATCH, kBatchSize);
    const dim3 block_prefix(RADIX_HISTOGRAM_SIZE);
    const dim3 grid_prefix(1, kBatchSize);

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        float* src = device_input;
        float* dst = device_output;
        for (int iter = 0; iter < RADIX_PASSES; ++iter) {
            RadixSortBuildHistogram<<<grid_dim, block_dim>>>(src, device_histogram, iter);
            RadixSortPrefixSum<<<grid_prefix, block_prefix>>>(device_histogram);
            RadixSortScatter<<<grid_dim, block_dim>>>(src, dst, device_histogram, iter);
            if (iter != RADIX_PASSES - 1) {
                std::swap(src, dst);
            }
        }
    }
    CheckCuda(cudaMemcpy(device_input, host_input.data(), total_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input to device_input");

    // Timed GPU radix sort over all passes.
    float* src = device_input;
    float* dst = device_output;
    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord start_event");
    for (int iter = 0; iter < RADIX_PASSES; ++iter) {
        RadixSortBuildHistogram<<<grid_dim, block_dim>>>(src, device_histogram, iter);
        RadixSortPrefixSum<<<grid_prefix, block_prefix>>>(device_histogram);
        RadixSortScatter<<<grid_dim, block_dim>>>(src, dst, device_histogram, iter);
        if (iter != RADIX_PASSES - 1) {
            std::swap(src, dst);
        }
    }

    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord stop_event");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop_event");

    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
            "cudaEventElapsedTime");
    std::cout << "GPU radix sort duration: " << elapsed_ms * 1000 << " us" << std::endl;

    CheckCuda(cudaMemcpy(host_result.data(), dst, total_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device_input to host_result");
    const float error = ComputeRmse(host_golden.data(), host_result.data(), num_elements);
    std::cout << "Radix sort RMSE = " << error << std::endl;

    // Cleanup GPU resources.
    CheckCuda(cudaEventDestroy(start_event), "cudaEventDestroy start_event");
    CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop_event");
    CheckCuda(cudaFree(device_histogram), "cudaFree device_histogram");
    CheckCuda(cudaFree(device_output), "cudaFree device_output");
    CheckCuda(cudaFree(device_input), "cudaFree device_input");

    return 0;
}
