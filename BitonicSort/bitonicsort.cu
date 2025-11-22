// compilation: nvcc -o bitonicsort.o -gencode=arch=compute_120,code=sm_120 bitonicsort.cu -O3
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define BATCH_SIZE 4
#define ARRAY_LENGTH (128 * 1024)
#define THREADS_PER_BLOCK 256
#define ENABLE_BITONIC_KERNEL 0
#define WORK_PER_THREAD 1

#define LOCAL_SIZE THREADS_PER_BLOCK

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

__global__ void BitonicBlockSortShared(
    float* data,        // [batch_size, N], row-major
    bool ascending
) {
    // One block = one segment of length LOCAL_SIZE in one row.
    // Grid: dim3 grid(num_segments_per_row, batch_size)

    __shared__ float sdata[LOCAL_SIZE];   // size = LOCAL_SIZE * sizeof(T)

    int row      = blockIdx.y;                     // which row
    int bx   = blockIdx.x;                     // which segment in this row
    int tid      = threadIdx.x;                    // 0..LOCAL_SIZE-1
    int segmentStart = bx * LOCAL_SIZE;       // index within row
    int iGlobal      = segmentStart + tid;       // global index in row [0..N)

    // We assume:
    //   row < BATCH_SIZE
    //   iGlobal < N
    // ensured by launch: grid.y = batch_size, grid.x = N/LOCAL_SIZE

    data += row * ARRAY_LENGTH;

    // Load local segment into shared memory
    sdata[tid] = data[iGlobal];
    __syncthreads();

    // Run full bitonic sort *for this segment* in shared memory.
    // IMPORTANT: we use the *global* index i_global in the direction test
    // so the pattern matches the global network for k <= LOCAL_SIZE.
    for (int k = 2; k <= LOCAL_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixjGlobal = iGlobal ^ j;                  // partner (global index)
            int ixjLocal  = ixjGlobal - segmentStart;    // partner (local index)

            if (ixjGlobal > iGlobal && ixjLocal >= 0 && ixjLocal < LOCAL_SIZE) {
                float x_i = sdata[tid];
                float x_j = sdata[ixjLocal];

                bool sort_up = ascending
                               ? ((iGlobal & k) == 0)
                               : ((iGlobal & k) != 0);

                if ((x_i > x_j) == sort_up) {
                    sdata[tid]      = x_j;
                    sdata[ixjLocal] = x_i;
                }
            }
            __syncthreads();
        }
    }

    // Store back the sorted segment
    data[iGlobal] = sdata[tid];
}

__global__ void BitonicSortFp32Kernel(
    float* data,          // pointer to data of shape [batch_size, N], row-major
    int j,
    int k,
    bool ascending    // true = sort each row ascending
) {
    int row = blockIdx.y;  // which batch row
    int baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * WORK_PER_THREAD;  // starting index this thread handles

    data += row * ARRAY_LENGTH;

    if (row >= BATCH_SIZE || baseIdx >= ARRAY_LENGTH) {
        return;
    }

#pragma unroll
    for (int t = 0; t < WORK_PER_THREAD; ++t) {
        int idx = baseIdx + t;
        if (idx >= ARRAY_LENGTH) {
            break;
        }

        int idxj = idx ^ j;    // partner index within the row

        if (idxj > idx && idxj < ARRAY_LENGTH) {
            float x_i  = data[idx];
            float x_j  = data[idxj];

            // Determine direction (ascending / descending) for this pair:
            bool sort_up = ascending ? ((idx & k) == 0) : ((idx & k) != 0);

            // If we want ascending and x_i > x_j, or want descending and x_i < x_j, swap.
            if ((x_i > x_j) == sort_up) {
                data[idx]  = x_j;
                data[idxj] = x_i;
            }
        }
    }
}


float ComputeRMSE(const float* __restrict__ golden, const float* __restrict__ x, size_t numElements) {
    float error = 0;
    float norm = 0;
    for (size_t i = 0; i < numElements; ++i) {
        error += (golden[i] - x[i]) * (golden[i] - x[i]);
        norm += golden[i] * golden[i];
    }
    return std::sqrt(error) / std::sqrt(norm);
}

void SortCPU(const float* __restrict__ input, float* __restrict__ output, int batchSize, int arrayLength) {
    std::vector<float> buffer(arrayLength);
    for (int batch = 0; batch < batchSize; ++batch) {
        const float* rowIn = input + batch * arrayLength;
        float* rowOut = output + batch * arrayLength;
        std::copy(rowIn, rowIn + arrayLength, buffer.begin());
        std::sort(buffer.begin(), buffer.end());
        std::copy(buffer.begin(), buffer.end(), rowOut);
    }
}


int main() {
    static_assert((ARRAY_LENGTH & (ARRAY_LENGTH - 1)) == 0, "ARRAY_LENGTH must be power of two for bitonic sort");

    // generate random seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    const size_t numElements = static_cast<size_t>(BATCH_SIZE) * ARRAY_LENGTH;
    const size_t totalBytes = numElements * sizeof(float);

    float* input = new float[numElements];
    float* golden = new float[numElements];

    for (size_t i = 0; i < numElements; ++i) {
        input[i] = distribution(generator);
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    SortCPU(input, golden, BATCH_SIZE, ARRAY_LENGTH);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuStop - cpuStart).count();
    std::cout << "CPU sort duration: " << cpuDuration << " us" << std::endl;

    float* inputDevice;
    float* outputDevice;
    float* result = new float[numElements];

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&inputDevice, totalBytes);
    cudaMalloc(&outputDevice, totalBytes);
    cudaMemcpy(inputDevice, input, totalBytes, cudaMemcpyHostToDevice);

    dim3 numThreadsLocal(THREADS_PER_BLOCK);                 // BLOCK_SIZE == LOCAL_SIZE
    dim3 numBlocksLocal(ARRAY_LENGTH / LOCAL_SIZE, BATCH_SIZE); 

    dim3 numThreads(THREADS_PER_BLOCK);
    dim3 numBlocks(CEIL_DIV(ARRAY_LENGTH, numThreads.x * WORK_PER_THREAD), BATCH_SIZE);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        BitonicBlockSortShared<<<numBlocksLocal, numThreadsLocal>>>(inputDevice, true);
        for (int k = LOCAL_SIZE << 1; k <= ARRAY_LENGTH; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                BitonicSortFp32Kernel<<<numBlocks, numThreads>>>(inputDevice, j, k, true);
            }
        }
    }


    // Outer loop over k and j (bitonic network)
    cudaMemcpy(inputDevice, input, totalBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    BitonicBlockSortShared<<<numBlocksLocal, numThreadsLocal>>>(inputDevice, true);
    for (int k = (LOCAL_SIZE << 1); k <= ARRAY_LENGTH; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            BitonicSortFp32Kernel<<<numBlocks, numThreads>>>(inputDevice, j, k, true);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "BitonicSortFp32Kernel duration: " << elapsedTime * 1000 << " us" << std::endl;
    float reachedMemBW = totalBytes * 2.0 / ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "BitonicSortFp32Kernel reachedMemBW: " << reachedMemBW << " TB/s" << std::endl;

    cudaMemcpy(result, inputDevice, totalBytes, cudaMemcpyDeviceToHost);
    float error = ComputeRMSE(golden, result, numElements);
    std::cout << "BitonicSortFp32Kernel error = " << error << std::endl;

    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete[] result;

    delete[] input;
    delete[] golden;
}
