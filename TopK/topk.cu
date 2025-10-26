// Use Radix Select to find the top-k elements along the last dimension of a tensor, return their indices.

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BATCH_SIZE 4
#define FEATURE_SIZE 128 * 1024
#define TOPK 2048
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define WARP_COUNT (THREADS_PER_BLOCK / WARP_SIZE)
#define RADIX_BITS 8
#define RADIX_HISTOGRAM_SIZE (1 << RADIX_BITS)
#define RADIX_PASSES (32 / RADIX_BITS)
#define WORK_PER_THREAD 16
#define WORK_PER_BLOCK (THREADS_PER_BLOCK * WORK_PER_THREAD)

// Kernel to perform one pass of radix select for top-k
// Each block.y processes one batch
// Each block.x processes a chunk of the feature dimension
__global__ void radixSelectTopK_fp32_kernel(
    const float* __restrict__ x,
    int* __restrict__ histogram,
    const int* __restrict__ targetTopK,
    const uint32_t* __restrict__ selectedRadix,
    int iterPass
) {
    // Calculate global thread index
    int batchIdx = blockIdx.y;
    int chunkStart = blockIdx.x * WORK_PER_BLOCK;
    int chunkStop = (blockIdx.x + 1) * WORK_PER_BLOCK;
    int tx = threadIdx.x;

    x += batchIdx * FEATURE_SIZE;
    histogram += batchIdx * RADIX_HISTOGRAM_SIZE;
    targetTopK += batchIdx;
    selectedRadix += batchIdx;

    // Skip if no more elements needed
    if (*targetTopK == 0) {
        return;
    }

    // Initialize local histogram
    __shared__ int sharedHistogram[RADIX_HISTOGRAM_SIZE];
    if (tx < RADIX_HISTOGRAM_SIZE) {
        sharedHistogram[tx] = 0;
    }
    __syncthreads();

    // Number of bits to right shift
    const int shiftBits = (RADIX_PASSES - iterPass - 1) * RADIX_BITS;
    const uint32_t mask = (1U << RADIX_BITS) - 1;
    // Loop over the elements in the chunk and build histogram
    if (iterPass == 0) {
        for (int i = chunkStart + tx * 4; i < chunkStop; i += blockDim.x * 4) {
            if (i >= FEATURE_SIZE) {
                break;
            }

            // If it is the first pass, we need to convert the float to uint32
            // For positive floats, we flip the sign bit to 1
            // For negative floats, we flip all bits

            uint4 uval = *((uint4*)&x[i]);
            if (~(uval.x >> 31)) {
                uval.x |= (1 << 31);
            } else {
                uval.x = ~uval.x;
            }
            if (~(uval.y >> 31)) {
                uval.y |= (1 << 31);
            } else {
                uval.y = ~uval.y;
            }
            if (~(uval.z >> 31)) {
                uval.z |= (1 << 31);
            } else {
                uval.z = ~uval.z;
            }
            if (~(uval.w >> 31)) {
                uval.w |= (1 << 31);
            } else {
                uval.w = ~uval.w;
            }

            // Write back the converted value
            *((uint4*)&x[i]) = uval;

            uval.x = uval.x >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.x - 1], 1);
            uval.y = uval.y >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.y - 1], 1);
            uval.z = uval.z >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.z - 1], 1);
            uval.w = uval.w >> shiftBits;
            atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.w - 1], 1);
        }
    } else {
        for (int i = chunkStart + tx * 4; i < chunkStop; i += blockDim.x * 4) {
            if (i >= FEATURE_SIZE) {
                break;
            }

            uint4 uval = *((uint4*)&x[i]);
            if (uval.x >> (shiftBits + RADIX_BITS) == *selectedRadix) {
                uval.x = (uval.x >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.x - 1], 1);
            }
            if (uval.y >> (shiftBits + RADIX_BITS) == *selectedRadix) {
                uval.y = (uval.y >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.y - 1], 1);
            }
            if (uval.z >> (shiftBits + RADIX_BITS) == *selectedRadix) {
                uval.z = (uval.z >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.z - 1], 1);
            }
            if (uval.w >> (shiftBits + RADIX_BITS) == *selectedRadix) {
                uval.w = (uval.w >> shiftBits) & mask;
                atomicAdd(&sharedHistogram[(1 << RADIX_BITS) - uval.w - 1], 1);
            }
        }

    }

    __syncthreads();

    // Compute exclusive prefix sum of sharedHistogram using Blelloch's scan
    if (tx < RADIX_HISTOGRAM_SIZE) {
        // Upsweep phase
        for (int offset = 1; offset < RADIX_HISTOGRAM_SIZE; offset *= 2) {
            if ((tx + 1) % (offset * 2) == 0) {
                sharedHistogram[tx] += sharedHistogram[tx - offset];
            }
            __syncthreads();
        }

        // Set last element to zero
        if (tx == RADIX_HISTOGRAM_SIZE - 1) {
            sharedHistogram[tx] = 0;
        }
        __syncthreads();
        // Downsweep phase
        for (int offset = RADIX_HISTOGRAM_SIZE / 2; offset >= 1; offset /= 2) {
            if ((tx + 1) % (offset * 2) == 0) {
                int temp = sharedHistogram[tx - offset];
                sharedHistogram[tx - offset] = sharedHistogram[tx];
                sharedHistogram[tx] += temp;
            }
            __syncthreads();
        }

        // Write back to global histogram
        if (tx < RADIX_HISTOGRAM_SIZE - 1) {
            atomicAdd(&histogram[tx], sharedHistogram[tx + 1]);
        }
    }

    
}

// Kernel to determine the radix bucket that contains the top-k elements
__global__ void findRadixBucketKernel(
    int* __restrict__ histogram,
    int* __restrict__ targetTopK,
    uint32_t* __restrict__ selectedRadix,
    int iterPass
) {
    int batchIdx = blockIdx.x;
    histogram += batchIdx * RADIX_HISTOGRAM_SIZE;
    targetTopK += batchIdx;
    selectedRadix += batchIdx;

    // Skip if no more elements needed
    if (*targetTopK == 0) {
        return;
    }

    int tx = threadIdx.x;

    // Find the smallest bucket that contains at least targetTopK elements
    if (tx < RADIX_HISTOGRAM_SIZE) {
        if (tx == 0 && histogram[tx] >= *targetTopK) {
            *selectedRadix <<= RADIX_BITS;
            *selectedRadix |= (1 << RADIX_BITS) - tx - 1;
        } else if (tx > 0 && histogram[tx - 1] < *targetTopK && histogram[tx] >= *targetTopK) {
            *selectedRadix <<= RADIX_BITS;
            *selectedRadix |= (1 << RADIX_BITS) - tx - 1;
            *targetTopK -= histogram[tx - 1];
            if (*targetTopK == 0) {
                *selectedRadix <<= (RADIX_PASSES - iterPass - 1) * RADIX_BITS;
            }
        }
    }

    // Reset histogram for next pass
    if (tx < RADIX_HISTOGRAM_SIZE) {
        histogram[tx] = 0;
    }
}

// Final kernel to gather top-k indices given the selected radix
// Each block.y processes one batch
// Each block.x processes a chunk of the feature dimension
__global__ void gatherTopKIndicesKernel(
    const float* __restrict__ x,
    bool* __restrict__ topkIndices,
    uint32_t* selectedRadix
) {
    int batchIdx = blockIdx.y;
    selectedRadix += batchIdx;

    int chunkStart = blockIdx.x * WORK_PER_BLOCK;
    int chunkStop = (blockIdx.x + 1) * WORK_PER_BLOCK;
    int tx = threadIdx.x;
    x += batchIdx * FEATURE_SIZE;
    topkIndices += batchIdx * FEATURE_SIZE;
    // Loop over the elements in the chunk and mark top-k indices
    for (int i = chunkStart + tx * 4; i < chunkStop; i += blockDim.x * 4) {
        if (i >= FEATURE_SIZE) {
            break;
        }
        uint4 uval = *((uint4*)&x[i]);
        topkIndices[i] = (uval.x >= *selectedRadix);
        topkIndices[i + 1] = (uval.y >= *selectedRadix);
        topkIndices[i + 2] = (uval.z >= *selectedRadix);
        topkIndices[i + 3] = (uval.w >= *selectedRadix);
    }
}

int main() {
    // define random generator
    assert(FEATURE_SIZE % 4 == 0);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1);
    // allocate and initialize host memory
    size_t input_size = BATCH_SIZE * FEATURE_SIZE;
    size_t output_size = BATCH_SIZE * TOPK;
    float* x = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        x[i] = distribution(generator);
    }
    int* result = new int[output_size];

    // CPU implementation of top-k (for validation)
    // first sort the entire feature dimension and then select top-k indices
    int* sorted_indices = new int[BATCH_SIZE * FEATURE_SIZE];
    int* resultGolden = new int[output_size];
    for (size_t row = 0; row < BATCH_SIZE; ++row) {
        std::iota(&sorted_indices[row * FEATURE_SIZE], &sorted_indices[(row + 1) * FEATURE_SIZE], 0);
        std::sort(&sorted_indices[row * FEATURE_SIZE], &sorted_indices[(row + 1) * FEATURE_SIZE],
                  [&x, row](int a, int b) {
                      return x[row * FEATURE_SIZE + a] > x[row * FEATURE_SIZE + b];
                  });
        std::copy_n(&sorted_indices[row * FEATURE_SIZE], TOPK, &resultGolden[row * TOPK]);
        // sort the result for comparing
        std::sort(&resultGolden[row * TOPK], &resultGolden[(row + 1) * TOPK]);
    }

    // GPU implementation
    float* xDevice = nullptr;
    bool* topkIndicesDevice = nullptr;
    int* histogramDevice = nullptr;
    int* targetTopK = new int[BATCH_SIZE];
    std::fill_n(targetTopK, BATCH_SIZE, TOPK);
    int* targetTopKDevice = nullptr;
    uint32_t* selectedRadixDevice = nullptr;

    cudaMalloc(&xDevice, input_size * sizeof(float));
    cudaMalloc(&topkIndicesDevice, input_size * sizeof(bool));
    cudaMalloc(&targetTopKDevice, BATCH_SIZE * sizeof(int));
    cudaMalloc(&histogramDevice, BATCH_SIZE * RADIX_HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&selectedRadixDevice, BATCH_SIZE * sizeof(uint32_t));
    cudaMemcpy(targetTopKDevice, targetTopK, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xDevice, x, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize histogram to zero
    cudaMemset(histogramDevice, 0, BATCH_SIZE * RADIX_HISTOGRAM_SIZE * sizeof(int));

    // Initialize selectedRadix to zero
    cudaMemset(selectedRadixDevice, 0, BATCH_SIZE * sizeof(uint32_t));

    // Initialize topkIndices to false
    cudaMemset(topkIndicesDevice, 0, input_size * sizeof(bool));

    // Kernel launch parameters
    // each block processes WORK_PER_BLOCK elements
    int numThreads = THREADS_PER_BLOCK;
    dim3 numBlocks(CEIL_DIV(FEATURE_SIZE, WORK_PER_BLOCK), BATCH_SIZE);

    // For profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        for (int iterPass = 0; iterPass < RADIX_PASSES; ++iterPass) {
            // Launch radix select kernel
            radixSelectTopK_fp32_kernel<<<numBlocks, numThreads>>>(
                xDevice,
                histogramDevice,
                targetTopKDevice,
                selectedRadixDevice,
                iterPass
            );

            // Launch find radix bucket kernel
            findRadixBucketKernel<<<BATCH_SIZE, RADIX_HISTOGRAM_SIZE>>>(
                histogramDevice,
                targetTopKDevice,
                selectedRadixDevice,
                iterPass
            );
        }
        gatherTopKIndicesKernel<<<numBlocks, numThreads>>>(
            xDevice,
            topkIndicesDevice,
            selectedRadixDevice
        );
    }

    cudaMemcpy(targetTopKDevice, targetTopK, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xDevice, x, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(histogramDevice, 0, BATCH_SIZE * RADIX_HISTOGRAM_SIZE * sizeof(int));
    cudaMemset(selectedRadixDevice, 0, BATCH_SIZE * sizeof(uint32_t));
    cudaMemset(topkIndicesDevice, 0, input_size * sizeof(bool));

    cudaEventRecord(start, 0);
    for (int iterPass = 0; iterPass < RADIX_PASSES; ++iterPass) {
        // Launch radix select kernel
        radixSelectTopK_fp32_kernel<<<numBlocks, numThreads>>>(
            xDevice,
            histogramDevice,
            targetTopKDevice,
            selectedRadixDevice,
            iterPass
        );

        // Launch find radix bucket kernel
        findRadixBucketKernel<<<BATCH_SIZE, RADIX_HISTOGRAM_SIZE>>>(
            histogramDevice,
            targetTopKDevice,
            selectedRadixDevice,
            iterPass
        );
    }

    // Launch gather top-k indices kernel
    gatherTopKIndicesKernel<<<numBlocks, numThreads>>>(
        xDevice,
        topkIndicesDevice,
        selectedRadixDevice
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Radix select top-k kernel time: " << elapsedTime * 1000 << " us" << std::endl;

    // Copy back the top-k indices
    bool* topkIndicesHost = new bool[input_size];
    cudaMemcpy(topkIndicesHost, topkIndicesDevice, input_size * sizeof(bool), cudaMemcpyDeviceToHost);
    // Extract the top-k indices from the boolean mask
    for (size_t row = 0; row < BATCH_SIZE; ++row) {
        int count = 0;
        for (size_t col = 0; col < FEATURE_SIZE; ++col) {
            if (topkIndicesHost[row * FEATURE_SIZE + col]) {
                result[row * TOPK + count] = col;
                count++;
                if (count == TOPK) {
                    break;
                }
            }
        }
        std::sort(&result[row * TOPK], &result[(row + 1) * TOPK]);
    }

    // Validate the result by computing set difference of result and resultGolden using std::set_difference
    for (size_t row = 0; row < BATCH_SIZE; ++row) {
        std::vector<int> diff1;
        std::set_difference(
            &resultGolden[row * TOPK],
            &resultGolden[(row + 1) * TOPK],
            &result[row * TOPK],
            &result[(row + 1) * TOPK],
            std::back_inserter(diff1)
        );
        std::set_difference(
            &result[row * TOPK],
            &result[(row + 1) * TOPK],
            &resultGolden[row * TOPK],
            &resultGolden[(row + 1) * TOPK],
            std::back_inserter(diff1)
        );
        std::cout << "Number of different indices in batch " << row << ": " << diff1.size() << std::endl;
    }

    // clean up
    cudaFree(xDevice);
    cudaFree(topkIndicesDevice);
    delete[] x;
    delete[] result;
    delete[] sorted_indices;
    delete[] resultGolden;
    delete[] topkIndicesHost;
    delete[] targetTopK;
    cudaFree(histogramDevice);
    cudaFree(targetTopKDevice);
    cudaFree(selectedRadixDevice);

    return 0;
}