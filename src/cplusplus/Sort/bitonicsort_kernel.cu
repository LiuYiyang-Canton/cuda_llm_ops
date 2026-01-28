// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-29
// Purpose: CUDA kernel implementation for bitonic sort.
// ==============================================================================
#include "Sort/bitonicsort_kernel.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWorkPerThread = 4;
constexpr int kLocalWorkPerThread = 4;
constexpr int kVectorWidth = 4;
constexpr int kPadding = 4;
constexpr int kWarpSize = 32;
constexpr int kLocalSize = kThreadsPerBlock * kLocalWorkPerThread;
constexpr int kLocalPad = (kLocalSize / kWarpSize) * kPadding;

/**
 * @brief Returns ceil(numerator / denominator) for positive integers.
 * @param numerator Dividend value.
 * @param denominator Divisor value; must be > 0.
 * @return Rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Sorts a local segment of one row using shared memory.
 * @param data Pointer to device buffer [batchSize, arrayLength].
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 * @param ascending Whether to sort ascending.
 */
__global__ void BitonicBlockSortShared(float* data, int batchSize, int arrayLength, bool ascending) {
    extern __shared__ float shared[];

    const int row = static_cast<int>(blockIdx.y);
    const int blockSegment = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int segmentStart = blockSegment * kLocalSize;

    if (row >= batchSize) {
        return;
    }

    float* rowData = data + row * arrayLength;

#pragma unroll
    for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
        const int localIndex = thread * kLocalWorkPerThread + offset;
        const int globalIndex = segmentStart + localIndex;
        const int paddedIndex = localIndex + (localIndex / kWarpSize) * kPadding;
        *reinterpret_cast<float4*>(&shared[paddedIndex]) =
            *reinterpret_cast<const float4*>(&rowData[globalIndex]);
    }
    __syncthreads();

    for (int k = 2; k <= kLocalSize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= kVectorWidth) {
#pragma unroll
                for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
                    const int localIndex = thread * kLocalWorkPerThread + offset;
                    const int globalIndex = segmentStart + localIndex;
                    const int ixjGlobal = globalIndex ^ j;
                    const int ixjLocal = ixjGlobal - segmentStart;

                    if (ixjGlobal > globalIndex && ixjLocal >= 0 && ixjLocal < kLocalSize) {
                        const int paddedIndex = localIndex + (localIndex / kWarpSize) * kPadding;
                        const int paddedIndexJ = ixjLocal + (ixjLocal / kWarpSize) * kPadding;
                        float4 valueI = *reinterpret_cast<float4*>(&shared[paddedIndex]);
                        float4 valueJ = *reinterpret_cast<float4*>(&shared[paddedIndexJ]);

                        const bool sortUp = ascending ? ((globalIndex & k) == 0) : ((globalIndex & k) != 0);

                        if ((valueI.x > valueJ.x) == sortUp) {
                            const float temp = valueI.x;
                            valueI.x = valueJ.x;
                            valueJ.x = temp;
                        }
                        if ((valueI.y > valueJ.y) == sortUp) {
                            const float temp = valueI.y;
                            valueI.y = valueJ.y;
                            valueJ.y = temp;
                        }
                        if ((valueI.z > valueJ.z) == sortUp) {
                            const float temp = valueI.z;
                            valueI.z = valueJ.z;
                            valueJ.z = temp;
                        }
                        if ((valueI.w > valueJ.w) == sortUp) {
                            const float temp = valueI.w;
                            valueI.w = valueJ.w;
                            valueJ.w = temp;
                        }
                        *reinterpret_cast<float4*>(&shared[paddedIndex]) = valueI;
                        *reinterpret_cast<float4*>(&shared[paddedIndexJ]) = valueJ;
                    }
                    __syncthreads();
                }
            } else {
                for (int offset = 0; offset < kLocalWorkPerThread; ++offset) {
                    const int localIndex = thread * kLocalWorkPerThread + offset;
                    const int globalIndex = segmentStart + localIndex;
                    const int ixjGlobal = globalIndex ^ j;
                    const int ixjLocal = ixjGlobal - segmentStart;

                    if (ixjGlobal > globalIndex && ixjLocal >= 0 && ixjLocal < kLocalSize) {
                        const int paddedIndex = localIndex + (localIndex / kWarpSize) * kPadding;
                        const int paddedIndexJ = ixjLocal + (ixjLocal / kWarpSize) * kPadding;
                        const float valueI = shared[paddedIndex];
                        const float valueJ = shared[paddedIndexJ];

                        const bool sortUp = ascending ? ((globalIndex & k) == 0) : ((globalIndex & k) != 0);

                        if ((valueI > valueJ) == sortUp) {
                            shared[paddedIndex] = valueJ;
                            shared[paddedIndexJ] = valueI;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

#pragma unroll
    for (int offset = 0; offset < kLocalWorkPerThread; offset += kVectorWidth) {
        const int localIndex = thread * kLocalWorkPerThread + offset;
        const int globalIndex = segmentStart + localIndex;
        const int paddedIndex = localIndex + (localIndex / kWarpSize) * kPadding;
        *reinterpret_cast<float4*>(&rowData[globalIndex]) =
            *reinterpret_cast<float4*>(&shared[paddedIndex]);
    }
}

/**
 * @brief Bitonic merge network on global memory for each row.
 * @param data Pointer to device buffer [batchSize, arrayLength].
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 * @param j Current stride in the merge network.
 * @param k Current subsequence length in the merge network.
 * @param ascending Whether the sort order is ascending.
 */
__global__ void BitonicSortFp32Kernel(float* data, int batchSize, int arrayLength, int j, int k, bool ascending) {
    const int row = static_cast<int>(blockIdx.y);
    const int baseIndex = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                           static_cast<int>(threadIdx.x)) * kWorkPerThread;

    float* rowData = data + row * arrayLength;

    if (row >= batchSize || baseIndex >= arrayLength) {
        return;
    }

    if (j >= kVectorWidth) {
#pragma unroll
        for (int offset = 0; offset < kWorkPerThread; offset += kVectorWidth) {
            const int index = baseIndex + offset;
            if (index >= arrayLength) {
                break;
            }

            const int indexJ = index ^ j;

            if (indexJ > index && indexJ < arrayLength) {
                float4 valueI = *reinterpret_cast<float4*>(&rowData[index]);
                float4 valueJ = *reinterpret_cast<float4*>(&rowData[indexJ]);

                const bool sortUp = ascending ? ((index & k) == 0) : ((index & k) != 0);

                if ((valueI.x > valueJ.x) == sortUp) {
                    const float temp = valueI.x;
                    valueI.x = valueJ.x;
                    valueJ.x = temp;
                }
                if ((valueI.y > valueJ.y) == sortUp) {
                    const float temp = valueI.y;
                    valueI.y = valueJ.y;
                    valueJ.y = temp;
                }
                if ((valueI.z > valueJ.z) == sortUp) {
                    const float temp = valueI.z;
                    valueI.z = valueJ.z;
                    valueJ.z = temp;
                }
                if ((valueI.w > valueJ.w) == sortUp) {
                    const float temp = valueI.w;
                    valueI.w = valueJ.w;
                    valueJ.w = temp;
                }
                *reinterpret_cast<float4*>(&rowData[index]) = valueI;
                *reinterpret_cast<float4*>(&rowData[indexJ]) = valueJ;
            }
        }
        return;
    }

#pragma unroll
    for (int offset = 0; offset < kWorkPerThread; ++offset) {
        const int index = baseIndex + offset;
        if (index >= arrayLength) {
            break;
        }

        const int indexJ = index ^ j;

        if (indexJ > index && indexJ < arrayLength) {
            const float valueI = rowData[index];
            const float valueJ = rowData[indexJ];

            const bool sortUp = ascending ? ((index & k) == 0) : ((index & k) != 0);

            if ((valueI > valueJ) == sortUp) {
                rowData[index] = valueJ;
                rowData[indexJ] = valueI;
            }
        }
    }
}

}  // namespace

/**
 * @brief Launches the bitonic sort kernels for fp32 data.
 * @param data Pointer to device input/output buffer of shape [batchSize, arrayLength].
 * @param batchSize Number of rows to sort.
 * @param arrayLength Number of elements per row.
 * @param ascending True to sort ascending, false for descending.
 */
void LaunchBitonicSortFp32Kernel(float* data, int batchSize, int arrayLength, bool ascending) {
    static_assert(kLocalWorkPerThread % kVectorWidth == 0, "Vectorized work must align with float4 width");
    if (batchSize <= 0 || arrayLength <= 0) {
        return;
    }
    if ((arrayLength & (arrayLength - 1)) != 0) {
        return;
    }
    if (arrayLength % kLocalSize != 0) {
        return;
    }

    const dim3 threadsLocal(kThreadsPerBlock);
    const dim3 blocksLocal(arrayLength / kLocalSize, batchSize);
    const dim3 threads(kThreadsPerBlock);
    const dim3 blocks(CeilDiv(arrayLength, kThreadsPerBlock * kWorkPerThread), batchSize);

    const size_t sharedMemoryBytes = static_cast<size_t>(kLocalSize + kLocalPad) * sizeof(float);

    cudaDeviceProp deviceProperties{};
    cudaGetDeviceProperties(&deviceProperties, 0);
    if (sharedMemoryBytes > deviceProperties.sharedMemPerBlock) {
        cudaFuncSetAttribute(BitonicBlockSortShared,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sharedMemoryBytes));
    }

    BitonicBlockSortShared<<<blocksLocal, threadsLocal, sharedMemoryBytes>>>(data,
                                                                             batchSize,
                                                                             arrayLength,
                                                                             ascending);
    for (int k = kLocalSize << 1; k <= arrayLength; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            BitonicSortFp32Kernel<<<blocks, threads>>>(data, batchSize, arrayLength, j, k, ascending);
        }
    }
}
