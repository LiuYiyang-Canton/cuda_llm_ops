// CUDA implementation of online safe softmax
// Compilation command: nvcc -o softmax -gencode=arch=compute_120,code=sm_120 -O3 softmax.cu
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define BATCH_SIZE 128
#define FEATURE_SIZE 16384
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define WARP_COUNT (THREADS_PER_BLOCK / WARP_SIZE)

// a struct that holds the max and sum for a tile
// each thread block computes one row's max and sum
struct TileMaxSum {
    float max;
    float sum;
};

// define a reduce operator for TileMaxSum
struct TileMaxSumReduce {
    __device__ TileMaxSum operator()(const TileMaxSum& a, const TileMaxSum& b) const {
        TileMaxSum result;
        result.max = fmaxf(a.max, b.max);
        result.sum = a.sum * __expf(a.max - result.max) + b.sum * __expf(b.max - result.max);
        return result;
    }
};

// Compute rowmax and rowsum
__global__ void compute_rowmax_rowsum_fp32_kernel(const float* __restrict__ x, float* __restrict__ rowmax, float* __restrict__ rowsum) {
    int warpID = threadIdx.x / WARP_SIZE;
    int batch_idx = blockIdx.x;
    int tx = threadIdx.x;
    x += batch_idx * FEATURE_SIZE;
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    TileMaxSum maxandsum;
    maxandsum.max = -INFINITY;
    maxandsum.sum = 0.0f;
    __shared__ TileMaxSum warpMaxSum[WARP_COUNT];

    // each thread processes multiple elements
    for (int col = tx * 4; col < FEATURE_SIZE; col += blockDim.x * 4) {
        float4 val = *(float4*)&x[col];
        TileMaxSum newmaxandsum;
        newmaxandsum.max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
        newmaxandsum.sum = __expf(val.x - newmaxandsum.max) + __expf(val.y - newmaxandsum.max)
                        + __expf(val.z - newmaxandsum.max) + __expf(val.w - newmaxandsum.max);
        maxandsum = TileMaxSumReduce()(maxandsum, newmaxandsum);
    }

    // warp-level reduction
    TileMaxSum warpResult = cg::reduce(warp, maxandsum, TileMaxSumReduce());

    // thread 0 of each warp writes to shared memory
    if (warp.thread_rank() == 0) {
        warpMaxSum[warpID] = warpResult;
    }
    __syncthreads();

    // tree reduction among warps
    for (int offset = WARP_COUNT / 2; offset > 0; offset /= 2) {
        if (tx < offset) {
            warpMaxSum[tx] = TileMaxSumReduce()(warpMaxSum[tx], warpMaxSum[tx + offset]);
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (tx == 0) {
        rowmax[batch_idx] = warpMaxSum[0].max;
        rowsum[batch_idx] = warpMaxSum[0].sum;
    }
}

// Kernel to compute softmax output using precomputed rowmax and rowsum
// Each thread block computes one row of softmax output
__global__ void softmax_fp32_kernel(const float* __restrict__ x, const float* __restrict__ rowmax, const float* __restrict__ rowsum, float* __restrict__ softmax_out) {
    int batch_idx = blockIdx.x;
    int tx = threadIdx.x;
    x += batch_idx * FEATURE_SIZE;
    softmax_out += batch_idx * FEATURE_SIZE;
    float max_val = rowmax[batch_idx];
    float sum_val = rowsum[batch_idx];

    // each thread processes multiple elements
    for (int col = tx * 4; col < FEATURE_SIZE; col += blockDim.x * 4) {
        float4 val = *(float4*)&x[col];
        float4 result;
        result.x = __expf(val.x - max_val) / sum_val;
        result.y = __expf(val.y - max_val) / sum_val;
        result.z = __expf(val.z - max_val) / sum_val;
        result.w = __expf(val.w - max_val) / sum_val;
        *(float4*)&softmax_out[col] = result;
    }
}

// Compute RMSE between golden and x
float ComputeRMSE(const float* __restrict__ golden, const float* __restrict x, size_t numElements) {
    float error = 0;
    float norm = 0;
    for (int i = 0; i < numElements; ++i) {
        error += (golden[i] - x[i]) * (golden[i] - x[i]);
        norm += golden[i] * golden[i];
    }
    return std::sqrt(error) / std::sqrt(norm);
}

int main() {
    assert(FEATURE_SIZE % 4 == 0);

    // define distribution
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float* x = new float[BATCH_SIZE * FEATURE_SIZE];
    float* resultGolden = new float[BATCH_SIZE * FEATURE_SIZE];

    for (int i = 0; i < BATCH_SIZE * FEATURE_SIZE; ++i) {
        x[i] = distribution(generator);
    }

    // CPU implementation
    for (int row = 0; row < BATCH_SIZE; ++row) {
        float max_val = -INFINITY;
        for (int col = 0; col < FEATURE_SIZE; ++col) {
            float val = x[row * FEATURE_SIZE + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        float sum_exp = 0.0f;
        for (int col = 0; col < FEATURE_SIZE; ++col) {
            float val = x[row * FEATURE_SIZE + col];
            sum_exp += expf(val - max_val);
        }
        for (int col = 0; col < FEATURE_SIZE; ++col) {
            float val = x[row * FEATURE_SIZE + col];
            resultGolden[row * FEATURE_SIZE + col] = expf(val - max_val) / sum_exp;
        }
    }

    // GPU implementation
    float* xDevice;
    float* resultDevice;
    float* rowmaxDevice;
    float* rowsumDevice;
    cudaMalloc(&xDevice, BATCH_SIZE * FEATURE_SIZE * sizeof(float));
    cudaMalloc(&resultDevice, BATCH_SIZE * FEATURE_SIZE * sizeof(float));
    cudaMalloc(&rowmaxDevice, BATCH_SIZE * sizeof(float));
    cudaMalloc(&rowsumDevice, BATCH_SIZE * sizeof(float));
    cudaMemcpy(xDevice, x, BATCH_SIZE * FEATURE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // For profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    dim3 numThreads(THREADS_PER_BLOCK);
    dim3 numBlocks(BATCH_SIZE);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        compute_rowmax_rowsum_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, rowmaxDevice, rowsumDevice);
        softmax_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, rowmaxDevice, rowsumDevice, resultDevice);
    }

    cudaMemset(resultDevice, 0, BATCH_SIZE * FEATURE_SIZE * sizeof(float));
    // each block computes one row
    cudaEventRecord(start);
    compute_rowmax_rowsum_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, rowmaxDevice, rowsumDevice);
    softmax_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, rowmaxDevice, rowsumDevice, resultDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "softmax_fp32_kernel duration: " << elapsedTime * 1000 << " us" << std::endl;

    // copy back the result
    float* result = new float[BATCH_SIZE * FEATURE_SIZE];
    cudaMemcpy(result, resultDevice, BATCH_SIZE * FEATURE_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    float error = ComputeRMSE(resultGolden, result, BATCH_SIZE * FEATURE_SIZE);
    std::cout << "softmax_fp32_kernel error: " << error << std::endl;
    float reached_memory_bandwidth = (BATCH_SIZE * FEATURE_SIZE * sizeof(float) * 2) / ((float)1024*1024*1024*1024) / (elapsedTime / 1000.0f);
    std::cout << "softmax_fp32_kernel reached memory bandwidth: " << reached_memory_bandwidth << " TB/s" << std::endl;

    // Free memory
    cudaFree(xDevice);
    cudaFree(resultDevice);
    cudaFree(rowmaxDevice);
    cudaFree(rowsumDevice);
    delete[] result;
    delete[] x;
    delete[] resultGolden;

    return 0;
}
