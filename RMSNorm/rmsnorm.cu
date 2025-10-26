// CUDA implementation of RMSNorm
// Compilation command: nvcc -o rmsnorm -gencode=arch=compute_120,code=sm_120 -O3 rmsnorm.cu 
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <random>
#include <chrono>

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

#define SEQLEN 4096
#define HIDDEN_SIZE 7168
#define eps 1e-6f

#define BLOCK_THREAD_NUM 256
#define WORK_PER_THREAD CEIL_DIV(HIDDEN_SIZE, BLOCK_THREAD_NUM)
#define WARP_SIZE 32
#define BLOCK_NUM_WARPS (BLOCK_THREAD_NUM / WARP_SIZE)

// each thread block handles one row
__global__ void rmsnorm_fp32_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ rmsnorm_out) {
    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    int tx = threadIdx.x;
    int laneID = tx % 32;
    __shared__ float warpSum[BLOCK_NUM_WARPS];

    int rowIndex = blockIdx.x;
    x += rowIndex * HIDDEN_SIZE;

    // sum of squares per thread
    float sum_sq = 0.0f;

#pragma unroll
    for (int i = tx * 4; i < HIDDEN_SIZE; i += blockDim.x * 4) {
        if (i + 3 < HIDDEN_SIZE) {
            float4 temp = *(float4*)&x[i];
            sum_sq += temp.x * temp.x + temp.y * temp.y + temp.z * temp.z + temp.w * temp.w;
        } else {
            for (int j = i; j < HIDDEN_SIZE; ++j) {
                float val = x[j];
                sum_sq += val * val;
            }
        }
    }

    // sum of squares per warp
    sum_sq = cooperative_groups::reduce(warp, sum_sq, cooperative_groups::plus<float>());
    if (laneID == 0) {
        warpSum[tx / WARP_SIZE] = sum_sq;
    }
    __syncthreads();
    // sum of squares per block
    for (int offset = 0; offset < BLOCK_NUM_WARPS; ++offset) {
        if (tx < offset) {
            warpSum[tx] += warpSum[offset];
        }
        __syncthreads();
    }

    float rms = warpSum[0];

    // we compute reverse sqrt here to save some computation in the next step
    rms = rsqrtf(rms / HIDDEN_SIZE + eps);

    // write output
#pragma unroll
    for (int i = tx * 4; i < HIDDEN_SIZE; i += blockDim.x * 4) {
        if (i + 3 < HIDDEN_SIZE) {
            float4 val = *(float4*)&x[i];
            float4 w = *(float4*)&weight[i];
            val.x = val.x * rms * w.x;
            val.y = val.y * rms * w.y;
            val.z = val.z * rms * w.z;
            val.w = val.w * rms * w.w;
            *(float4*)&rmsnorm_out[rowIndex * HIDDEN_SIZE + i] = val;
        } else {
            for (int j = i; j < HIDDEN_SIZE; ++j) {
                rmsnorm_out[rowIndex * HIDDEN_SIZE + j] = x[j] * rms * weight[j];
            }
        }
    }
}

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
    assert(HIDDEN_SIZE % 4 == 0); // for float4 load/store
    // decalre the uniform random number generator
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);


    float* x = new float[SEQLEN * HIDDEN_SIZE];
    float* weight = new float[HIDDEN_SIZE];
    float* rmsnormGolden = new float[SEQLEN * HIDDEN_SIZE];
    float* rmsnormOut = new float[SEQLEN * HIDDEN_SIZE];

    for (int i = 0; i < SEQLEN * HIDDEN_SIZE; ++i) {
        x[i] = distribution(generator);
    }
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        weight[i] = distribution(generator);
    }

    // CPU implementation
    for (int row = 0; row < SEQLEN; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < HIDDEN_SIZE; ++col) {
            float val = x[row * HIDDEN_SIZE + col];
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / HIDDEN_SIZE + eps);
        for (int col = 0; col < HIDDEN_SIZE; ++col) {
            rmsnormGolden[row * HIDDEN_SIZE + col] = x[row * HIDDEN_SIZE + col] * rms * weight[col];
        }
    }

    // GPU implementation
    float *xDevice, *weightDevice, *rmsnorm_outDevice;
    cudaMalloc((void**)&xDevice, SEQLEN * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&weightDevice, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&rmsnorm_outDevice, SEQLEN * HIDDEN_SIZE * sizeof(float));
    cudaMemcpy(xDevice, x, SEQLEN * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weightDevice, weight, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // For profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 numThreads(256);
    dim3 numBlocks(SEQLEN);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        rmsnorm_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, weightDevice, rmsnorm_outDevice);
    }

    cudaEventRecord(start);
    rmsnorm_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, weightDevice, rmsnorm_outDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "rmsnorm_fp32_kernel elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    float reached_mem_bw = (SEQLEN * HIDDEN_SIZE * sizeof(float) * 2) / ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000.0f);
    std::cout << "rmsnorm_fp32_kernel reached memory bandwidth: " << reached_mem_bw << " TB/s" << std::endl;

    cudaMemcpy(rmsnormOut, rmsnorm_outDevice, SEQLEN * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float error = ComputeRMSE(rmsnormGolden, rmsnormOut, SEQLEN * HIDDEN_SIZE);
    std::cout << "rmsnorm_fp32_kernel error: " << error << std::endl;

    // clean up
    cudaFree(xDevice);
    cudaFree(weightDevice);
    cudaFree(rmsnorm_outDevice);
    delete[] x;
    delete[] rmsnormGolden;
    delete[] rmsnormOut;
    return 0;
}
