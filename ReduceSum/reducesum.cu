// compilation: nvcc  -o reducesum -gencode=arch=compute_120,code=sm_120  -lcublas reducesum.cu  -O3
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cassert>
#include <cublas_v2.h>

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

#define M 4096
#define N 4096

// each thread block handles four rows
#define BLOCK_THREAD_NUM 256
#define WORK_PER_THREAD CEIL_DIV(N, BLOCK_THREAD_NUM)
#define WORK_PER_BLOCK (BLOCK_THREAD_NUM * WORK_PER_THREAD)

#define WARP_SIZE 32
#define BLOCK_NUM_WARPS (BLOCK_THREAD_NUM / WARP_SIZE)

__global__ void reducesum_fp32_kernel(const float* __restrict__ x, float* __restrict__ result) {
    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    int tx = threadIdx.x;
    int warpID = tx / WARP_SIZE;
    int laneID = tx % WARP_SIZE;
    __shared__ float4 warpSum[BLOCK_NUM_WARPS];

    int rowIndex = blockIdx.x * 4;

    x += rowIndex * N;

    // sum per thread
    float4 sum = {0, 0, 0, 0};
#pragma unroll
    for (int i = tx * 4; i < N; i += blockDim.x * 4) {
        if (i < N) {
            float4 temp = *(float4*)&x[i];
            float4 temp2 = *(float4*)&x[N + i];
            float4 temp3 = *(float4*)&x[2 * N + i];
            float4 temp4 = *(float4*)&x[3 * N + i];
            sum.x += temp.x + temp.y + temp.z + temp.w;
            sum.y += temp2.x + temp2.y + temp2.z + temp2.w;
            sum.z += temp3.x + temp3.y + temp3.z + temp3.w;
            sum.w += temp4.x + temp4.y + temp4.z + temp4.w;
        }
    }
    
    // sum per warp
    sum.x = cooperative_groups::reduce(warp, sum.x, cooperative_groups::plus<float>());
    sum.y = cooperative_groups::reduce(warp, sum.y, cooperative_groups::plus<float>());
    sum.z = cooperative_groups::reduce(warp, sum.z, cooperative_groups::plus<float>());
    sum.w = cooperative_groups::reduce(warp, sum.w, cooperative_groups::plus<float>());
    if (laneID == 0) {
        warpSum[warpID].x = sum.x;
        warpSum[warpID].y = sum.y;
        warpSum[warpID].z = sum.z;
        warpSum[warpID].w = sum.w;
    }
    __syncthreads();

    // sum per block
#pragma unroll
    for (int offset = BLOCK_NUM_WARPS / 2; offset != 0; offset >>= 1) {
        if (tx < offset) {
            warpSum[tx].x += warpSum[tx + offset].x;
            warpSum[tx].y += warpSum[tx + offset].y;
            warpSum[tx].z += warpSum[tx + offset].z;
            warpSum[tx].w += warpSum[tx + offset].w;
        }
        __syncthreads();
    }

    if (tx == 0) {
        *(float4*)&result[rowIndex] = warpSum[0];
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
    assert(M % 4 == 0);
    assert(N % 4 == 0);
    // generate random seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    float* x = new float[M * N];
    float* resultGolden = new float[M];

    for (int i = 0; i < M * N; ++i) {
        x[i] = distribution(generator);
    }

    std::fill_n(resultGolden, M, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            resultGolden[i] += x[i * N + j];
        }
    }

    // for profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float* xDevice;
    float* resultDevice;
    float* result = new float[M];
    cudaMalloc(&xDevice, M * N * sizeof(float));
    cudaMalloc(&resultDevice, M * sizeof(float));
    cudaMemcpy(xDevice, x, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // --- cublas ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;
    float* weight = new float[N];
    std::fill_n(weight, N, 1);
    float* weightDevice;
    cudaMalloc(&weightDevice, N * sizeof(float));
    cudaMemcpy(weightDevice, weight, N * sizeof(float), cudaMemcpyHostToDevice);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        cublasSgemv(
            handle,
            CUBLAS_OP_T,  // Important: Use Transpose to handle Row-Major layout
            N,            // m (rows of op(A)): M in standard row sum A * x formula
            M,            // n (cols of op(A)): N in standard row sum A * x formula
            &alpha,
            xDevice,
            N,          // Leading dimension of A
            weightDevice,
            1,            // incx
            &beta,
            resultDevice,
            1             // incy
        );
    }

    cudaEventRecord(start);
    cublasSgemv(
        handle,
        CUBLAS_OP_T,  // Important: Use Transpose to handle Row-Major layout
        N,            // m (rows of op(A)): M in standard row sum A * x formula
        M,            // n (cols of op(A)): N in standard row sum A * x formula
        &alpha,
        xDevice,
        N,          // Leading dimension of A
        weightDevice,
        1,            // incx
        &beta,
        resultDevice,
        1             // incy
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "cublasSgemv duration: " << elapsedTime * 1000 << " us" << std::endl;
    float reachedMemBw = ((float)M * N * sizeof(float) + M * sizeof(float)) / ((float) 1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "cublasSgemv reachedMemBw: " << reachedMemBw << " TB/s" << std::endl;
    cudaMemcpy(result, resultDevice, M * sizeof(float), cudaMemcpyDeviceToHost);
    float error = ComputeRMSE(resultGolden, result, M);
    std::cout << "cublasSgemv error = " << error << std::endl;

    // my implementation
    dim3 numThreads(BLOCK_THREAD_NUM);
    dim3 numBlocks(M / 4);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        reducesum_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, resultDevice);
    }

    cudaEventRecord(start);
    reducesum_fp32_kernel<<<numBlocks, numThreads>>>(xDevice, resultDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "reducesum_fp32_kernel duration: " << elapsedTime * 1000 << " us" << std::endl;
    reachedMemBw = ((float)M * N * sizeof(float) + M * sizeof(float)) / ((float) 1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "reducesum_fp32_kernel reachedMemBw: " << reachedMemBw << " TB/s" << std::endl;

    cudaMemcpy(result, resultDevice, M * sizeof(float), cudaMemcpyDeviceToHost);

    error = ComputeRMSE(resultGolden, result, M);
    std::cout << "reducesum_fp32_kernel error = " << error << std::endl;

    delete[] weight;
    delete[] x;
    delete[] resultGolden;

    cudaFree(result);
    cudaFree(xDevice);
    cudaFree(weightDevice);
}