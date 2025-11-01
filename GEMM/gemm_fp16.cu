// A custom implementation of Half-Precision General Matrix Multiplication (GEMM) in CUDA, accumulating results in single-precision floating point.
// Matrix A is row-major MxK fp16 matrix
// Matrix B is column-major KxN fp16 matrix
// The output Matrix C is row-major MxN fp32 matrix

// Compilation command: nvcc -o gemm_fp16 -gencode=arch=compute_120,code=sm_120  gemm_fp16.cu -O3 -Xcompiler -fopenmp -lcublas

#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <mma.h>
#include "omp.h"
#include <cublas_v2.h>

// FP16 data type
using fp16_t = half;
using namespace nvcuda;

// Matrix dimensions
#define M 4096
#define N 4096
#define K 4096

// Each thread block computes BLOCK_M X BLOCK_N output elements
#define BLOCK_M 128
#define BLOCK_N 64

// How many warps along M and N dimensions in a thread block
#define BLOCK_NUM_WARPS_M 4
#define BLOCK_NUM_WARPS_N 2
#define BLOCK_NUM_WARPS (BLOCK_NUM_WARPS_M * BLOCK_NUM_WARPS_N)

// Each warp computes WARP_M X WARP_N output elements
#define WARP_M (BLOCK_M / BLOCK_NUM_WARPS_M)
#define WARP_N (BLOCK_N / BLOCK_NUM_WARPS_N)
#define WARP_SIZE 32

// The finest granularity of computation
#define FRAGMENT_SIZE 16

// How many fragments along M and N dimensions in a warp
#define WARP_NUM_FRAGMENTS_M (WARP_M / FRAGMENT_SIZE)
#define WARP_NUM_FRAGMENTS_N (WARP_N / FRAGMENT_SIZE)

// Total number of threads in a block
#define BLOCK_NUM_THREADS (BLOCK_NUM_WARPS_M * BLOCK_NUM_WARPS_N * WARP_SIZE)

// Number of fragments along K dimension computed per iteration
#define K_FRAGMENTS_PER_ITER 2

// Shape of a tile of matrix A loaded each time per block
#define A_TILE_M BLOCK_M
#define A_TILE_K (FRAGMENT_SIZE * K_FRAGMENTS_PER_ITER)

// Shape of a tile of matrix B loaded each time per block (A_TILE_K should be equal to B_TILE_K)
#define B_TILE_N BLOCK_N
#define B_TILE_K (FRAGMENT_SIZE * K_FRAGMENTS_PER_ITER)

// Number of elements per vectorized load
#define NUM_ELEMENTS_PER_LOAD (sizeof(int4) / sizeof(fp16_t))

// Number of threads loading a row of A_TILE
#define A_TILE_NUM_THREADS_PER_ROW (A_TILE_K / NUM_ELEMENTS_PER_LOAD)
// Number of rows loaded by each warp in A_TILE (assuming BLOCK_NUM_WARPS_M divides A_TILE_M)
#define A_TILE_NUM_ROWS_PER_WARP (A_TILE_M / BLOCK_NUM_WARPS)
// Number of rows loaded by each warp for each iteration
#define A_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / A_TILE_NUM_THREADS_PER_ROW)
// Number of iterations required for each warp to load its portion of A_TILE
#define A_TILE_NUM_LOAD_ITERS (A_TILE_NUM_ROWS_PER_WARP / (A_TILE_NUM_ROWS_PER_WARP_PER_ITER))
// Number of threads loading a column of B_TILE
#define B_TILE_NUM_THREADS_PER_COL (B_TILE_K / NUM_ELEMENTS_PER_LOAD)
// Number of columns loaded by each warp in B_TILE (assuming BLOCK_NUM_WARPS_N divides B_TILE_N)
#define B_TILE_NUM_COLS_PER_WARP (B_TILE_N / BLOCK_NUM_WARPS)
// Number of columns loaded by each warp for each iteration
#define B_TILE_NUM_COLS_PER_WARP_PER_ITER (WARP_SIZE / B_TILE_NUM_THREADS_PER_COL)
// Number of iterations required for each warp to load its portion of A_TILE
#define B_TILE_NUM_LOAD_ITERS (B_TILE_NUM_COLS_PER_WARP / (B_TILE_NUM_COLS_PER_WARP_PER_ITER))

// Shared memory padding to avoid bank conflicts
#define SHARED_MEM_PADDING 8

// Leading dimension of shared memory buffers
#define SHARED_MEM_STRIDE (A_TILE_K + SHARED_MEM_PADDING)

// Kernel for FP16 GEMM with FP32 accumulation
__global__ void gemm_fp16_kernel(const fp16_t* __restrict__ A, const fp16_t* __restrict__ B, float* __restrict__ C) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int warpID = tx / WARP_SIZE;
    int laneID = tx % WARP_SIZE;
    int globalOutputRow = bx * BLOCK_M;
    int globalOutputCol = by * BLOCK_N;

    // Dynamic shared memory for tiles of A and B
    extern __shared__ fp16_t shared_mem[][SHARED_MEM_STRIDE];
    fp16_t* As = &shared_mem[0][0];
    fp16_t* Bs = &shared_mem[A_TILE_M][0];

    // WMMA fragments for accumulation
    wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float> cFrag[WARP_NUM_FRAGMENTS_M][WARP_NUM_FRAGMENTS_N];

    // Initialize output fragments to zero
#pragma unroll
    for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
        }
    }

    // Each iteration loads BLOCK_M x A_TILE_K tile of A and BLOCK_N * B_TILE_K tile of B into shared memory, and
    // computes partial GEMM results and accumulates into output fragments
#pragma unroll
    for (int kTile = 0; kTile < K; kTile += A_TILE_K) {
        // Load A_TILE into shared memory
#pragma unroll
        for (int iter = 0; iter < A_TILE_NUM_LOAD_ITERS; ++iter) {
            int row = warpID * A_TILE_NUM_ROWS_PER_WARP + (laneID / A_TILE_NUM_THREADS_PER_ROW) + iter * A_TILE_NUM_ROWS_PER_WARP_PER_ITER;
            int col = (laneID % A_TILE_NUM_THREADS_PER_ROW) * NUM_ELEMENTS_PER_LOAD;
            int globalRow = globalOutputRow + row;
            int globalCol = kTile + col;
            *(int4*)&As[row * SHARED_MEM_STRIDE + col] = *(int4*)&A[globalRow * K + globalCol];
        }
        // Load B_TILE into shared memory
#pragma unroll
        for (int iter = 0; iter < B_TILE_NUM_LOAD_ITERS; ++iter) {
            int row = warpID * B_TILE_NUM_COLS_PER_WARP + (laneID / B_TILE_NUM_THREADS_PER_COL) + iter * B_TILE_NUM_COLS_PER_WARP_PER_ITER;
            int col = (laneID % B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
            int globalRow = globalOutputCol + row;
            int globalCol = kTile + col;
            *(int4*)&Bs[row * SHARED_MEM_STRIDE + col] = *(int4*)&B[globalRow * K + globalCol];
        }
        __syncthreads();
        // Compute partial GEMM results
#pragma unroll
        for (int kFrag = 0; kFrag < K_FRAGMENTS_PER_ITER; ++kFrag) {
            wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, fp16_t, wmma::row_major> aFrag[WARP_NUM_FRAGMENTS_M];
            wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, fp16_t, wmma::col_major> bFrag[WARP_NUM_FRAGMENTS_N];
            // Load A fragments from shared memory to WMMA fragments
#pragma unroll
            for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
                int row = warpID / BLOCK_NUM_WARPS_N * WARP_M + i * FRAGMENT_SIZE;
                int col = kFrag * FRAGMENT_SIZE;
                wmma::load_matrix_sync(aFrag[i], &As[row * SHARED_MEM_STRIDE + col], SHARED_MEM_STRIDE);
            }
#pragma unroll
            for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
                int row = warpID % BLOCK_NUM_WARPS_N * WARP_N + j * FRAGMENT_SIZE;
                int col = kFrag * FRAGMENT_SIZE;
                wmma::load_matrix_sync(bFrag[j], &Bs[row * SHARED_MEM_STRIDE + col], SHARED_MEM_STRIDE);
            }

            // Perform matrix multiplication and accumulate results
#pragma unroll
            for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
                    wmma::mma_sync(cFrag[i][j], aFrag[i], bFrag[j], cFrag[i][j]);
                }
            }
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Store the output fragments to global memory
#pragma unroll
    for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
            int row = (warpID / BLOCK_NUM_WARPS_N) * WARP_M + i * FRAGMENT_SIZE;
            int col = (warpID % BLOCK_NUM_WARPS_N) * WARP_N + j * FRAGMENT_SIZE;
            int globalRow = globalOutputRow + row;
            int globalCol = globalOutputCol + col;
            wmma::store_matrix_sync(&C[globalRow * N + globalCol], cFrag[i][j], N, wmma::mem_row_major);
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
    assert(A_TILE_K == B_TILE_K);
    assert(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % A_TILE_K == 0);

    // generate random seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    // Allocate and initialize host matrices
    fp16_t *AHost = new fp16_t[M * K];
    fp16_t *BHost = new fp16_t[K * N];
    float *CTrue = new float[M * N];
    for (int i = 0; i < M * K; i++) {
        AHost[i] = static_cast<fp16_t>(distribution(generator));
    }
    for (int i = 0; i < K * N; i++) {
        BHost[i] = static_cast<fp16_t>(distribution(generator));
    }

    // CPU computation for verification
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(AHost[m * K + k]) * __half2float(BHost[n * K + k]);
            }
            CTrue[m * N + n] = sum;
        }
    }

    // Allocate device matrices
    fp16_t *ADevice;
    fp16_t *BDevice;
    float *CDevice;
    cudaMalloc(&ADevice, M * K * sizeof(fp16_t));
    cudaMalloc(&BDevice, N * K * sizeof(fp16_t));
    cudaMalloc(&CDevice, M * N * sizeof(float));
    cudaMemcpy(ADevice, AHost, M * K * sizeof(fp16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(BDevice, BHost, N * K * sizeof(fp16_t), cudaMemcpyHostToDevice);
    float *CHost = new float[M * N];

    // For profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cublas warm up
    for (int i = 0; i < 1000; ++i) {
        cublasGemmEx(
            handle,             // cuBLAS handle
            CUBLAS_OP_T,        // transa: Transpose A (for row-major A)
            CUBLAS_OP_N,        // transb: No transpose B (for column-major B)
            N,                  // m: rows of op(A) and C (M)
            M,                  // n: columns of op(B) and C (N)
            K,                  // k: inner dimension
            &alpha,             // alpha (pointer to float)
            BDevice,                // A matrix (fp16)
            CUDA_R_16F,         // A type (fp16)
            K,                // lda: Leading dimension of A (M)
            ADevice,                // B matrix (fp16)
            CUDA_R_16F,         // B type (fp16)
            K,                // ldb: Leading dimension of B (K)
            &beta,              // beta (pointer to float)
            CDevice,                // C matrix (fp32 output/input)
            CUDA_R_32F,         // C type (fp32)
            N,                // ldc: Leading dimension of C (M)
            CUBLAS_COMPUTE_32F_FAST_16F, // Compute type (fp16 multiplication, fp32 accumulation with Tensor Cores)
            CUBLAS_GEMM_DEFAULT_TENSOR_OP // Gemm algorithm selection
        );
    }
    cudaEventRecord(start);
    cublasGemmEx(
        handle,             // cuBLAS handle
        CUBLAS_OP_T,        // transa: Transpose A (for row-major A)
        CUBLAS_OP_N,        // transb: No transpose B (for column-major B)
        N,                  // m: rows of op(A) and C (M)
        M,                  // n: columns of op(B) and C (N)
        K,                  // k: inner dimension
        &alpha,             // alpha (pointer to float)
        BDevice,                // A matrix (fp16)
        CUDA_R_16F,         // A type (fp16)
        K,                // lda: Leading dimension of A (M)
        ADevice,                // B matrix (fp16)
        CUDA_R_16F,         // B type (fp16)
        K,                // ldb: Leading dimension of B (K)
        &beta,              // beta (pointer to float)
        CDevice,                // C matrix (fp32 output/input)
        CUDA_R_32F,         // C type (fp32)
        N,                // ldc: Leading dimension of C (M)
        CUBLAS_COMPUTE_32F_FAST_16F, // Compute type (fp16 multiplication, fp32 accumulation with Tensor Cores)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP // Gemm algorithm selection
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "cuBLAS gemmEx elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    float peakTFLOPS = static_cast<float>(2.0 * M) * static_cast<float>(N) * static_cast<float>(K) / 1e12 / (elapsedTime / 1e3);
    std::cout << "cuBLAS gemmEx peak TFLOPS: " << peakTFLOPS << " TFLOPS" << std::endl;

    cudaMemcpy(CHost, CDevice, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float error = ComputeRMSE(CTrue, CHost, M * N);
    std::cout << "cuBLAS gemmEx error: " << error << std::endl;

    dim3 numThreads(BLOCK_NUM_THREADS);
    dim3 numBlocks(M / BLOCK_M, N / BLOCK_N);

    size_t sharedMemSize = (A_TILE_M + B_TILE_N) * SHARED_MEM_STRIDE * sizeof(fp16_t);
    // Check if shared memory size exceeds the limit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        // Increase the limit to the required size
        cudaFuncSetAttribute(gemm_fp16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    }

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        gemm_fp16_kernel<<<numBlocks, numThreads, sharedMemSize>>>(ADevice, BDevice, CDevice);
    }


    cudaEventRecord(start);
    // Launch kernel
    gemm_fp16_kernel<<<numBlocks, numThreads, sharedMemSize>>>(ADevice, BDevice, CDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "gemm_fp16_kernel elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    peakTFLOPS = static_cast<float>(2.0 * M) * static_cast<float>(N) * static_cast<float>(K) / 1e12 / (elapsedTime / 1e3);
    std::cout << "gemm_fp16_kernel peak TFLOPS: " << peakTFLOPS << " TFLOPS" << std::endl;

    // Copy result back to host
    cudaMemcpy(CHost, CDevice, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // Verify results
    error = ComputeRMSE(CTrue, CHost, M * N);
    std::cout << "gemm_fp16_kernel error: " << error << std::endl;

    // Clean up
    delete[] AHost;
    delete[] BHost;
    delete[] CTrue;
    delete[] CHost;
    cudaFree(ADevice);
    cudaFree(BDevice);
    cudaFree(CDevice);

    return 0;
}
