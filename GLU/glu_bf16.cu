// A custom implementation of GLU (Gated Linear Unit), accumulating results in single-precision floating point.
// Matrix A is row-major MxK bf16 matrix
// Matrix B is column-major KxN bf16 matrix
// The output Matrix C is row-major MxN fp32 matrix

// Compilation command: nvcc -o glu_bf16 -gencode=arch=compute_120,code=sm_120  glu_bf16.cu  -Xcompiler -fopenmp -O3 --use_fast_math

#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <mma.h>
#include "omp.h"

// FP16 data type
using bf16_t = __nv_bfloat16;
using namespace nvcuda;

__host__ __device__ float sigmoid(float x) {
    return (1.0f / (1 + expf(-x)));
}
__host__ __device__ float gaussianError(float x) {
    return 0.5 * (1.0f + erf(x * rsqrtf(2)));
}
__host__ __device__ float swish(float x) {
    return (x / (1 + expf(-x)));
}

// Matrix dimensions (Llama 3.1 70B architecture, tensor parallelism 8)
#define BATCH_SIZE 128
#define INTERMEDIATE_SIZE 3584
#define K 8192

// GLU type (0: GLU, 1: SwiGLU, 2: GeGLU)
#define GLU_TYPE 0

// Activation Type
#if GLU_TYPE == 0
    #define ACTIVATION_TYPE sigmoid
#elif GLU_TYPE == 1
    #define ACTIVATION_TYPE swish
#elif GLU_TYPE == 2
    #define ACTIVATION_TYPE gaussianError
#endif

// Each thread block computes BLOCK_M X BLOCK_N output elements
// Each thread block computes BLOCK_M X (2 * BLOCK_N) of the GEMM, then using activation to reduce this to a BLOCK_M X BLOCK_N result
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
#define K_FRAGMENTS_PER_ITER 8

// Shape of a tile of matrix A loaded each time per block
#define A_TILE_M BLOCK_M
#define A_TILE_K (FRAGMENT_SIZE * K_FRAGMENTS_PER_ITER)

// Shape of a tile of matrix B loaded each time per block (A_TILE_K should be equal to B_TILE_K)
#define B_TILE_N BLOCK_N
#define B_TILE_K (FRAGMENT_SIZE * K_FRAGMENTS_PER_ITER)

// Number of elements per vectorized load
#define NUM_ELEMENTS_PER_LOAD (sizeof(int4) / sizeof(bf16_t))

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

// Leading dimension of shared memory buffers for the gemm result
#define SHARED_MEM_C_STRIDE (BLOCK_N * 2 + SHARED_MEM_PADDING)

// During writing from shared memory to global memory, number of elements per vectorized writing
#define S2G_NUM_ELEMENTS_PER_LOAD (sizeof(float4) / sizeof(float))
// Number of threads loading a row of C_TILE
#define C_TILE_NUM_THREADS_PER_ROW (BLOCK_N / S2G_NUM_ELEMENTS_PER_LOAD)
// Number of rows written by each warp in C_TILE
#define C_TILE_NUM_ROWS_PER_WARP (BLOCK_M / BLOCK_NUM_WARPS)
// Number of rows written by each warp for each iteration
#define C_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / C_TILE_NUM_THREADS_PER_ROW)
// Number of iterations required for each warp to written its portion of C_TILE
#define C_TILE_NUM_WRITE_ITERS (C_TILE_NUM_ROWS_PER_WARP / (C_TILE_NUM_ROWS_PER_WARP_PER_ITER))

// Kernel for FP16 GEMM with FP32 accumulation
__global__ void glu_bf16_kernel(const bf16_t* __restrict__ x, const bf16_t* __restrict__ weight, float* __restrict__ result) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int warpID = tx / WARP_SIZE;
    int laneID = tx % WARP_SIZE;
    int globalOutputRow = bx * BLOCK_M;
    int globalOutputCol = by * BLOCK_N;

    // Dynamic shared memory for tiles of A and B
    extern __shared__ bf16_t shared_mem[][SHARED_MEM_STRIDE];
    bf16_t* xs = &shared_mem[0][0];
    bf16_t* weight1s = &shared_mem[A_TILE_M][0];
    bf16_t* weight2s = &shared_mem[A_TILE_M + B_TILE_N][0];

    // WMMA fragments for accumulation
    wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float> cFrag[2][WARP_NUM_FRAGMENTS_M][WARP_NUM_FRAGMENTS_N];

    // Initialize output fragments to zero
#pragma unroll
    for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
            wmma::fill_fragment(cFrag[0][i][j], 0.0f);
            wmma::fill_fragment(cFrag[1][i][j], 0.0f);
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
            *(int4*)&xs[row * SHARED_MEM_STRIDE + col] = *(int4*)&x[globalRow * K + globalCol];
        }

#pragma unroll
        for (int iter = 0; iter < B_TILE_NUM_LOAD_ITERS; ++iter) {
            int row = warpID * B_TILE_NUM_COLS_PER_WARP + (laneID / B_TILE_NUM_THREADS_PER_COL) + iter * B_TILE_NUM_COLS_PER_WARP_PER_ITER;
            int col = (laneID % B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
            int globalRow = globalOutputCol + row;
            int globalCol = kTile + col;
            *(int4*)&weight1s[row * SHARED_MEM_STRIDE + col] = *(int4*)&weight[globalRow * K + globalCol];
            *(int4*)&weight2s[row * SHARED_MEM_STRIDE + col] = *(int4*)&weight[(globalRow + globalOutputCol) * K + globalCol];
        }
        __syncthreads();

        // Compute partial GEMM results
#pragma unroll
        for (int kFrag = 0; kFrag < K_FRAGMENTS_PER_ITER; ++kFrag) {
            wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major> aFrag[WARP_NUM_FRAGMENTS_M];
            wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::col_major> bFrag[2][WARP_NUM_FRAGMENTS_N];
            // Load A fragments from shared memory to WMMA fragments
#pragma unroll
            for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
                int row = warpID / BLOCK_NUM_WARPS_N * WARP_M + i * FRAGMENT_SIZE;
                int col = kFrag * FRAGMENT_SIZE;
                wmma::load_matrix_sync(aFrag[i], &xs[row * SHARED_MEM_STRIDE + col], SHARED_MEM_STRIDE);
            }
#pragma unroll
            for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
                int row = warpID % BLOCK_NUM_WARPS_N * WARP_N + j * FRAGMENT_SIZE;
                int col = kFrag * FRAGMENT_SIZE;
                wmma::load_matrix_sync(bFrag[0][j], &weight1s[row * SHARED_MEM_STRIDE + col], SHARED_MEM_STRIDE);
                wmma::load_matrix_sync(bFrag[1][j], &weight2s[row * SHARED_MEM_STRIDE + col], SHARED_MEM_STRIDE);
            }

            // Perform matrix multiplication and accumulate results
#pragma unroll
            for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
                    wmma::mma_sync(cFrag[0][i][j], aFrag[i], bFrag[0][j], cFrag[0][i][j]);
                    wmma::mma_sync(cFrag[1][i][j], aFrag[i], bFrag[1][j], cFrag[1][i][j]);
                }
            }
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Store the output framents back to shared memory
    float* gemmResult = (float*)&shared_mem[0][0];
#pragma unroll
    for (int i = 0; i < WARP_NUM_FRAGMENTS_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_NUM_FRAGMENTS_N; ++j) {
            int row = (warpID / BLOCK_NUM_WARPS_N) * WARP_M + i * FRAGMENT_SIZE;
            int col = (warpID % BLOCK_NUM_WARPS_N) * WARP_N + j * FRAGMENT_SIZE;
            wmma::store_matrix_sync(&gemmResult[row * (SHARED_MEM_C_STRIDE) + col], cFrag[0][i][j], SHARED_MEM_C_STRIDE, wmma::mem_row_major);
            wmma::store_matrix_sync(&gemmResult[row * (SHARED_MEM_C_STRIDE) + (col + BLOCK_N)], cFrag[1][i][j], SHARED_MEM_C_STRIDE, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Write final result back to global memory
    for (int i = 0; i < C_TILE_NUM_WRITE_ITERS; ++i) {
        int row = warpID * C_TILE_NUM_ROWS_PER_WARP + (laneID / C_TILE_NUM_THREADS_PER_ROW) + i * C_TILE_NUM_ROWS_PER_WARP_PER_ITER;
        int col = (laneID % C_TILE_NUM_THREADS_PER_ROW) * S2G_NUM_ELEMENTS_PER_LOAD;
        float4 result1 = *(float4*)&gemmResult[row * SHARED_MEM_C_STRIDE + col];
        float4 result2 = *(float4*)&gemmResult[row * SHARED_MEM_C_STRIDE + col + BLOCK_N];
        float4 finalResult = {
            result1.x * ACTIVATION_TYPE(result2.x),
            result1.y * ACTIVATION_TYPE(result2.y),
            result1.z * ACTIVATION_TYPE(result2.z),
            result1.w * ACTIVATION_TYPE(result2.w)
        };
        *(float4*)&result[(globalOutputRow + row) * INTERMEDIATE_SIZE + (globalOutputCol + col)] = finalResult;
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
    assert(BATCH_SIZE % BLOCK_M == 0 && INTERMEDIATE_SIZE % BLOCK_N == 0 && K % A_TILE_K == 0);

    // generate random seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    // Allocate and initialize host matrices
    bf16_t *xHost = new bf16_t[BATCH_SIZE * K];
    bf16_t *weightHost = new bf16_t[K * INTERMEDIATE_SIZE * 2];
    float *resultTrue = new float[BATCH_SIZE * INTERMEDIATE_SIZE];
    for (int i = 0; i < BATCH_SIZE * K; i++) {
        xHost[i] = static_cast<bf16_t>(distribution(generator));
    }
    for (int i = 0; i < K * INTERMEDIATE_SIZE * 2; i++) {
        weightHost[i] = static_cast<bf16_t>(distribution(generator));
    }

    // CPU computation for verification
#pragma omp parallel for
    for (int m = 0; m < BATCH_SIZE; m++) {
        for (int n = 0; n < INTERMEDIATE_SIZE; n++) {
            float sum = 0.0f;
            float sum2 = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __bfloat162float(xHost[m * K + k]) * __bfloat162float(weightHost[n * K + k]);
            }
            for (int k = 0; k < K; k++) {
                sum2 += __bfloat162float(xHost[m * K + k]) * __bfloat162float(weightHost[(n + INTERMEDIATE_SIZE) * K + k]);
            }
            resultTrue[m * INTERMEDIATE_SIZE + n] = sum * ACTIVATION_TYPE(sum2);
        }
    }

    // Allocate device matrices
    bf16_t *xDevice;
    bf16_t *weightDevice;
    float *resultDevice;
    cudaMalloc(&xDevice, BATCH_SIZE * K * sizeof(bf16_t));
    cudaMalloc(&weightDevice, INTERMEDIATE_SIZE * 2 * K * sizeof(bf16_t));
    cudaMalloc(&resultDevice, BATCH_SIZE * INTERMEDIATE_SIZE * sizeof(float));
    cudaMemcpy(xDevice, xHost, BATCH_SIZE * K * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(weightDevice, weightHost, INTERMEDIATE_SIZE * 2 * K * sizeof(bf16_t), cudaMemcpyHostToDevice);
    float *resultHost = new float[BATCH_SIZE * INTERMEDIATE_SIZE];

    // For profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 numThreads(BLOCK_NUM_THREADS);
    dim3 numBlocks(BATCH_SIZE / BLOCK_M, INTERMEDIATE_SIZE / BLOCK_N);

    size_t sharedMemSize = std::max((A_TILE_M + B_TILE_N + B_TILE_N) * SHARED_MEM_STRIDE * sizeof(bf16_t), \
                            BLOCK_M * SHARED_MEM_C_STRIDE * sizeof(float));
    // Check if shared memory size exceeds the limit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        // Increase the limit to the required size
        cudaFuncSetAttribute(glu_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    }

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        glu_bf16_kernel<<<numBlocks, numThreads, sharedMemSize>>>(xDevice, weightDevice, resultDevice);
    }

    cudaEventRecord(start);
    // Launch kernel
    glu_bf16_kernel<<<numBlocks, numThreads, sharedMemSize>>>(xDevice, weightDevice, resultDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "glu_bf16_kernel elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    float peakTFLOPS = ((2.0f * BATCH_SIZE) * INTERMEDIATE_SIZE * 2 * K + BATCH_SIZE * INTERMEDIATE_SIZE * 2.0) / 1e12 / (elapsedTime / 1e3);
    std::cout << "glu_bf16_kernel peak TFLOPS: " << peakTFLOPS << " TFLOPS" << std::endl;

    // Copy result back to host
    cudaMemcpy(resultHost, resultDevice, BATCH_SIZE * INTERMEDIATE_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // Verify results
    float error = ComputeRMSE(resultTrue, resultHost, BATCH_SIZE * INTERMEDIATE_SIZE);
    std::cout << "glu_bf16_kernel error: " << error << std::endl;

    // Clean up
    delete[] xHost;
    delete[] weightHost;
    delete[] resultTrue;
    delete[] resultHost;
    cudaFree(xDevice);
    cudaFree(weightDevice);
    cudaFree(resultDevice);

    return 0;
}
