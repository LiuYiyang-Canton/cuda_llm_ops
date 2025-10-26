// compilation: nvcc  -o elementwiseadd -gencode=arch=compute_120,code=sm_120  -lcublas elementwiseadd.cu
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

template <const int work_per_block>
__global__ void elementwiseadd_fp32_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result, int N) {
    int row = blockIdx.y;
    int col_start = blockIdx.x * work_per_block;
    int col_end = (blockIdx.x + 1) * work_per_block;
    int tx = threadIdx.x;
    for (int i = col_start + tx * 4; i < col_end; i += blockDim.x * 4) {
        if (i >= N) {
            return;
        }
        auto alocal = *(float4*)&a[row * N + i];
        auto blocal = *(float4*)&b[row * N + i];
        *(float4*)&result[row * N + i] = {
                              alocal.x + blocal.x,
                              alocal.y + blocal.y,
                              alocal.z + blocal.z,
                              alocal.w + blocal.w
        };
    }

}

float ComputeRMSE(const float* __restrict__ golden, const float* __restrict x, size_t N) {
    float error = 0;
    float norm = 0;
    for (int i = 0; i < N; ++i) {
        error += (golden[i] - x[i]) * (golden[i] - x[i]);
        norm += golden[i] * golden[i];
    }
    return std::sqrt(error) / std::sqrt(norm);
}

int main() {
    // generate random seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    int N = 4096;

    // reasonable assumption for general DNN
    assert(N % 4 == 0);

    float* a = new float[N * N];
    float* b = new float[N * N];
    float* resultGolden = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        a[i] = distribution(generator);
    }
    for (int i = 0; i < N * N; ++i) {
        b[i] = distribution(generator);
    }
    for (int i = 0; i < N * N; ++i) {
        resultGolden[i] = a[i] + b[i];
    }

    float* aDevice;
    float* bDevice;
    float* resultDevice;
    float* result = new float[N * N];

    size_t matrix_size = N * N * sizeof(float);

    // for profiling
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&aDevice, matrix_size);
    cudaMalloc(&bDevice, matrix_size);
    cudaMalloc(&resultDevice, matrix_size);

    cudaMemcpy(aDevice, a, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, b, matrix_size, cudaMemcpyHostToDevice);

    // --- cublas ---
    float alpha = 1.0;
    float beta = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // --- cublas warm up ---
    for (int i = 0; i < 1000; ++i) {
        cublasSgeam(
            handle,
            CUBLAS_OP_N, // transa: No transpose for A
            CUBLAS_OP_N, // transb: No transpose for B
            N,        // m: Number of rows in A and C
            N,        // n: Number of columns in B and C
            &alpha,      // alpha: Scaling factor for A (1.0)
            aDevice,         // A: Device pointer to matrix A
            N,        // lda: Leading dimension of A
            &beta,       // beta: Scaling factor for B (1.0)
            bDevice,         // B: Device pointer to matrix B
            N,        // ldb: Leading dimension of B
            resultDevice,         // C: Device pointer to result matrix C
            N         // ldc: Leading dimension of C (rows)
        );
    }

    cudaEventRecord(start);
    cublasSgeam(
        handle,
        CUBLAS_OP_N, // transa: No transpose for A
        CUBLAS_OP_N, // transb: No transpose for B
        N,        // m: Number of rows in A and C
        N,        // n: Number of columns in B and C
        &alpha,      // alpha: Scaling factor for A (1.0)
        aDevice,         // A: Device pointer to matrix A
        N,        // lda: Leading dimension of A
        &beta,       // beta: Scaling factor for B (1.0)
        bDevice,         // B: Device pointer to matrix B
        N,        // ldb: Leading dimension of B
        resultDevice,         // C: Device pointer to result matrix C
        N         // ldc: Leading dimension of C (rows)
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "cublasSgeam duration: " << elapsedTime * 1000 << " us" << std::endl;

    cudaMemcpy(result, resultDevice, matrix_size, cudaMemcpyDeviceToHost);
    float error = ComputeRMSE(resultGolden, result, N * N);
    std::cout << "cublasSgeam error = " << error << std::endl;
    float reached_mem_bw = matrix_size * 3.0 / ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "cublasSgeam reached_mem_bw: " << reached_mem_bw << " TB/s" << std::endl;

    // each blockIdx.y computes result for one row
    // each blockIdx.x computes result for a continuous tile of one row

    constexpr int numThreads = 256;
    constexpr int work_per_thread = 4;
    constexpr int work_per_block = numThreads * work_per_thread;
    dim3 numBlocks(CEIL_DIV(N, work_per_block), N);

    // warm up
    for (int i = 0; i < 1000; ++i) {
        elementwiseadd_fp32_kernel<work_per_block><<<numBlocks, numThreads>>>(aDevice, bDevice, resultDevice, N);
    }
    cudaMemset(resultDevice, 0, matrix_size);

    cudaEventRecord(start);
    elementwiseadd_fp32_kernel<work_per_block><<<numBlocks, numThreads>>>(aDevice, bDevice, resultDevice, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "elementwise_add_fp32_kernel duration: " << elapsedTime * 1000 << " us" << std::endl;
    reached_mem_bw = matrix_size * 3.0 / ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "elementwise_add_fp32_kernel reached_mem_bw: " << reached_mem_bw << " TB/s" << std::endl;

    cudaMemcpy(result, resultDevice, matrix_size, cudaMemcpyDeviceToHost);

    error = ComputeRMSE(resultGolden, result, N * N);
    std::cout << "elementwiseadd_fp32_kernel error = " << error << std::endl;

    cudaFree(aDevice);
    cudaFree(bDevice);
    cudaFree(resultDevice);

    delete[] a;
    delete[] b;
    delete[] resultGolden;
    delete[] result;
}