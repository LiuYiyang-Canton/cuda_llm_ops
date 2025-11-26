// Rotary position embedding: bf16 I/O, shapes [batch, seqlen, hidden_dim].
// Reports kernel runtime, bandwidth, and RMSE against the CPU baseline.
// compilation: nvcc -o rope.o -gencode=arch=compute_120,code=sm_120 rope.cu -O3

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

/**
 * Returns ceil(numerator / denominator) for positive integers.
 *
 * @param numerator dividend.
 * @param denominator divisor (must be > 0).
 * @return rounded-up quotient.
 */
constexpr int CeilDiv(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/**
 * Validates CUDA API results and aborts if an error occurs.
 *
 * @param result cudaError_t status code returned by CUDA runtime.
 * @param message human-readable context for diagnostics.
 */
void CheckCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * Computes the RMSE between golden and computed bf16 buffers.
 *
 * @param golden pointer to reference data on host.
 * @param values pointer to computed data on host.
 * @param count number of bf16 elements to compare.
 * @return RMSE metric in fp32.
 */
float ComputeBf16Rmse(const bf16* __restrict__ golden, const bf16* __restrict__ values, size_t count) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double ref = static_cast<double>(__bfloat162float(golden[i]));
        const double got = static_cast<double>(__bfloat162float(values[i]));
        const double diff = ref - got;
        error += diff * diff;
        norm += ref * ref;
    }
    if (norm == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::sqrt(error) / std::sqrt(norm));
}

/**
 * Reference CPU rotary position embedding.
 *
 * Input/output shapes: (batch_size, seqlen, hidden_dim). hidden_dim must be even.
 * Values are stored in bf16; accumulation uses fp32 for stability.
 *
 * @param input bf16 pointer to input tensor in host memory.
 * @param output bf16 pointer to output tensor in host memory.
 * @param batch_size batch dimension.
 * @param seqlen sequence length.
 * @param hidden_dim hidden size (must be even).
 * @param base RoPE base frequency (defaults to 10000).
 */
void ComputeCpuRope(const bf16* __restrict__ input,
                    bf16* __restrict__ output,
                    int batch_size,
                    int seqlen,
                    int hidden_dim,
                    float base = 10000.0f) {
    assert(hidden_dim % 2 == 0);
    const int half_dim = hidden_dim / 2;

    std::vector<float> inv_freq(half_dim);
    const float hidden_inv = 1.0f / static_cast<float>(hidden_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq[i] = std::pow(base, -2.0f * static_cast<float>(i) * hidden_inv);
    }

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int token = 0; token < seqlen; ++token) {
            const size_t base_index = (static_cast<size_t>(batch) * seqlen + token) * hidden_dim;
            for (int i = 0; i < half_dim; ++i) {
                const float angle = static_cast<float>(token) * inv_freq[i];
                const float cos_value = std::cos(angle);
                const float sin_value = std::sin(angle);
                const size_t even_index = base_index + static_cast<size_t>(2 * i);
                const size_t odd_index = even_index + 1;
                const float even = __bfloat162float(input[even_index]);
                const float odd = __bfloat162float(input[odd_index]);
                output[even_index] = __float2bfloat16(even * cos_value - odd * sin_value);
                output[odd_index] = __float2bfloat16(even * sin_value + odd * cos_value);
            }
        }
    }
}

/**
 * CUDA kernel for rotary position embedding.
 *
 * blockIdx.z = batch dimension, blockIdx.y = block of sequence positions,
 * blockIdx.x covers tiles along the hidden dimension. Each thread handles
 * ValuesPerThread contiguous bf16 elements along the hidden dimension for
 * each token assigned to the block.
 */
/**
 * @tparam ValuesPerThread Number of contiguous bf16 elements each thread processes along the hidden dimension.
 * @tparam TokensPerBlock Number of tokens each block covers along the sequence dimension.
 * @param input Pointer to device input tensor of shape [batch, seqlen, hidden_dim].
 * @param output Pointer to device output tensor of shape [batch, seqlen, hidden_dim].
 * @param hidden_dim Hidden size (must be even).
 * @param seqlen Sequence length.
 * @param base RoPE base frequency.
 */
template <int ValuesPerThread, int TokensPerBlock>
__global__ void RopeBf16Kernel(const bf16* __restrict__ input,
                           bf16* __restrict__ output,
                           int hidden_dim,
                           int seqlen,
                           float base = 10000.0f) {
    
    const int batchIdx = blockIdx.z;
    const int blockSeqStart = blockIdx.y * TokensPerBlock;

    input += batchIdx * seqlen * hidden_dim + blockSeqStart * hidden_dim;
    output += batchIdx * seqlen * hidden_dim + blockSeqStart * hidden_dim;

    extern __shared__ float shared_inv_freq[];

    // Compute the inverse frequency
    const int tx = threadIdx.x;
    if (tx < hidden_dim) {
        shared_inv_freq[tx] = powf(base, -2.0f * (tx / 2) / (float)hidden_dim);
    }
    __syncthreads();

    // Get the token index this thread is processing
    const int localTokenIdx = tx / (hidden_dim / ValuesPerThread);
    const int valueStartIdx = (tx % (hidden_dim / ValuesPerThread)) * ValuesPerThread;
    const int globalTokenIdx = blockSeqStart + localTokenIdx;

    if (globalTokenIdx >= seqlen) {
        return; // Out of bounds
    }

    for (int dim = 0; dim < ValuesPerThread; dim += 8) {
        // Vectorized processing of 8 values using int4 view
        bf16 val[8];
        *(int4*)&val[0] = *(int4*)&input[localTokenIdx * hidden_dim + valueStartIdx + dim];
        // Local buffer for output
        bf16 out_val[8];

        const int hiddenIdx = valueStartIdx + dim;
        const int hiddenIdx1 = valueStartIdx + dim + 2;
        const int hiddenIdx2 = valueStartIdx + dim + 4;
        const int hiddenIdx3 = valueStartIdx + dim + 6;
        const float4 angle = {
                    globalTokenIdx * shared_inv_freq[hiddenIdx],
                    globalTokenIdx * shared_inv_freq[hiddenIdx1],
                    globalTokenIdx * shared_inv_freq[hiddenIdx2],
                    globalTokenIdx * shared_inv_freq[hiddenIdx3]
                };

        float4 cos_value;
        float4 sin_value;
        __sincosf(angle.x, &sin_value.x, &cos_value.x);
        __sincosf(angle.y, &sin_value.y, &cos_value.y);
        __sincosf(angle.z, &sin_value.z, &cos_value.z);
        __sincosf(angle.w, &sin_value.w, &cos_value.w);

        float2 pair0 = __bfloat1622float2(*(__nv_bfloat162*)&val[0]);
        float2 pair1 = __bfloat1622float2(*(__nv_bfloat162*)&val[2]);
        float2 pair2 = __bfloat1622float2(*(__nv_bfloat162*)&val[4]);
        float2 pair3 = __bfloat1622float2(*(__nv_bfloat162*)&val[6]);

        *(__nv_bfloat162*)&out_val[0] = __float22bfloat162_rn({pair0.x * cos_value.x - pair0.y * sin_value.x, pair0.x * sin_value.x + pair0.y * cos_value.x});
        *(__nv_bfloat162*)&out_val[2] = __float22bfloat162_rn({pair1.x * cos_value.y - pair1.y * sin_value.y, pair1.x * sin_value.y + pair1.y * cos_value.y});
        *(__nv_bfloat162*)&out_val[4] = __float22bfloat162_rn({pair2.x * cos_value.z - pair2.y * sin_value.z, pair2.x * sin_value.z + pair2.y * cos_value.z});
        *(__nv_bfloat162*)&out_val[6] = __float22bfloat162_rn({pair3.x * cos_value.w - pair3.y * sin_value.w, pair3.x * sin_value.w + pair3.y * cos_value.w});

        // Store the results back to global memory
        *(int4*)&output[localTokenIdx * hidden_dim + valueStartIdx + dim] = *(int4*)&out_val[0];
    }
}

}  // namespace

// Driver: generates data, runs CPU reference and CUDA kernel, reports timing/error.
int main() {
    std::mt19937 generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    constexpr int BATCH_SIZE = 1;
    constexpr int SEQLEN = 512 * 1024;
    constexpr int HIDDEN_DIM = 128;
    static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be even for RoPE.");

    const size_t total_elements = static_cast<size_t>(BATCH_SIZE) * SEQLEN * HIDDEN_DIM;
    const size_t total_bytes = total_elements * sizeof(bf16);

    // Host input buffer initialization.
    std::vector<bf16> host_input(total_elements);
    std::generate(host_input.begin(), host_input.end(), [&]() {
        return static_cast<bf16>(distribution(generator));
    });

    std::vector<bf16> host_reference(total_elements);
    const auto cpu_start = std::chrono::steady_clock::now();
    ComputeCpuRope(host_input.data(), host_reference.data(), BATCH_SIZE, SEQLEN, HIDDEN_DIM);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU RoPE duration: " << cpu_ms * 1000.0 << " us" << std::endl;

    // Device memory allocation and input transfer.
    bf16* device_input = nullptr;
    bf16* device_output = nullptr;
    CheckCuda(cudaMalloc(&device_input, total_bytes), "cudaMalloc device_input");
    CheckCuda(cudaMalloc(&device_output, total_bytes), "cudaMalloc device_output");

    CheckCuda(cudaMemcpy(device_input, host_input.data(), total_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host_input to device_input");
    CheckCuda(cudaMemset(device_output, 0, total_bytes), "cudaMemset device_output");

    // CUDA events for timing.
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start_event");
    CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop_event");

    constexpr int kThreadsPerBlock = 256;
    constexpr int kValuesPerThread = 8;  // number of bf16 values per thread along hidden dim.
    constexpr int kThreadsPerToken = CeilDiv(HIDDEN_DIM, kValuesPerThread);
    constexpr int kTokensPerBlock = kThreadsPerBlock / kThreadsPerToken;  // number of tokens per block.
    constexpr int kHiddenPerTile = kThreadsPerToken * kValuesPerThread;
    static_assert(kValuesPerThread % 8 == 0, "kValuesPerThread must be multiple of 8 for vectorized loads/stores");
    static_assert(kThreadsPerBlock >= HIDDEN_DIM, "Number of threads per block should not be less than hidden_dim");
    const dim3 num_blocks(CeilDiv(HIDDEN_DIM, kHiddenPerTile),
                          CeilDiv(SEQLEN, kTokensPerBlock),
                          BATCH_SIZE);
    const size_t shared_mem_bytes = static_cast<size_t>(HIDDEN_DIM) * sizeof(float);

    constexpr int kWarmupIterations = 100;

    // Warmup iterations to stabilize timing.
    for (int i = 0; i < kWarmupIterations; ++i) {
        RopeBf16Kernel<kValuesPerThread, kTokensPerBlock>
            <<<num_blocks, kThreadsPerBlock, shared_mem_bytes>>>(
            device_input, device_output, HIDDEN_DIM, SEQLEN);
    }
    CheckCuda(cudaGetLastError(), "RopeKernel warmup");

    CheckCuda(cudaMemset(device_output, 0, total_bytes), "cudaMemset device_output before timing");

    // Timed kernel executions (averaged over 10 runs).
    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventRecord(start_event), "cudaEventRecord kernel start");
    for (int i = 0; i < 10; ++i) {
        RopeBf16Kernel<kValuesPerThread, kTokensPerBlock>
            <<<num_blocks, kThreadsPerBlock, shared_mem_bytes>>>(
            device_input, device_output, HIDDEN_DIM, SEQLEN);
    }
    CheckCuda(cudaGetLastError(), "RopeKernel run");
    CheckCuda(cudaEventRecord(stop_event), "cudaEventRecord kernel stop");
    CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize kernel");
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime kernel");
    elapsed_ms /= 10.0f;  // average over 10 iterations
    std::cout << "RopeKernel duration: " << elapsed_ms * 1000.0f << " us" << std::endl;

    const double bytes_processed = static_cast<double>(total_bytes) * 2.0;
    const double kernel_bw =
        bytes_processed / (static_cast<double>(elapsed_ms) / 1000.0) / static_cast<double>(1ULL << 40);
    std::cout << "RopeKernel reached_mem_bw: " << kernel_bw << " TB/s" << std::endl;

    std::vector<bf16> host_output(total_elements);
    CheckCuda(cudaMemcpy(host_output.data(), device_output, total_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy device_output to host");
    // Validate correctness against CPU reference.
    const float error = ComputeBf16Rmse(host_reference.data(), host_output.data(), total_elements);
    std::cout << "RopeKernel error = " << std::scientific << std::setprecision(4) << error
              << std::defaultfloat << std::endl;

    CheckCuda(cudaEventDestroy(start_event), "cudaEventDestroy start_event");
    CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop_event");
    CheckCuda(cudaFree(device_input), "cudaFree device_input");
    CheckCuda(cudaFree(device_output), "cudaFree device_output");

    return 0;
}
