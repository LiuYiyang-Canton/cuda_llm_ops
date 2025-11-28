// A custom cuda implementation of Inference Flash GQA
// Input and output type should both be bf16, accumulated in fp32

// Compilation command: nvcc -o infer_gqa.o -gencode=arch=compute_120,code=sm_120  infer_flash_gqa.cu   -Xcompiler -fopenmp -O3 --use_fast_math

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <cassert>
#include <cuda_bf16.h>
#include "omp.h"
#include "mma.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace nvcuda;
using bf16_t = __nv_bfloat16;
namespace cg = cooperative_groups;

// We use Llama 3.1 70B architecture
// We assume a batch size of 1 and only 1 KV head
#define BATCH_SIZE 60
#define SEQLEN 4096
#define NUM_QUERY_HEADS 128
#define NUM_KV_HEADS 8
#define GROUP_SIZE ((NUM_QUERY_HEADS) / (NUM_KV_HEADS))
#define HEAD_DIM 128

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)
#define WARP_SIZE 32

// Utility functions to get the 1D index of a 3D/4D tensor
#define GET_1D_INDEX_FROM_3D(i1, i2, i3, D2, D3) ((i1) * (D2) * (D3) + (i2) * (D3) + (i3))
#define GET_1D_INDEX_FROM_4D(i1, i2, i3, i4, D2, D3, D4) ((i1) * (D2) * (D3) * (D4) + (i2) * (D3) * (D4) + (i3) * (D4) + (i4))

// --- Kernel Configuration ---
// Each thread block processes every pair of (batch, head group).
#define FLASH_BLOCK_SIZE 128
#define FLASH_NUM_BLOCKS CEIL_DIV(SEQLEN, FLASH_BLOCK_SIZE)

// The finest granularity of WMMA computation 
#define FRAGMENT_SIZE 16

// Padding added to shared memory to lessen bank conflicts
#define SHARED_MEM_BF16_PADDING 8
#define SHARED_MEM_FP32_PADDING 0

// Number of elements per vectorized load
#define NUM_ELEMENTS_PER_LOAD (sizeof(int4) / sizeof(bf16_t))
#define NUM_ELEMENTS_PER_FP32_LOAD (sizeof(int4) / sizeof(float))

// For the first MM (Matrix Multilpication), each thread block performs a GROUP_SIZE x HEAD_DIM x FLASH_BLOCK_SIZE MM
#define MM1_BLOCK_M GROUP_SIZE
#define MM1_BLOCK_N FLASH_BLOCK_SIZE

// For the first MM, number of warps along each dimension
#define MM1_BLOCK_NUM_WARPS_M 1
#define MM1_BLOCK_NUM_WARPS_N 8
#define BLOCK_NUM_WARPS (MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N)

// Each warp computes MM1_WARP_M X MM1_WARP_N output elements
#define MM1_WARP_M (MM1_BLOCK_M / MM1_BLOCK_NUM_WARPS_M)
#define MM1_WARP_N (MM1_BLOCK_N / MM1_BLOCK_NUM_WARPS_N)

// How many fragments along M and N dimensions in a warp
#define MM1_WARP_NUM_FRAGMENTS_M (MM1_WARP_M / FRAGMENT_SIZE)
#define MM1_WARP_NUM_FRAGMENTS_N (MM1_WARP_N / FRAGMENT_SIZE)

// Total number of threads in a block
#define BLOCK_NUM_THREADS (MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N * WARP_SIZE)

// Number of fragments along the reduce dimension computed per iteration
#define MM1_K_FRAGMENTS_PER_ITER 2

// Shape of a tile of matrix A loaded each time per block
#define MM1_A_TILE_M MM1_BLOCK_M
#define MM1_A_TILE_K (FRAGMENT_SIZE * MM1_K_FRAGMENTS_PER_ITER)

// Shape of a tile of matrix B loaded each time per block (A_TILE_K should be equal to B_TILE_K)
#define MM1_B_TILE_N MM1_BLOCK_N
#define MM1_B_TILE_K (FRAGMENT_SIZE * MM1_K_FRAGMENTS_PER_ITER)

// Number of threads loading a row of A_TILE
#define MM1_A_TILE_NUM_THREADS_PER_ROW (MM1_A_TILE_K / NUM_ELEMENTS_PER_LOAD)
// Number of rows loaded by each warp in A_TILE (assuming BLOCK_NUM_WARPS_M divides A_TILE_M)
#define MM1_A_TILE_NUM_ROWS_PER_WARP (MM1_A_TILE_M / BLOCK_NUM_WARPS)
// Number of rows loaded by each warp for each iteration
#define MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM1_A_TILE_NUM_THREADS_PER_ROW)
// Number of iterations required for each warp to load its portion of A_TILE
#define MM1_A_TILE_NUM_LOAD_ITERS CEIL_DIV(MM1_A_TILE_NUM_ROWS_PER_WARP, MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER)
// Number of threads loading a column of B_TILE
#define MM1_B_TILE_NUM_THREADS_PER_COL (MM1_B_TILE_K / NUM_ELEMENTS_PER_LOAD)
// Number of columns loaded by each warp in B_TILE (assuming BLOCK_NUM_WARPS_N divides B_TILE_N)
#define MM1_B_TILE_NUM_COLS_PER_WARP (MM1_B_TILE_N / BLOCK_NUM_WARPS)
// Number of columns loaded by each warp for each iteration
#define MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER (WARP_SIZE / MM1_B_TILE_NUM_THREADS_PER_COL)
// Number of iterations required for each warp to load its portion of A_TILE
#define MM1_B_TILE_NUM_LOAD_ITERS (MM1_B_TILE_NUM_COLS_PER_WARP / (MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER))

// Leading dimension of shared memory buffers
#define MM1_SHARED_MEM_BF16_STRIDE (MM1_A_TILE_K + SHARED_MEM_BF16_PADDING)
#define MM1_SHARED_MEM_FP32_STRIDE (MM1_BLOCK_N + SHARED_MEM_FP32_PADDING)

// For each iteration in softmax, each warp handles one row
#define SOFTMAX_SHARED_MEM_BF16_STRIDE (MM1_BLOCK_N + SHARED_MEM_BF16_PADDING)
#define SOFTMAX_NUM_ROWS_PER_WARP (MM1_BLOCK_M / BLOCK_NUM_WARPS)

// For the second MM (Matrix Multilpication), each thread block performs a GROUP_SIZE x FLASH_BLOCK_SIZE x HEAD_DIM MM
#define MM2_BLOCK_M GROUP_SIZE
#define MM2_BLOCK_N HEAD_DIM

// Number of warps along each dimension
#define MM2_BLOCK_NUM_WARPS_M 1
#define MM2_BLOCK_NUM_WARPS_N 8

// Each warp computes MM1_WARP_M X MM1_WARP_N output elements
#define MM2_WARP_M (MM2_BLOCK_M / MM2_BLOCK_NUM_WARPS_M)
#define MM2_WARP_N (MM2_BLOCK_N / MM2_BLOCK_NUM_WARPS_N)

// How many fragments along M and N dimensions in a warp
#define MM2_WARP_NUM_FRAGMENTS_M (MM2_WARP_M / FRAGMENT_SIZE)
#define MM2_WARP_NUM_FRAGMENTS_N (MM2_WARP_N / FRAGMENT_SIZE)

// Number of fragments along the reduce dimension computed per iteration
#define MM2_K_FRAGMENTS_PER_ITER 8

// Shape of a tile of matrix A loaded each time per block
#define MM2_A_TILE_M MM2_BLOCK_M
#define MM2_A_TILE_K (FRAGMENT_SIZE * MM2_K_FRAGMENTS_PER_ITER)

// Shape of a tile of matrix B loaded each time per block (A_TILE_K should be equal to B_TILE_K)
// For the second MM, matrix B is row-major, so its layout is K x N
#define MM2_B_TILE_K (FRAGMENT_SIZE * MM2_K_FRAGMENTS_PER_ITER)
#define MM2_B_TILE_N MM2_BLOCK_N

// Number of threads loading a column of B_TILE
#define MM2_B_TILE_NUM_THREADS_PER_COL (MM2_B_TILE_N / NUM_ELEMENTS_PER_LOAD)
// Number of rows loaded by each warp in B_TILE
#define MM2_B_TILE_NUM_ROWS_PER_WARP (MM2_B_TILE_K / BLOCK_NUM_WARPS)
// Number of rows loaded by each warp for each iteration
#define MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM2_B_TILE_NUM_THREADS_PER_COL)
// Number of iterations required for each warp to load its portion of A_TILE
#define MM2_B_TILE_NUM_LOAD_ITERS (MM2_B_TILE_NUM_ROWS_PER_WARP / (MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER))

// Updating OIntermediate
#define MM2_O_TILE_NUM_THREADS_PER_ROW (MM2_BLOCK_N / NUM_ELEMENTS_PER_FP32_LOAD)
#define MM2_O_TILE_NUM_ROWS_PER_WARP (MM2_BLOCK_M / BLOCK_NUM_WARPS)
#define MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER (WARP_SIZE / MM2_O_TILE_NUM_THREADS_PER_ROW)
#define MM2_O_TILE_NUM_LOAD_ITERS (MM2_O_TILE_NUM_ROWS_PER_WARP / (MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER))

// Leading dimension of shared memory buffers
#define MM2_SHARED_MEM_BF16_STRIDE (SOFTMAX_SHARED_MEM_BF16_STRIDE)
#define MM2_V_SHARED_MEM_BF16_STRIDE (MM2_B_TILE_N + SHARED_MEM_BF16_PADDING)
#define MM2_SHARED_MEM_FP32_STRIDE (MM2_BLOCK_N + SHARED_MEM_FP32_PADDING)

// Final writing of O
#define UPDATE_O_NUM_ROWS_PER_WARP (MM2_BLOCK_M / BLOCK_NUM_WARPS)

// --- End of Kernel Configuration ---

// Each block.y handles one batch
// Each block.x handles one group of query heads
__global__ void infer_flash_gqa_bf16_kernel(const bf16_t* __restrict__ Q, const bf16_t* __restrict__ K, const bf16_t* __restrict__ V, float scale, 
                                            bf16_t* __restrict__ O, float* __restrict__ OIntermediate) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batchIdx = by;
    int groupIdx = bx;
    int warpID = tx / WARP_SIZE;
    int laneID = tx % WARP_SIZE;

    // Remap input's start positions
    Q = &Q[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * GROUP_SIZE, 0, NUM_QUERY_HEADS, HEAD_DIM)];
    K = &K[GET_1D_INDEX_FROM_4D(batchIdx, 0, groupIdx, 0, SEQLEN, NUM_KV_HEADS, HEAD_DIM)];
    V = &V[GET_1D_INDEX_FROM_4D(batchIdx, 0, groupIdx, 0, SEQLEN, NUM_KV_HEADS, HEAD_DIM)];
    O = &O[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * GROUP_SIZE, 0, NUM_QUERY_HEADS, HEAD_DIM)];
    OIntermediate = &OIntermediate[GET_1D_INDEX_FROM_3D(batchIdx, groupIdx * GROUP_SIZE, 0, NUM_QUERY_HEADS, HEAD_DIM)];

    // Static shared memory for rowmax and rowsum
    __shared__ float rowmax[GROUP_SIZE];
    __shared__ float newrowmax[GROUP_SIZE];
    __shared__ float rowsum[GROUP_SIZE];
    __shared__ float newrowsum[GROUP_SIZE];

    // Initialize rowmax and rowsum
    if (tx == 0) {
        for (int head = 0; head < GROUP_SIZE; ++head) {
            rowmax[head] = -INFINITY;
            rowsum[head] = 0.0;
        }
    }

    // Dynamic shared memory for tiles of A and B
    extern __shared__ bf16_t shared_mem[];
    bf16_t* qs = &shared_mem[0];
    bf16_t* ks = &shared_mem[MM1_A_TILE_M * MM1_SHARED_MEM_BF16_STRIDE];
    for (int flashIter = 0; flashIter < FLASH_NUM_BLOCKS; ++flashIter) {

        // WMMA fragments for accumulation s = q * k^T
        wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float> sFrag[MM1_WARP_NUM_FRAGMENTS_M][MM1_WARP_NUM_FRAGMENTS_N];

        // Initialize output fragments to zero
    #pragma unroll
        for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                wmma::fill_fragment(sFrag[i][j], 0.0f);
            }
        }

        // The first MM: S = Q * K^T
        // Each iteration loads BLOCK_M x A_TILE_K tile of A and BLOCK_N * B_TILE_K tile of B into shared memory, and
        // computes partial GEMM results and accumulates into output fragments
    #pragma unroll
        for (int innerTile = 0; innerTile < HEAD_DIM; innerTile += MM1_A_TILE_K) {
            // Load A_TILE into shared memory
    #pragma unroll
            for (int iter = 0; iter < MM1_A_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM1_A_TILE_NUM_ROWS_PER_WARP + (laneID / MM1_A_TILE_NUM_THREADS_PER_ROW) + iter * MM1_A_TILE_NUM_ROWS_PER_WARP_PER_ITER;
                if (row < (warpID + 1) * MM1_A_TILE_NUM_ROWS_PER_WARP) {
                    int col = (laneID % MM1_A_TILE_NUM_THREADS_PER_ROW) * NUM_ELEMENTS_PER_LOAD;
                    *(int4*)&qs[row * MM1_SHARED_MEM_BF16_STRIDE + col] = *(int4*)&Q[row * HEAD_DIM + innerTile + col];
                }
            }
            // Load B_TILE into shared memory
    #pragma unroll
            for (int iter = 0; iter < MM1_B_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM1_B_TILE_NUM_COLS_PER_WARP + (laneID / MM1_B_TILE_NUM_THREADS_PER_COL) + iter * MM1_B_TILE_NUM_COLS_PER_WARP_PER_ITER;
                int col = (laneID % MM1_B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
                *(int4*)&ks[row * MM1_SHARED_MEM_BF16_STRIDE + col] = *(int4*)&K[row * NUM_KV_HEADS * HEAD_DIM + innerTile + col];
            }
            __syncthreads();
            // Compute partial GEMM results
    #pragma unroll
            for (int innerFrag = 0; innerFrag < MM1_K_FRAGMENTS_PER_ITER; ++innerFrag) {
                wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major> qFrag[MM1_WARP_NUM_FRAGMENTS_M];
                wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::col_major> kFrag[MM1_WARP_NUM_FRAGMENTS_N];
                // Load A fragments from shared memory to WMMA fragments
    #pragma unroll
                for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
                    int row = warpID / MM1_BLOCK_NUM_WARPS_N * MM1_WARP_M + i * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(qFrag[i], &qs[row * MM1_SHARED_MEM_BF16_STRIDE + col], MM1_SHARED_MEM_BF16_STRIDE);
                }
    #pragma unroll
                for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                    int row = warpID % MM1_BLOCK_NUM_WARPS_N * MM1_WARP_N + j * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(kFrag[j], &ks[row * MM1_SHARED_MEM_BF16_STRIDE + col], MM1_SHARED_MEM_BF16_STRIDE);
                }

                // Perform matrix multiplication and accumulate results
    #pragma unroll
                for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
                    for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                        wmma::mma_sync(sFrag[i][j], qFrag[i], kFrag[j], sFrag[i][j]);
                    }
                }
            }

            // Synchronize before loading the next tile
            __syncthreads();
        }

        // We write S to shared memory started at some offset, leaving some space for P (bf16)
        float* mm1Result = (float*)&shared_mem[MM1_BLOCK_M * SOFTMAX_SHARED_MEM_BF16_STRIDE];
    #pragma unroll
        for (int i = 0; i < MM1_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM1_WARP_NUM_FRAGMENTS_N; ++j) {
                int row = (warpID / MM1_BLOCK_NUM_WARPS_N) * MM1_WARP_M + i * FRAGMENT_SIZE;
                int col = (warpID % MM1_BLOCK_NUM_WARPS_N) * MM1_WARP_N + j * FRAGMENT_SIZE;
                wmma::store_matrix_sync(&mm1Result[row * (MM1_SHARED_MEM_FP32_STRIDE) + col], sFrag[i][j], MM1_SHARED_MEM_FP32_STRIDE, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Softmax
        bf16_t* ps = &shared_mem[0];
        auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
        for (int softmaxIter = 0; softmaxIter < SOFTMAX_NUM_ROWS_PER_WARP; ++softmaxIter) {
            int row = warpID * SOFTMAX_NUM_ROWS_PER_WARP + softmaxIter;
            float rowmaxLocal = rowmax[row];
            float rowsumLocal = 0.f;
            if (laneID * 4 >= FLASH_BLOCK_SIZE) {
                break;
            }
            float4 sValue = *(float4*)&mm1Result[row * MM1_SHARED_MEM_FP32_STRIDE + laneID * 4];
            rowmaxLocal = fmaxf(sValue.x * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.y * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.z * scale, rowmaxLocal);
            rowmaxLocal = fmaxf(sValue.w * scale, rowmaxLocal);
            newrowmax[row] = cg::reduce(warp, rowmaxLocal, cg::greater<float>());
            __syncwarp();
            rowsumLocal = expf(sValue.x * scale - newrowmax[row]) + expf(sValue.y * scale - newrowmax[row]) + expf(sValue.z * scale - newrowmax[row]) + expf(sValue.w * scale - newrowmax[row]);
            newrowsum[row] = cg::reduce(warp, rowsumLocal, cg::plus<float>());
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4] = __float2bfloat16(expf(sValue.x * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 1] = __float2bfloat16(expf(sValue.y * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 2] = __float2bfloat16(expf(sValue.z * scale - newrowmax[row]));
            ps[row * SOFTMAX_SHARED_MEM_BF16_STRIDE + laneID * 4 + 3] = __float2bfloat16(expf(sValue.w * scale - newrowmax[row]));
        }

        __syncthreads();
        // WMMA fragments for accumulation o = p * v
        wmma::fragment<wmma::accumulator, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, float> oFrag[MM2_WARP_NUM_FRAGMENTS_M][MM2_WARP_NUM_FRAGMENTS_N];

        // Initialize output fragments to zero
    #pragma unroll
        for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                wmma::fill_fragment(oFrag[i][j], 0.0f);
            }
        }

        // The second MM: O = P * V
        // Each iteration loads BLOCK_M x A_TILE_K tile of A and BLOCK_N * B_TILE_K tile of B into shared memory, and
        // computes partial GEMM results and accumulates into output fragments
        bf16_t* vs = &shared_mem[MM2_A_TILE_M * MM2_SHARED_MEM_BF16_STRIDE];
    #pragma unroll
        for (int innerTile = 0; innerTile < FLASH_BLOCK_SIZE; innerTile += MM2_B_TILE_K) {
            // P is already in shared memory
            #pragma unroll
            for (int iter = 0; iter < MM2_B_TILE_NUM_LOAD_ITERS; ++iter) {
                int row = warpID * MM2_B_TILE_NUM_ROWS_PER_WARP + (laneID / MM2_B_TILE_NUM_THREADS_PER_COL) + iter * MM2_B_TILE_NUM_ROWS_PER_WARP_PER_ITER;
                int col = (laneID % MM2_B_TILE_NUM_THREADS_PER_COL) * NUM_ELEMENTS_PER_LOAD;
                *(int4*)&vs[row * MM2_V_SHARED_MEM_BF16_STRIDE + col] = *(int4*)&V[(row + innerTile) * NUM_KV_HEADS * HEAD_DIM + col];
            }
            __syncthreads();
            // Compute partial GEMM results
    #pragma unroll
            for (int innerFrag = 0; innerFrag < MM2_K_FRAGMENTS_PER_ITER; ++innerFrag) {
                wmma::fragment<wmma::matrix_a, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major> pFrag[MM2_WARP_NUM_FRAGMENTS_M];
                wmma::fragment<wmma::matrix_b, FRAGMENT_SIZE, FRAGMENT_SIZE, FRAGMENT_SIZE, bf16_t, wmma::row_major> vFrag[MM2_WARP_NUM_FRAGMENTS_N];
                // Load A fragments from shared memory to WMMA fragments
    #pragma unroll
                for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
                    int row = warpID / MM2_BLOCK_NUM_WARPS_N * MM2_WARP_M + i * FRAGMENT_SIZE;
                    int col = innerFrag * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(pFrag[i], &ps[row * MM2_SHARED_MEM_BF16_STRIDE + innerTile + col], MM2_SHARED_MEM_BF16_STRIDE);
                }
    #pragma unroll
                for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                    int row = innerFrag * FRAGMENT_SIZE;
                    int col = warpID % MM2_BLOCK_NUM_WARPS_N * MM2_WARP_N + j * FRAGMENT_SIZE;
                    wmma::load_matrix_sync(vFrag[j], &vs[row * MM2_V_SHARED_MEM_BF16_STRIDE + col], MM2_V_SHARED_MEM_BF16_STRIDE);
                }

                // Perform matrix multiplication and accumulate results
    #pragma unroll
                for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
                    for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                        wmma::mma_sync(oFrag[i][j], pFrag[i], vFrag[j], oFrag[i][j]);
                    }
                }
            }

            // Synchronize before loading the next tile
            __syncthreads();
        }
            // Write result to shared memory
        float* mm2Result = (float*)&shared_mem[0];
    #pragma unroll
        for (int i = 0; i < MM2_WARP_NUM_FRAGMENTS_M; ++i) {
    #pragma unroll
            for (int j = 0; j < MM2_WARP_NUM_FRAGMENTS_N; ++j) {
                int row = (warpID / MM2_BLOCK_NUM_WARPS_N) * MM2_WARP_M + i * FRAGMENT_SIZE;
                int col = (warpID % MM2_BLOCK_NUM_WARPS_N) * MM2_WARP_N + j * FRAGMENT_SIZE;
                wmma::store_matrix_sync(&mm2Result[row * (MM2_SHARED_MEM_FP32_STRIDE) + col], oFrag[i][j], MM2_SHARED_MEM_FP32_STRIDE, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int iter = 0; iter < MM2_O_TILE_NUM_LOAD_ITERS; ++iter) {
            int row = warpID * MM2_O_TILE_NUM_ROWS_PER_WARP + (laneID / MM2_O_TILE_NUM_THREADS_PER_ROW) + iter * MM2_O_TILE_NUM_ROWS_PER_WARP_PER_ITER;
            int col = (laneID % MM2_O_TILE_NUM_THREADS_PER_ROW) * NUM_ELEMENTS_PER_FP32_LOAD;
            float updateCoef = expf(rowmax[row] - newrowmax[row]);
            if (flashIter > 0) {
                float4 oValueOld = *(float4*)&OIntermediate[row * HEAD_DIM + col];
                float4 oValueNew = *(float4*)&mm2Result[row * MM2_SHARED_MEM_FP32_STRIDE + col];
                *(float4*)&OIntermediate[row * HEAD_DIM + col] = {
                    oValueOld.x * updateCoef + oValueNew.x,
                    oValueOld.y * updateCoef + oValueNew.y,
                    oValueOld.z * updateCoef + oValueNew.z,
                    oValueOld.w * updateCoef + oValueNew.w
                };
            } else {
                *(float4*)&OIntermediate[row * HEAD_DIM + col] = *(float4*)&mm2Result[row * MM2_SHARED_MEM_FP32_STRIDE + col];
            }
        }
        __syncthreads();
        if (tx < GROUP_SIZE) {
            float updateCoef = expf(rowmax[tx] - newrowmax[tx]);
            rowsum[tx] = rowsum[tx] * updateCoef + newrowsum[tx];
            rowmax[tx] = newrowmax[tx];
        }
        K += FLASH_BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
        V += FLASH_BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    }

    // Final normalize of O
    __syncthreads();
    for (int updateOIter = 0; updateOIter < UPDATE_O_NUM_ROWS_PER_WARP; ++updateOIter) {
        int row = warpID * UPDATE_O_NUM_ROWS_PER_WARP + updateOIter;
        if (laneID * 4 >= HEAD_DIM) {
            break;
        }
        float normalizer = 1.0f / rowsum[row];
        float4 oValue = *(float4*)&OIntermediate[row * MM2_SHARED_MEM_FP32_STRIDE + laneID * 4];
        O[row * HEAD_DIM + laneID * 4] = __float2bfloat16(oValue.x * normalizer);
        O[row * HEAD_DIM + laneID * 4 + 1] = __float2bfloat16(oValue.y * normalizer);
        O[row * HEAD_DIM + laneID * 4 + 2] = __float2bfloat16(oValue.z * normalizer);
        O[row * HEAD_DIM + laneID * 4 + 3] = __float2bfloat16(oValue.w * normalizer);
    }
}

float ComputeBF16RMSE(const bf16_t* __restrict__ golden, const bf16_t* __restrict x, size_t numElements) {
    float error = 0;
    float norm = 0;
    for (int i = 0; i < numElements; ++i) {
        float y1 = __bfloat162float(golden[i]);
        float y2 = __bfloat162float(x[i]);
        
        error += (y1 - y2) * (y1 - y2);
        norm += y1 * y1;
    }
    return std::sqrt(error) / std::sqrt(norm);
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
    assert(SEQLEN % 4 == 0);
    assert(MM1_BLOCK_NUM_WARPS_M * MM1_BLOCK_NUM_WARPS_N == MM2_BLOCK_NUM_WARPS_M * MM2_BLOCK_NUM_WARPS_N);
    // For the second MM, since matrix B is row-major, to ensure consistent shared memory stride with matrix A, we need the following
    assert(SOFTMAX_SHARED_MEM_BF16_STRIDE == MM2_SHARED_MEM_BF16_STRIDE);
    assert(WARP_SIZE * 4 >= FLASH_BLOCK_SIZE);
    assert(WARP_SIZE * 4 >= HEAD_DIM);
    // Define random data generation
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    // Generate random input
    // Input: Q, K, V
    // Shape: BSND, or (BATCH_SIZE, SEQLEN, NUM_HEADS, HEAD_DIM)
    bf16_t* QHost = new bf16_t[BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM];
    bf16_t* KHost = new bf16_t[BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM];
    bf16_t* VHost = new bf16_t[BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM];
    for (int i = 0; i < BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM; ++i) {
        QHost[i] = bf16_t(distribution(generator));
    }
    for (int i = 0; i < BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM; ++i) {
        KHost[i] = bf16_t(distribution(generator));
    }
    for (int i = 0; i < BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM; ++i) {
        VHost[i] = bf16_t(distribution(generator));
    }
    // Define scale
    float scale = rsqrtf((float)HEAD_DIM);

    // CPU implementation of Prefill GQA
    float* SHostCPU = new float[BATCH_SIZE * NUM_QUERY_HEADS * FLASH_BLOCK_SIZE];
    bf16_t* PHostCPU = new bf16_t[BATCH_SIZE * NUM_QUERY_HEADS * FLASH_BLOCK_SIZE];
    float* OIntermediateHostCPU = new float[BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM];
    bf16_t* OHostCPU = new bf16_t[BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM];
    double t = omp_get_wtime();
#pragma omp parallel for
    for (int batch = 0; batch < BATCH_SIZE; ++batch) {
        for (int head = 0; head < NUM_QUERY_HEADS; ++head) {
            float rowmax = -INFINITY;
            float newrowmax;
            float rowsum = 0.f;
            float newrowsum;
            for (int flashIter = 0; flashIter < FLASH_NUM_BLOCKS; ++flashIter){
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    float qkScore = 0.;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        qkScore += __bfloat162float(QHost[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)]) * \
                                __bfloat162float(KHost[GET_1D_INDEX_FROM_4D(batch, token, head / GROUP_SIZE, d, SEQLEN, NUM_KV_HEADS, HEAD_DIM)]);
                    }
                    SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] = qkScore; // (B, N, S)
                }
                newrowmax = rowmax;
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    newrowmax = fmaxf(newrowmax, SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] * scale);
                }
                newrowsum = 0.f;
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    newrowsum += expf(SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] * scale - newrowmax);
                }
                for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                    PHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] = \
                        bf16_t(expf(SHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)] * scale - newrowmax));
                }

                for (int d = 0; d < HEAD_DIM; ++d) {
                    float o = 0.f;
                    for (int token = flashIter * FLASH_BLOCK_SIZE; token < (flashIter + 1) * FLASH_BLOCK_SIZE; ++token) {
                        o += __bfloat162float(PHostCPU[GET_1D_INDEX_FROM_3D(batch, head, token - flashIter * FLASH_BLOCK_SIZE, NUM_QUERY_HEADS, FLASH_BLOCK_SIZE)]) * \
                            __bfloat162float(VHost[GET_1D_INDEX_FROM_4D(batch, token, head / GROUP_SIZE, d, SEQLEN, NUM_KV_HEADS, HEAD_DIM)]);
                    }
                    if (flashIter == 0) {
                        OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] = o;
                    } else {
                        OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] = \
                            OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] * expf(rowmax - newrowmax) + o;

                    }
                }
                rowsum = rowsum * expf(rowmax - newrowmax) + newrowsum;
                rowmax = newrowmax;
            }
            
            for (int d = 0; d < HEAD_DIM; ++d) {
                OHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] = __float2bfloat16(OIntermediateHostCPU[GET_1D_INDEX_FROM_3D(batch, head, d, NUM_QUERY_HEADS, HEAD_DIM)] / rowsum);
            }
        }
    }
    std::cout << "CPU time = " << omp_get_wtime() - t << "s" << std::endl;

    // Allocate GPU memory
    bf16_t* QDevice;
    bf16_t* KDevice;
    bf16_t* VDevice;
    bf16_t* ODevice;
    float* OIntermediateDevice;
    cudaMalloc(&QDevice, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&KDevice, BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&VDevice, BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&ODevice, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMalloc(&OIntermediateDevice, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(float));
    cudaMemcpy(QDevice, QHost, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(KDevice, KHost, BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(VDevice, VHost, BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyHostToDevice);

    // Temporary space for debugging
    float* SDevice;
    cudaMalloc(&SDevice, BATCH_SIZE * NUM_QUERY_HEADS * SEQLEN * sizeof(float));
    bf16_t* PDevice;
    cudaMalloc(&PDevice, BATCH_SIZE * NUM_QUERY_HEADS * SEQLEN * sizeof(bf16_t));

    // For GPU profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    dim3 numThreads(BLOCK_NUM_THREADS);
    dim3 numBlocks(NUM_KV_HEADS, BATCH_SIZE);
    size_t shmemBytes = std::max((MM1_A_TILE_M + MM1_B_TILE_N) * MM1_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t), \
                                  MM1_BLOCK_M * SOFTMAX_SHARED_MEM_BF16_STRIDE * sizeof(bf16_t) + MM1_BLOCK_M * MM1_SHARED_MEM_FP32_STRIDE * sizeof(float));
    shmemBytes = std::max(shmemBytes, (MM2_A_TILE_M * MM2_SHARED_MEM_BF16_STRIDE + MM2_B_TILE_K * MM2_V_SHARED_MEM_BF16_STRIDE) * sizeof(bf16_t));
    std::cout << "shared memory usage per block: " << shmemBytes / 1024 << " KB" << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (shmemBytes > deviceProp.sharedMemPerBlock) {
        // Increase the limit to the required size
        cudaFuncSetAttribute(infer_flash_gqa_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemBytes);
    }

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        infer_flash_gqa_bf16_kernel<<<numBlocks, numThreads, shmemBytes>>>(QDevice, KDevice, VDevice, scale, ODevice, OIntermediateDevice);
    }
    cudaMemset(ODevice, 0, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t));
    cudaMemset(OIntermediateDevice, 0, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(float));

    cudaEventRecord(start);
    infer_flash_gqa_bf16_kernel<<<numBlocks, numThreads, shmemBytes>>>(QDevice, KDevice, VDevice, scale, ODevice, OIntermediateDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "infer_flash_gqa_bf16_kernel elapsed time: " << elapsedTime * 1000 << " us" << std::endl;
    float reached_mem_bw =  (
                            BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t) + \
                            BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t) + \
                            BATCH_SIZE * SEQLEN * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16_t) + \
                            BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t) \
                            ) / ((float)1024 * 1024 * 1024 * 1024) / (elapsedTime / 1000);
    std::cout << "infer_flash_gqa_bf16_kernel reached_mem_bw: " << reached_mem_bw << " TB/s" << std::endl;
    float peakTFLOPS = ((float)BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * SEQLEN) * 2 / 1e12 / (elapsedTime / 1e3);
    std::cout << "infer_flash_gqa_bf16_kernel peak TFLOPS: " << peakTFLOPS << " TFLOPS" << std::endl;

    bf16_t* OHost = new bf16_t[BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM];
    cudaMemcpy(OHost, ODevice, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    float error = ComputeBF16RMSE(OHostCPU, OHost, BATCH_SIZE * NUM_QUERY_HEADS * HEAD_DIM);
    std::cout << "O error = " << error << std::endl;

    // Clean up
    delete[] QHost;
    delete[] KHost;
    delete[] VHost;
    delete[] OHost;
    delete[] SHostCPU;
    delete[] PHostCPU;
    delete[] OHostCPU;
    
    return 0;
}
