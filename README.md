<div align="center">
  <h1>⚡ CUDA Operators for Large Language Models</h1>
  <p><strong>CUDA kernels used frequently in LLMs.</strong></p>
  <p>Acer Shadow SH16-73 · RTX 5080 Laptop (16&nbsp;GB GDDR7 · 896&nbsp;GB/s) · CUDA 13.0</p>
</div>

<div align="center">
  <a href="#"><img src="https://img.shields.io/badge/CUDA-13.0-3DDC84?style=flat-square" alt="CUDA 13.0 badge"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-RTX%205080%20Laptop-006DC3?style=flat-square" alt="GPU badge"></a>
  <a href="#"><img src="https://img.shields.io/badge/OS-Linux-informational?style=flat-square" alt="OS badge"></a>
</div>

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">

## <span style="color:#ffb600;">Project Overview</span>

This repository hosts CUDA implementations of critical operators used in **Large Language Models (LLMs)**.

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">

## <span style="color:#ffb600;">Core Operators</span>

The following operator families use custom CUDA kernel implementations.

<details open>
<summary><strong>ElementwiseAdd</strong></summary>

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type | GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|cublasSgeam| - |`(4096,4096)`|fp32| 319.488 | 0.573122 |
|ElementwiseAddFp32Kernel | CUDA |`(4096,4096)`|fp32| 293.92 | 0.622977 |
|elementwiseadd_fp32_kernel |Triton |`(4096,4096)`|fp32| 288.111 | 0.636 |
</details>


<details>
<summary><strong>GLU (GLU/SwiGLU/GeGLU)</strong></summary>

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GLU Type|Output Shape|Output Type| GPU Time (us)| GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |:--- |
|glu_bf16_kernel|CUDA |`(128,8192), (8192, 3584), (8192, 3584)`|bf16|GLU/SwiGLU/GeGLU|`(128, 3584)`|fp32  | 500.8 | 30.0186 |

</details>

<details open>
<summary><strong>GEMM</strong></summary>

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type | Accum Type | GPU Time (us)| GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |
|cublasSgemmEx| - |`(4096,4096)`|fp16| fp32 | fp32 | 2251.28 | 61.0492 |
|gemm_fp16_kernel| CUDA |`(4096,4096)`|fp16|fp32  | fp32 | 2099.3 | 65.4691 |
|gemm_fp16_kernel| Triton |`(4096,4096)`|fp16|fp32  |  fp32 |2392.465 | 57.447 |
|gemm_fp8_kernel| Triton |`(4096,4096)`|fp8|fp32  |  fp32 | 1228.735 | 111.854 |

> **Note**: Asynchronous memcpy and similar techniques were intentionally excluded from CUDA kernel to keep the kernels laptop-friendly.
</details>

<details>
<summary><strong>FlashAttention</strong></summary>

**Description**

Inference FlashAttention with grouped-query attention.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|Accum Type| GPU Time (us)| GPU Memory BW (TB/s) |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |
|infer_flash_gqa_bf16_kernel| CUDA |`Batch = 60`<br>`SeqLen=4096`<br>`Q Heads=128`<br>`KV Heads=8`<br>`HeadDim=128`|bf16|fp16|fp32|  1825.86 | 0.503382 |
</details>

<details>
<summary><strong>LayerNorm</strong></summary>

**Description**

Computes LayerNorm along the hidden dimension.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|layernorm_fp32_kernel| CUDA |`(4096,7168)`|fp32|  368.64 | 0.57949 |
</details>

<details>
<summary><strong>ReduceSum</strong></summary>

**Description**

Computes rowsum of a 2D matrix.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type| GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |
|cublasSgemv| |`(4096,4096)`|fp32|`(4096,)`| 126.08 | 0.484217 |
|ReduceSumFp32Kernel (CUDA)| CUDA |`(4096,4096)`|fp32|`(4096,)`| 90.144  | 0.67725 |
|reducesum_fp32_kernel (Triton)| Triton |`(4096,4096)`|fp32|`(4096,)`| 111.682 | 0.547 |
</details>

<details>
<summary><strong>RMSNorm</strong></summary>

**Description**

Computes RMSNorm along the hidden dimension.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|rmsnorm_fp32_kernel| CUDA |`(4096,7168)`|fp32|  370.688 | 0.576288 |
</details>

<details>
<summary><strong>Softmax</strong></summary>

**Description**

Online safe softmax.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|softmax_fp32_kernel| CUDA |`(128,16384)`|fp32|  20.096 | 0.759295 |
|softmax_fp32_kernel| CUDA |`(32, 131072)`|fp32|   61.44 | 0.496705 |
</details>

<details>
<summary><strong>Sort</strong></summary>

**Description**

- Bitonic sort for power-of-two arrays using shared-memory tile sorting followed by global merge.
- LSD radix sort with histogram + prefix-sum + scatter passes.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type | GPU Time (us)|
| :--- | :--- | :--- |:--- |:--- |
|BitonicSortFp32Kernel| CUDA |`BATCH_SIZE=60`<br> `ARRAY_LENGTH=2K`|fp32| 55.936 |
|| |`BATCH_SIZE=1`<br> `ARRAY_LENGTH=1M`|fp32| 1367.26 |
|RadixSortFp32Kernel| CUDA |`BATCH_SIZE=60`<br> `ARRAY_LENGTH=2K`|fp32| 136.768 |
|| |`BATCH_SIZE=1`<br> `ARRAY_LENGTH=1M`|fp32| 200.064 |
</details>


<details>
<summary><strong>TopK</strong></summary>

**Description**

Radix Select + Blelloch Scan.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)|
| :--- | :--- | :--- |:--- |:--- |
|radixSelectTopK_fp32_kernel| CUDA |`(4, 131072)`|fp32| 65.05   |
</details>

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">
<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">

## 🚀 <span style="color:#ffb600;">Quick Start</span>

```bash
nvcc -o softmax.o -gencode=arch=compute_120,code=sm_120 -O3 Softmax/softmax.cu
./softmax
```

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">
