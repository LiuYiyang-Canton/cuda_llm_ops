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

<details>
<summary><strong>ElementwiseAdd</strong></summary>

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type | GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|cublasSgeam| - |`(4096,4096)`|fp32| 319.488 | 0.573122 |
|ElementwiseAddFp32Kernel | CUDA |`(4096,4096)`|fp32| 293.92 | 0.622977 |
|elementwiseadd_fp32_kernel |Triton |`(4096,4096)`|fp32| 288.111 | 0.636 |
</details>


<details>
<summary><strong>GLU</strong></summary>

**Description**

For input $X \in \mathbb{R}^{B \times d_{\text{in}}}$ and projection weights $W_a, W_b \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, the gated output is

$$Y = (X W_a) \odot \sigma(X W_b), \quad Y \in \mathbb{R}^{B \times d_{\text{out}}}$$

where $\sigma$ is a configurable activation (e.g., sigmoid for GLU, SiLU for SwiGLU, GELU for GeGLU).

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type| GPU Time (us)| GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |
|glu_bf16_kernel|CUDA |`BATCH=128`<br>`IN_DIM=8192`<br>`OUT_DIM=3584`|bf16|fp32  | 311.872 | 48.2034 |

</details>

<details>
<summary><strong>GEMM</strong></summary>

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type | Accum Type | GPU Time (us)| GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |
|cublasSgemmEx| - |`M=N=K=4096`|fp16| fp32 | fp32 | 2251.28 | 61.0492 |
|gemm_fp16_kernel| CUDA |`M=N=K=4096`|fp16|fp32  | fp32 | 2099.3 | 65.4691 |
|gemm_fp16_kernel| Triton |`M=N=K=4096`|fp16|fp32  |  fp32 | 2388.533 | 57.541 |
|gemm_fp8_kernel| Triton |`M=N=K=4096`|fp8_e4m3|fp32  |  fp32 | 1177.447 | 116.726 |

</details>

<details>
<summary><strong>FlashGQA</strong></summary>

**Description**

FlashGQA for inference.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|Accum Type| GPU Time (us)| GPU Memory BW (TB/s) |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |
|infer_flash_gqa_bf16_kernel| CUDA |`Batch = 60`<br>`SeqLen = 4096`<br>`Q Heads = 128`<br>`KV Heads = 8`<br>`HeadDim = 128`|bf16|bf16|fp32|  1825.86 | 0.503382 |
|infer_flash_gqa_bf16_kernel| Triton |`Batch = 60`<br>`SeqLen = 4096`<br>`Q Heads = 128`<br>`KV Heads = 8`<br>`HeadDim = 128`|bf16|bf16|fp32|  1850.12 | 0.496780 |
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
<summary><strong>Linear Attention: SSD (Mamba2)</strong></summary>

**Description**

Chunkise implementation of State Space Duality from Mamba2.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (ms)| GPU TFLOPS|
| :--- | :--- | :--- |:--- |:--- |:--- |
|ssd_mamba2_kernel| Triton |`Batch = 1`<br>`SeqLen = 128K`<br>`BC Heads = 1`<br>`X Heads = 8`<br>`StateDim = 64`<br>`HeadDim = 64`<br>`Chunk size = 128`|bf16 (except the fp32 decay) |  2.609 | 14.017 |
</details>

<details>
<summary><strong>Linear Attention: Gated DeltaNet</strong></summary>

**Description**

Chunkise implementation of Gated DeltaNet.

**Performance**

| Kernel Type | Input Shape | Input Type |GPU Time (ms)| GPU TFLOPS|
| :--- | :--- |:--- |:--- |:--- |
|Triton |`Batch = 1`<br>`SeqLen = 128K`<br>`Num Heads = 8`<br>`HeadDim = 64`<br>`Chunk size = 64`|bf16 (except the fp32 decay) |  7.707 | 9.660 |
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
<summary><strong>RoPE</strong></summary>

**Description**

Applies rotary position embeddings along the hidden dimension for bf16 inputs/outputs.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |
|RopeBf16Kernel| CUDA |`BATCH = 1`<br>`SEQLEN = 512K`<br> `HIDDEN = 128`|bf16|bf16|510.112 | 0.478602 |
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
|BitonicSortFp32Kernel| CUDA |`BATCH = 60`<br> `ARRAY_LENGTH = 2K`|fp32| 55.936 |
|| |`BATCH = 1`<br> `ARRAY_LENGTH = 1M`|fp32| 1367.26 |
|RadixSortFp32Kernel| CUDA |`BATCH = 60`<br> `ARRAY_LENGTH = 2K`|fp32| 136.768 |
|| |`BATCH = 1`<br> `ARRAY_LENGTH = 1M`|fp32| 200.064 |
</details>


<details>
<summary><strong>TopK</strong></summary>

**Description**

Radix Select + Blelloch Scan.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)|
| :--- | :--- | :--- |:--- |:--- |
|radixSelectTopK_fp32_kernel| CUDA |`BATCH = 4`<br>`SEQLEN = 128K`<br>`TopK = 2K`|fp32| 65.05   |
</details>

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">
<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">

## 🚀 <span style="color:#ffb600;">Quick Start</span>

```bash
nvcc -o softmax.o -gencode=arch=compute_120,code=sm_120 -O3 Softmax/softmax.cu
./softmax.o
```

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">
