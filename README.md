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
<summary><strong>Engram</strong></summary>

**Description**

CPU hash generation for Engram features.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type | CPU Time (us) |
| :--- | :--- | :--- | :--- | :--- |
|GetNgramHashes| C++ |`B = 1`<br>`T = 128K`<br>`MaxNgram = 3`<br>`Heads = 8`|int64| 638 |
</details>

<details>
<summary><strong>Gated Linear Units</strong></summary>

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
<summary><strong>GQA</strong></summary>

**Description**

- `decode_flash_gqa_bf16_kernel`: decoding phase of inference.
- `flash_gqa_forward_bf16_kernel`: prefill phase of inference or training.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|Accum Type| GPU Time (us)| GPU Memory BW (TB/s) | GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |:--- |
|decode_flash_gqa_bf16_kernel| CUDA |`Batch = 60`<br>`SeqLen = 4096`<br>`Q Heads = 128`<br>`KV Heads = 8`<br>`HeadDim = 128`|bf16|bf16|fp32|  1340 | 0.685 | ---
|decode_flash_gqa_bf16_kernel| Triton |`Batch = 60`<br>`SeqLen = 4096`<br>`Q Heads = 128`<br>`KV Heads = 8`<br>`HeadDim = 128`|bf16|bf16|fp32|  1470 | 0.625 | ---
|flash_gqa_forward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 1`<br>`HeadDim = 128`|bf16|bf16|fp32| 1130 |---| 60.81 |
|flash_gqa_backward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 1`<br>`HeadDim = 128`|bf16|bf16|fp32| 3100 |---| 55.42 |
</details>

<details>
<summary><strong>LayerNorm</strong></summary>

**Description**

Computes LayerNorm along the hidden dimension.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|LayerNormKernel| CUDA |`BATCH = 1`<br>`SEQLEN = 4K`<br>`HIDDEN = 7168`|fp32|  368.64 | 0.57949 |
|LayerNormBackwardKernel| CUDA |`BATCH = 1`<br>`SEQLEN = 128K`<br>`HIDDEN = 2048`|fp32| 6405.992 | 0.655 |
</details>

<details>
<summary><strong>Linear Attention: Gated DeltaNet</strong></summary>

**Description**

Chunkise implementation of Gated DeltaNet.

**Performance**

| Kernel Type | Input Shape | Input Type |GPU Time (ms)| GPU TFLOPS| GPU Memory BW (TB/s)|
| :--- | :--- |:--- |:--- |:--- |:--- |
|Triton |`Batch = 1`<br>`SeqLen = 128K`<br>`Num Heads = 8`<br>`HeadDim = 64`<br>`Chunk size = 64`|bf16 (except the fp32 decay) |  8.338 | 8.929 | 0.426 |
</details>

<details>
<summary><strong>Linear Attention: SSD (Mamba2)</strong></summary>

**Description**

Chunkise implementation of State Space Duality from Mamba2.

**Performance**

| Kernel Type | Input Shape | Input Type |GPU Time (ms)| GPU TFLOPS| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|Triton |`Batch = 1`<br>`SeqLen = 128K`<br>`BC Heads = 1`<br>`X Heads = 8`<br>`StateDim = 64`<br>`HeadDim = 64`<br>`Chunk size = 128`|bf16 (except the fp32 decay) |  2.609 | 14.017 | 0.388
</details>

<details>
<summary><strong>mHC (DeepSeek)</strong></summary>

**Description**

mHCSinkhornKernel: Sinkhorn-Knopp iterations for multi-hyper-connection streams.

**Performance**

| Kernel | Kernel Type | Input | Input Type | GPU Time (us) | GPU Memory BW (GB/s) |
| :--- | :--- | :--- |:--- |:--- |:--- |
|mHCSinkhornKernel| CUDA |`Batch = 2`<br>`SeqLen = 128K`<br>`HC Streams = 4`<br>`Iterations = 20`|fp32| 32.94 | 509.34 |
|mHCSinkhornBackwardKernel| CUDA |`Batch = 2`<br>`SeqLen = 128K`<br>`HC Streams = 4`<br>`Iterations = 20`|fp32| 111.2 | 452.44 |
</details>

<details>
<summary><strong>MLA</strong></summary>

**Description**

- `flash_mla_forward_bf16_kernel`: prefill phase of inference or training.
- `flash_mla_backward_bf16_kernel`: backward pass for Flash MLA.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|Accum Type| GPU Time (us)| GPU Memory BW (TB/s) | GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |:--- |
|flash_mla_forward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 16`<br>`HeadDim(QK) = 192`<br>`HeadDim(V) = 128`|bf16|bf16|fp32| 1440 |---| 59.65 |
|flash_mla_backward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 16`<br>`HeadDim(QK) = 192`<br>`HeadDim(V) = 128`|bf16|bf16|fp32| 4450 |---| 50.19 |
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

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (GB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|RMSNormKernel| CUDA |`BATCH = 1`<br>`SEQLEN = 4K`<br>`HIDDEN = 7168`|fp32|  370.688 | 576.288 |
|RMSNormBackwardKernel| CUDA |`BATCH = 1`<br>`SEQLEN = 128K`<br>`HIDDEN = 2048`|fp32| 6497.6 | 661.089 |
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
<summary><strong>Sliding Window Attention</strong></summary>

**Description**

- `flash_swa_forward_bf16_kernel`: prefill phase of inference or training (with sink).
- `flash_swa_backward_bf16_kernel`: backward pass for Flash SWA (with sink).

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |Output Type|Accum Type| GPU Time (us)| GPU Memory BW (TB/s) | GPU TFLOPS |
| :--- | :--- | :--- |:--- |:--- |:--- |:--- |:--- |:--- |
|flash_swa_forward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 1`<br>`HeadDim(QK) = 128`<br>`HeadDim(V) = 128`<br>`Window = 512`|bf16|bf16|fp32| 280.000 |0.115646| 61.356676 |
|flash_swa_backward_bf16_kernel| Triton |`Batch = 1`<br>`SeqLen = 4096`<br>`Q Heads = 16`<br>`KV Heads = 1`<br>`HeadDim(QK) = 128`<br>`HeadDim(V) = 128`<br>`Window = 512`|bf16|bf16|fp32| 870.000 |0.052992| 49.367440 |
</details>

<details>
<summary><strong>Softmax</strong></summary>

**Description**

Online safe softmax.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|Softmax| CUDA |`(32, 131072)`|fp32|   61.44 | 0.496705 |
|SoftmaxBackward| CUDA |`(32, 131072)`|fp32|   106.3 | 0.44 |
</details>

<details>
<summary><strong>SoftmaxCrossEntropy</strong></summary>

**Description**

Fused softmax + cross-entropy loss.

**Performance**

| Kernel | Kernel Type | Input Shape | Input Type |GPU Time (us)| GPU Memory BW (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|SoftmaxCrossEntropy| CUDA |`(32, 131072)`|fp32|   37.6 | 0.415 |
|SoftmaxCrossEntropyBackward| CUDA |`(32, 131072)`|fp32|   34.816 | 0.342 |
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
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
./build/examples/test_elementwiseadd.o
```

<hr style="border:0;height:1px;background:#ffb600;margin:32px 0;">
