# ⚡️ CUDA Operators for Large Language Models

## Project Overview

This repositary is a high-performance C++/CUDA library implementing **optimized kernels** for critical operators used in **Large Language Models (LLMs)**.

***

## Operators

The following core LLM operators have custom CUDA kernel implementations in this repository.

All experiements were run on a Acer Shadow SH16-73 Laptop with Nvidia Geforce RTX 5080 Laptop GPU (16GB GDDR7, Global Memory Bandwidth 890 GB/s).

### ElementwiseAdd

**Performance**

| Kernel | Input Shape | Input Type | GPU Time (us)| GPU Memory Bandwidth (TB/s)|
| :--- | :--- | :--- |:--- |:--- |
|cublasSgeam|`(4096,4096)`|fp32| 319.488 | 0.573122 |
|elementwiseadd_fp32_kernel|`(4096,4096)`|fp32| 293.92 | 0.622977 |

### LayerNorm
**Description**

Computes LayerNorm along the hidden dimension.

**Performance**

| Kernel | Input Shape | Input Type |GPU Time (us)| GPU Memory Bandwidth (TB/s)|
| :--- | :--- | :--- |:--- |:--- |
|layernorm_fp32_kernel|`(4096,7168)`|fp32|  368.64 | 0.57949 |

### ReduceSum
**Description**

Computes rowsum of a 2D matrix.

**Performance**

| Kernel | Input Shape | Input Type |Output Type| GPU Time (us)| GPU Memory Bandwidth (TB/s)|
| :--- | :--- | :--- |:--- |:--- |:--- |
|cublasSgemv|`(4096,4096)`|fp32|`(4096,)`| 126.08 | 0.484217 |
|elementwiseadd_fp32_kernel|`(4096,4096)`|fp32|`(4096,)`| 98.304  | 0.621033 |

### RMSNorm
**Description**

Computes RMSNorm along the hidden dimension.

**Performance**

| Kernel | Input Shape | Input Type |GPU Time (us)| GPU Memory Bandwidth (TB/s)|
| :--- | :--- | :--- |:--- |:--- |
|rmsnorm_fp32_kernel|`(4096,7168)`|fp32|  370.688 | 0.576288 |

### Softmax
**Description**

Online safe softmax.

**Performance**

| Kernel | Input Shape | Input Type |GPU Time (us)| GPU Memory Bandwidth (TB/s)|
| :--- | :--- | :--- |:--- |:--- |
|softmax_fp32_kernel|`(128,16384)`|fp32|  20.096 | 0.759295 |
|softmax_fp32_kernel|`(32, 131072)`|fp32|   61.44 | 0.496705 |

### TopK
**Description**

Radix Select + Blelloch Scan.

**Performance**

| Kernel | Input Shape | Input Type |GPU Time (us)|
| :--- | :--- | :--- |:--- |
|radixSelectTopK_fp32_kernel|`(4, 131072)`|fp32| 65.05   |


***

## Compilation and Running