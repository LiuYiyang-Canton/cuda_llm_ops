# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Triton Flash GQA decoding kernel and launch wrapper.
# ==============================================================================
"""Triton Flash GQA decoding kernel and launch wrapper."""

import functools
import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_g": 1, "block_kv": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_g": 1, "block_kv": 128}, num_warps=4, num_stages=3),
        triton.Config({"block_g": 2, "block_kv": 64}, num_warps=8, num_stages=2),
        triton.Config({"block_g": 2, "block_kv": 128}, num_warps=8, num_stages=3),
        triton.Config({"block_g": 4, "block_kv": 64}, num_warps=8, num_stages=2),
        triton.Config({"block_g": 4, "block_kv": 128}, num_warps=8, num_stages=3),
    ],
    key=["q_heads"],
)
@triton.jit
def decode_flash_gqa_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_heads,
    seqlen_q,
    seqlen_kv,
    scale,
    block_g: tl.constexpr,
    block_kv: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    """
    Compute grouped query attention for decoding on bf16 inputs with fp32 accumulation.

    Main feature:
        Computes streaming softmax attention for grouped query heads during decoding.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim]
        o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim]
        q_heads: int32 scalar, number of query heads
        seqlen_q: int32 scalar, query sequence length
        seqlen_kv: int32 scalar, key/value sequence length
        scale: float32 scalar, attention scaling factor
        block_g: constexpr int32 scalar, query-head groups per block
        block_kv: constexpr int32 scalar, kv tokens per iteration
        head_dim: constexpr int32 scalar, head dimension
        group_size: constexpr int32 scalar, query heads per kv head
        num_kv_heads: constexpr int32 scalar, number of kv heads

    Outputs:
        None
    """
    pid_g = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    for g in range(0, block_g):
        head_offsets = pid_g * block_g * group_size + g * group_size + tl.arange(0, group_size)
        head_mask = head_offsets < q_heads
        kv_offsets = tl.arange(0, block_kv)
        d_offsets = tl.arange(0, head_dim)

        q_base = (pid_b * q_heads + head_offsets) * seqlen_q * head_dim
        q_ptrs = q_ptr + q_base[:, None] + d_offsets[None, :]
        q_vals = tl.load(q_ptrs, mask=head_mask[:, None], other=0.0)

        row_max = tl.full((group_size,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((group_size,), dtype=tl.float32)
        acc = tl.zeros((group_size, head_dim), dtype=tl.float32)

        kv_head = pid_g * block_g + g
        k_base = (pid_b * num_kv_heads + kv_head) * seqlen_kv * head_dim
        for start_kv in range(0, seqlen_kv, block_kv):
            kv_idx = start_kv + kv_offsets
            kv_mask = kv_idx < seqlen_kv

            kv_offsets_2d = kv_idx * head_dim
            k_ptrs = k_ptr + (k_base + kv_offsets_2d[:, None] + d_offsets[None, :])
            v_ptrs = v_ptr + (k_base + kv_offsets_2d[:, None] + d_offsets[None, :])

            k_vals = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
            v_vals = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            qk_scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32) * scale
            qk_scores = tl.where(head_mask[:, None] & kv_mask[None, :], qk_scores, -float("inf"))

            row_max_new = tl.maximum(row_max, tl.max(qk_scores, axis=1))
            p = tl.exp(qk_scores - row_max_new[:, None])
            alpha = tl.exp(row_max - row_max_new)

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            row_max = row_max_new

        row_sum_safe = tl.where(row_sum > 0, row_sum, 1.0)
        o_vals = acc / row_sum_safe[:, None]

        o_base = (pid_b * q_heads + head_offsets) * seqlen_q * head_dim
        o_ptrs = o_ptr + o_base[:, None] + d_offsets[None, :]
        tl.store(o_ptrs, o_vals.to(tl.bfloat16), mask=head_mask[:, None])


def decode_flash_gqa_bf16_grid(meta, num_kv_heads, batch):
    """
    Compute the Triton grid for Flash GQA decoding.

    Main feature:
        Computes a 2D grid over KV head tiles and batch.

    Inputs:
        meta: dict with key "block_g" (int32)
        num_kv_heads: int32 scalar, number of kv heads
        batch: int32 scalar, batch size

    Outputs:
        grid: tuple[int32, int32] with shape [2]
    """
    return (
        triton.cdiv(num_kv_heads, meta["block_g"]),
        batch,
    )


def launch_decode_flash_gqa_bf16_kernel(q, k, v, out=None):
    """
    Launch the Triton Flash GQA decoding kernel on bf16 tensors.

    Main feature:
        Validates inputs, computes launch parameters, and dispatches Triton.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        out: optional torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]

    Outputs:
        out: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("Flash GQA decoding expects bf16 inputs")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Flash GQA decoding kernel expects contiguous Q/K/V tensors")

    batch, q_heads, seqlen_q, head_dim = q.shape
    k_batch, num_kv_heads, seqlen_kv, k_head_dim = k.shape
    v_batch, v_heads, v_seqlen, v_head_dim = v.shape

    if batch != k_batch or batch != v_batch:
        raise ValueError("Q, K, and V must share batch dimension")
    if seqlen_kv != v_seqlen:
        raise ValueError("K and V must share sequence length")
    if k_head_dim != v_head_dim or head_dim != k_head_dim:
        raise ValueError("Q, K, and V must share head_dim")
    if v_heads != num_kv_heads:
        raise ValueError("K and V must share number of KV heads")
    if q_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")

    group_size = q_heads // num_kv_heads

    if out is None:
        out = torch.empty_like(q)
    if not out.is_contiguous():
        raise ValueError("Output tensor must be contiguous")

    scale = 1.0 / math.sqrt(head_dim)
    grid = functools.partial(decode_flash_gqa_bf16_grid, num_kv_heads=num_kv_heads, batch=batch)

    decode_flash_gqa_bf16_kernel[grid](
        q,
        k,
        v,
        out,
        q_heads,
        seqlen_q,
        seqlen_kv,
        scale,
        head_dim=head_dim,
        group_size=group_size,
        num_kv_heads=num_kv_heads,
    )
    return out
