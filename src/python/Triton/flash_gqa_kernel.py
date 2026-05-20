# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-01
# Purpose: Triton implementation for Flash GQA forward kernel.
# ==============================================================================

import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_gqa_forward_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    q_heads,
    seqlen_q,
    seqlen_kv,
    scale,
    causal: tl.constexpr,
    block_g: tl.constexpr,
    block_q: tl.constexpr,
    block_hq: tl.constexpr,
    block_kv_sub: tl.constexpr,
    head_dim_qk: tl.constexpr,
    head_dim_v: tl.constexpr,
    block_dim_qk: tl.constexpr,
    block_dim_v: tl.constexpr,
    group_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    blocks_per_kv: tl.constexpr,
):
    """
    Prototype Flash GQA forward kernel (bf16 inputs, fp32 accumulation).

    Main feature:
        Reuses K/V blocks across a small group of query heads per KV head.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_v]
        lse_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        q_heads: int32 scalar, number of query heads
        seqlen_q: int32 scalar, query sequence length
        seqlen_kv: int32 scalar, key/value sequence length
        scale: float32 scalar, attention scaling factor
        causal: constexpr bool, apply causal mask if True
        block_g: constexpr int32 scalar, query heads per kv head in this program
        block_q: constexpr int32 scalar, query tokens per block
        block_hq: constexpr int32 scalar, query head tokens per block (block_g * block_q)
        block_kv_sub: constexpr int32 scalar, kv sub-block per iteration
        head_dim_qk: constexpr int32 scalar, head dimension for Q/K
        head_dim_v: constexpr int32 scalar, head dimension for V/output
        block_dim_qk: constexpr int32 scalar, power-of-two block for head_dim_qk
        block_dim_v: constexpr int32 scalar, power-of-two block for head_dim_v
        group_size: constexpr int32 scalar, query heads per kv head
        num_kv_heads: constexpr int32 scalar, number of kv heads
        blocks_per_kv: constexpr int32 scalar, head blocks per kv head

    Outputs:
        None
    """
    pid_g = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    d_offsets_qk = tl.arange(0, block_dim_qk)
    d_offsets_v = tl.arange(0, block_dim_v)
    dim_mask_qk = d_offsets_qk < head_dim_qk
    dim_mask_v = d_offsets_v < head_dim_v

    kv_head = pid_g // blocks_per_kv
    head_block = pid_g - kv_head * blocks_per_kv
    head_start = kv_head * group_size + head_block * block_g

    hq_offsets = tl.arange(0, block_hq)
    head_offsets = hq_offsets // block_q
    q_offsets = pid_q * block_q + (hq_offsets - head_offsets * block_q)
    q_mask = q_offsets < seqlen_q
    head_idx = head_start + head_offsets

    q_base = (pid_b * q_heads + head_idx) * seqlen_q * head_dim_qk
    q_ptrs = q_ptr + q_base[:, None] + q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
    q_block = tl.load(q_ptrs, mask=q_mask[:, None] & dim_mask_qk[None, :], other=0.0)

    rowmax = tl.full((block_hq,), -float("inf"), dtype=tl.float32)
    rowsum = tl.zeros((block_hq,), dtype=tl.float32)
    o_head = tl.zeros((block_hq, block_dim_v), dtype=tl.float32)

    k_base = (pid_b * num_kv_heads + kv_head) * seqlen_kv * head_dim_qk
    v_base = (pid_b * num_kv_heads + kv_head) * seqlen_kv * head_dim_v

    # flash loop over kv tokens (clamp causal range to actual KV length)
    seq_kv_end = seqlen_kv
    if causal:
        seq_kv_end = min((pid_q + 1) * block_q, seqlen_kv)
    for kv_start in range(0, seq_kv_end, block_kv_sub):
        kv_offsets = kv_start + tl.arange(0, block_kv_sub)
        kv_mask = kv_offsets < seqlen_kv
        k_ptrs = k_ptr + k_base + kv_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
        v_ptrs = v_ptr + v_base + kv_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None] & dim_mask_qk[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None] & dim_mask_v[None, :], other=0.0)

        if causal:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            attn_mask = kv_mask[None, :] & causal_mask
        else:
            attn_mask = kv_mask[None, :]

        s = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32) * scale
        s = tl.where(attn_mask, s, -float("inf"))

        new_rowmax = tl.maximum(rowmax, tl.max(s, axis=1))
        p = tl.exp(s - new_rowmax[:, None])
        p_bf16 = tl.cast(p, tl.bfloat16)
        alpha = tl.exp(rowmax - new_rowmax)
        o_head = tl.dot(p_bf16, v_block, acc=o_head * alpha[:, None], out_dtype=tl.float32)

        rowmax = new_rowmax
        rowsum = tl.sum(p, axis=1) + rowsum * alpha

    o_head /= rowsum[:, None]
    lse_head = rowmax + tl.log(rowsum)

    o_base = (pid_b * q_heads + head_idx) * seqlen_q * head_dim_v
    o_ptrs = o_ptr + o_base[:, None] + q_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
    lse_ptrs = lse_ptr + (pid_b * q_heads + head_idx) * seqlen_q + q_offsets
    tl.store(o_ptrs, o_head.to(tl.bfloat16), mask=q_mask[:, None] & dim_mask_v[None, :])
    tl.store(lse_ptrs, lse_head, mask=q_mask)


def flash_gqa_forward_grid(meta, q_heads, batch, seqlen_q, num_kv_heads):
    """
    Compute the grid for the Flash GQA forward kernel.

    Main feature:
        Computes a 3D grid over head blocks per kv head, query blocks, and batch.

    Inputs:
        meta: dict with keys "block_g" and "block_q"
        q_heads: int32 scalar, number of query heads
        batch: int32 scalar, batch size
        seqlen_q: int32 scalar, query sequence length
        num_kv_heads: int32 scalar, number of kv heads

    Outputs:
        grid: tuple[int32, int32, int32] with shape [3]
    """
    group_size = q_heads // num_kv_heads
    blocks_per_kv = group_size // meta["block_g"]
    return (
        num_kv_heads * blocks_per_kv,
        triton.cdiv(seqlen_q, meta["block_q"]),
        batch,
    )


def _next_power_of_2(value):
    """
    Compute the next power-of-two >= value.

    Main feature:
        Returns a power-of-two integer for Triton block sizing.

    Inputs:
        value: int32 scalar, positive integer

    Outputs:
        pow2_value: int32 scalar, smallest power-of-two >= value
    """
    if value < 1:
        raise ValueError("value must be >= 1")
    return 1 << (value - 1).bit_length()


def launch_flash_gqa_forward_bf16_kernel(
    q,
    k,
    v,
    scale=None,
    out=None,
    lse=None,
    causal=True,
    block_kv_sub=32,
    num_warps=8,
    num_stages=3,
):
    """
    Launch the prototype Flash GQA forward kernel.

    Main feature:
        Validates inputs, allocates outputs, and dispatches Triton.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        scale: optional float scalar, attention scaling factor
        out: optional torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
             [batch, q_heads, seqlen_q, head_dim_v]
        lse: optional torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q]
        causal: bool scalar, apply causal mask if True
        block_kv_sub: int32 scalar, kv sub-block per flash GQA iteration
        num_warps: int32 scalar, Triton num_warps launch hint
        num_stages: int32 scalar, Triton num_stages launch hint

    Outputs:
        out: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
             [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q]
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("Flash GQA forward expects bf16 inputs")
    if q.dim() not in (3, 4):
        raise ValueError("Q must be 3D or 4D")
    if k.dim() != q.dim() or v.dim() != q.dim():
        raise ValueError("Q, K, and V must have matching ranks")

    squeeze_output = q.dim() == 3
    if squeeze_output:
        q_batched = q.contiguous().unsqueeze(0)
        k_batched = k.contiguous().unsqueeze(0)
        v_batched = v.contiguous().unsqueeze(0)
    else:
        q_batched = q.contiguous()
        k_batched = k.contiguous()
        v_batched = v.contiguous()

    batch, q_heads, seqlen_q, head_dim_qk = q_batched.shape
    _, num_kv_heads, seqlen_kv, k_head_dim = k_batched.shape
    _, v_heads, v_seqlen, head_dim_v = v_batched.shape
    if seqlen_kv != v_seqlen:
        raise ValueError("K and V must share sequence length")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    if v_heads != num_kv_heads:
        raise ValueError("K and V must share number of KV heads")
    if q_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")

    group_size = q_heads // num_kv_heads
    if group_size >= 8 and group_size % 4 == 0:
        block_g = 4
    elif group_size >= 4 and group_size % 2 == 0:
        block_g = 2
    else:
        block_g = 1
    blocks_per_kv = group_size // block_g
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)

    block_q = 32
    block_hq = block_g * block_q
    block_dim_qk = _next_power_of_2(head_dim_qk)
    block_dim_v = _next_power_of_2(head_dim_v)
    if out is None:
        out_batched = torch.empty(
            (batch, q_heads, seqlen_q, head_dim_v),
            device=q_batched.device,
            dtype=q_batched.dtype,
        )
    else:
        if squeeze_output:
            if out.dim() != 3:
                raise ValueError("Output must be 3D when Q is 3D")
            out_batched = out.contiguous().unsqueeze(0)
        else:
            if out.dim() != 4:
                raise ValueError("Output must be 4D when Q is 4D")
            out_batched = out.contiguous()
    if out_batched.shape != (batch, q_heads, seqlen_q, head_dim_v):
        raise ValueError("Output must have shape [batch, q_heads, seqlen_q, head_dim_v]")
    if lse is None:
        lse_batched = torch.empty((batch, q_heads, seqlen_q), device=q_batched.device, dtype=torch.float32)
    else:
        if squeeze_output:
            if lse.dim() != 2:
                raise ValueError("LSE must be 2D when Q is 3D")
            lse_batched = lse.contiguous().unsqueeze(0)
        else:
            if lse.dim() != 3:
                raise ValueError("LSE must be 3D when Q is 4D")
            lse_batched = lse.contiguous()

    grid = functools.partial(
        flash_gqa_forward_grid,
        q_heads=q_heads,
        batch=batch,
        seqlen_q=seqlen_q,
        num_kv_heads=num_kv_heads,
    )

    launch_num_stages = num_stages
    if block_dim_qk > 128 or block_dim_v > 128:
        launch_num_stages = min(num_stages, 2)

    flash_gqa_forward_bf16_kernel[grid](
        q_batched,
        k_batched,
        v_batched,
        out_batched,
        lse_batched,
        q_heads,
        seqlen_q,
        seqlen_kv,
        scale,
        causal=causal,
        block_g=block_g,
        block_q=block_q,
        block_hq=block_hq,
        block_kv_sub=block_kv_sub,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        block_dim_qk=block_dim_qk,
        block_dim_v=block_dim_v,
        group_size=group_size,
        num_kv_heads=num_kv_heads,
        blocks_per_kv=blocks_per_kv,
        num_warps=num_warps,
        num_stages=launch_num_stages,
    )
    if squeeze_output:
        return out_batched.squeeze(0), lse_batched.squeeze(0)
    return out_batched, lse_batched
