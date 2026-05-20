# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-06
# Purpose: Triton implementation for Flash Sparse GQA forward kernel.
# ==============================================================================

import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_sgqa_forward_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    sparse_kv_indices_ptr,
    o_ptr,
    lse_ptr,
    q_heads,
    seqlen_q,
    seqlen_kv,
    scale,
    causal: tl.constexpr,
    block_q: tl.constexpr,
    block_sparse_sub: tl.constexpr,
    head_dim_qk: tl.constexpr,
    head_dim_v: tl.constexpr,
    block_dim_qk: tl.constexpr,
    block_dim_v: tl.constexpr,
    group_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_sparse_kv: tl.constexpr,
):
    """
    Compute sparse grouped-query attention for one KV head group per program.

    Main feature:
        Each program is keyed by one KV head and one query block, and reuses sparse KV indices
        across all query heads in that KV-head group.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        sparse_kv_indices_ptr: pointer to int32 with shape [batch, num_kv_heads, seqlen_q, num_sparse_kv]
        o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_v]
        lse_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        q_heads: int32 scalar, number of query heads
        seqlen_q: int32 scalar, query sequence length
        seqlen_kv: int32 scalar, key/value sequence length
        scale: float32 scalar, attention scaling factor
        causal: constexpr bool, apply causal mask if True
        block_q: constexpr int32 scalar, query tokens per block
        block_sparse_sub: constexpr int32 scalar, sparse kv entries per sub-iteration
        head_dim_qk: constexpr int32 scalar, head dimension for Q/K
        head_dim_v: constexpr int32 scalar, head dimension for V/output
        block_dim_qk: constexpr int32 scalar, power-of-two block for head_dim_qk
        block_dim_v: constexpr int32 scalar, power-of-two block for head_dim_v
        group_size: constexpr int32 scalar, query heads per KV head
        num_kv_heads: constexpr int32 scalar, number of KV heads
        num_sparse_kv: constexpr int32 scalar, sparse KV list length

    Outputs:
        None
    """
    pid_kv = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    head_offsets = tl.arange(0, group_size)
    head_idx = pid_kv * group_size + head_offsets

    d_offsets_qk = tl.arange(0, block_dim_qk)
    d_offsets_v = tl.arange(0, block_dim_v)
    dim_mask_qk = d_offsets_qk < head_dim_qk
    dim_mask_v = d_offsets_v < head_dim_v

    k_base = (pid_b * num_kv_heads + pid_kv) * seqlen_kv * head_dim_qk
    v_base = (pid_b * num_kv_heads + pid_kv) * seqlen_kv * head_dim_v

    q_pos = pid_q * block_q
    q_valid = q_pos < seqlen_q

    q_base = (pid_b * q_heads + head_idx) * seqlen_q * head_dim_qk
    q_ptrs = q_ptr + q_base[:, None] + q_pos * head_dim_qk + d_offsets_qk[None, :]
    q_group = tl.load(q_ptrs, mask=q_valid & dim_mask_qk[None, :], other=0.0)

    rowmax = tl.full((group_size,), -1.0e20, dtype=tl.float32)
    rowsum = tl.zeros((group_size,), dtype=tl.float32)
    o_group = tl.zeros((group_size, block_dim_v), dtype=tl.float32)

    sparse_base = ((pid_b * num_kv_heads + pid_kv) * seqlen_q + q_pos) * num_sparse_kv
    for sparse_start in range(0, num_sparse_kv, block_sparse_sub):
        sparse_offsets = sparse_start + tl.arange(0, block_sparse_sub)
        sparse_mask = sparse_offsets < num_sparse_kv
        sparse_ptrs = sparse_kv_indices_ptr + sparse_base + sparse_offsets
        kv_idx = tl.load(sparse_ptrs, mask=q_valid & sparse_mask, other=-1)

        valid_mask = q_valid & sparse_mask
        valid_mask = valid_mask & (kv_idx >= 0) & (kv_idx < seqlen_kv)
        if causal:
            valid_mask = valid_mask & (kv_idx <= q_pos)

        k_ptrs = k_ptr + k_base + kv_idx[:, None] * head_dim_qk + d_offsets_qk[None, :]
        v_ptrs = v_ptr + v_base + kv_idx[:, None] * head_dim_v + d_offsets_v[None, :]
        k_block = tl.load(k_ptrs, mask=valid_mask[:, None] & dim_mask_qk[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=valid_mask[:, None] & dim_mask_v[None, :], other=0.0)

        scores = tl.dot(q_group, tl.trans(k_block), out_dtype=tl.float32) * scale
        scores = tl.where(valid_mask[None, :], scores, -1.0e20)

        new_rowmax = tl.maximum(rowmax, tl.max(scores, axis=1))
        p = tl.exp(scores - new_rowmax[:, None])
        p = tl.where(valid_mask[None, :], p, 0.0)
        alpha = tl.exp(rowmax - new_rowmax)
        o_group = tl.dot(
            tl.cast(p, tl.bfloat16),
            v_block,
            acc=o_group * alpha[:, None],
            out_dtype=tl.float32,
        )
        rowmax = new_rowmax
        rowsum = tl.sum(p, axis=1) + rowsum * alpha

    denom = tl.where(rowsum > 0, rowsum, 1.0)
    o_group = o_group / denom[:, None]
    lse_group = tl.where(rowsum > 0, rowmax + tl.log(rowsum), -float("inf"))

    o_base = (pid_b * q_heads + head_idx) * seqlen_q * head_dim_v
    o_ptrs = o_ptr + o_base[:, None] + q_pos * head_dim_v + d_offsets_v[None, :]
    lse_ptrs = lse_ptr + (pid_b * q_heads + head_idx) * seqlen_q + q_pos

    tl.store(o_ptrs, o_group.to(tl.bfloat16), mask=q_valid & dim_mask_v[None, :])
    tl.store(lse_ptrs, lse_group, mask=q_valid)


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


def _select_sgqa_forward_launch_config(
    seqlen_q,
    q_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    num_sparse_kv,
):
    """
    Select SGQA forward launch configuration from shape heuristics.

    Main feature:
        Returns deterministic launch parameters for sparse-subtile size, warps, and stages.

    Inputs:
        seqlen_q: int32 scalar, query sequence length
        q_heads: int32 scalar, number of query heads
        num_kv_heads: int32 scalar, number of KV heads
        head_dim_qk: int32 scalar, Q/K head dimension
        head_dim_v: int32 scalar, V/output head dimension
        num_sparse_kv: int32 scalar, sparse KV list length

    Outputs:
        block_sparse_sub: int32 scalar, sparse indices processed per sub-iteration
        num_warps: int32 scalar, Triton num_warps launch hint
        num_stages: int32 scalar, Triton num_stages launch hint
    """
    if (
        head_dim_qk == 128
        and head_dim_v == 128
        and num_kv_heads == 1
        and q_heads <= 16
        and num_sparse_kv >= 256
    ):
        return 16, 1, 1
    if head_dim_qk <= 64 or head_dim_v <= 64:
        return 32, 2, 1
    if num_kv_heads > 1 or q_heads >= 32:
        return 64, 4, 1
    if seqlen_q >= 8192:
        return 64, 4, 1
    return 16, 2, 2


def launch_flash_sgqa_forward_bf16_kernel(
    q,
    k,
    v,
    sparse_kv_indices,
    num_sparse_kv,
    scale=None,
    out=None,
    lse=None,
    causal=True,
    block_sparse_sub=None,
    num_warps=None,
    num_stages=None,
):
    """
    Launch the Flash SGQA forward kernel.

    Main feature:
        Validates inputs, normalizes batch layout, and dispatches Triton sparse GQA forward.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, num_sparse_kv]
        num_sparse_kv: int32 scalar, sparse KV list length
        scale: optional float scalar, attention scaling factor
        out: optional torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
             [batch, q_heads, seqlen_q, head_dim_v], must be contiguous when provided
        lse: optional torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q], must be contiguous when provided
        causal: bool scalar, apply causal mask if True
        block_sparse_sub: optional int32 scalar, sparse indices processed per sub-iteration
        num_warps: optional int32 scalar, Triton num_warps launch hint
        num_stages: optional int32 scalar, Triton num_stages launch hint

    Outputs:
        out: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
             [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q]
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("Flash SGQA forward expects bf16 inputs")
    if sparse_kv_indices.dtype != torch.int32:
        raise ValueError("sparse_kv_indices must be int32")
    if sparse_kv_indices.dim() != 4:
        raise ValueError("sparse_kv_indices must be 4D [batch, num_kv_heads, seqlen_q, num_sparse_kv]")
    if not isinstance(num_sparse_kv, int):
        raise ValueError("num_sparse_kv must be an int")
    if num_sparse_kv < 1:
        raise ValueError("num_sparse_kv must be >= 1")
    if block_sparse_sub is not None and (not isinstance(block_sparse_sub, int) or block_sparse_sub < 1):
        raise ValueError("block_sparse_sub must be None or an int >= 1")
    if num_warps is not None and (not isinstance(num_warps, int) or num_warps < 1):
        raise ValueError("num_warps must be None or an int >= 1")
    if num_stages is not None and (not isinstance(num_stages, int) or num_stages < 1):
        raise ValueError("num_stages must be None or an int >= 1")
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

    expected_sparse_shape = (batch, num_kv_heads, seqlen_q, num_sparse_kv)
    if sparse_kv_indices.shape != expected_sparse_shape:
        raise ValueError(
            "sparse_kv_indices must have shape "
            f"[batch, num_kv_heads, seqlen_q, {num_sparse_kv}]"
        )
    sparse_kv_indices_batched = sparse_kv_indices.contiguous()

    group_size = q_heads // num_kv_heads
    block_q = 1

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)

    (
        auto_block_sparse_sub,
        auto_num_warps,
        auto_num_stages,
    ) = _select_sgqa_forward_launch_config(
        seqlen_q=seqlen_q,
        q_heads=q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        num_sparse_kv=num_sparse_kv,
    )
    effective_block_sparse_sub = block_sparse_sub if block_sparse_sub is not None else auto_block_sparse_sub
    effective_num_warps = num_warps if num_warps is not None else auto_num_warps
    effective_num_stages = num_stages if num_stages is not None else auto_num_stages

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
            if not out.is_contiguous():
                raise ValueError("Output must be contiguous when provided")
            out_batched = out.unsqueeze(0)
        else:
            if out.dim() != 4:
                raise ValueError("Output must be 4D when Q is 4D")
            if not out.is_contiguous():
                raise ValueError("Output must be contiguous when provided")
            out_batched = out
    if out_batched.shape != (batch, q_heads, seqlen_q, head_dim_v):
        raise ValueError("Output must have shape [batch, q_heads, seqlen_q, head_dim_v]")

    if lse is None:
        lse_batched = torch.empty((batch, q_heads, seqlen_q), device=q_batched.device, dtype=torch.float32)
    else:
        if squeeze_output:
            if lse.dim() != 2:
                raise ValueError("LSE must be 2D when Q is 3D")
            if not lse.is_contiguous():
                raise ValueError("LSE must be contiguous when provided")
            lse_batched = lse.unsqueeze(0)
        else:
            if lse.dim() != 3:
                raise ValueError("LSE must be 3D when Q is 4D")
            if not lse.is_contiguous():
                raise ValueError("LSE must be contiguous when provided")
            lse_batched = lse
    if lse_batched.shape != (batch, q_heads, seqlen_q):
        raise ValueError("LSE must have shape [batch, q_heads, seqlen_q]")

    launch_num_stages = effective_num_stages
    if block_dim_qk > 128 or block_dim_v > 128:
        launch_num_stages = min(effective_num_stages, 2)

    if seqlen_q > 0:
        grid = (num_kv_heads, triton.cdiv(seqlen_q, block_q), batch)
        flash_sgqa_forward_bf16_kernel[grid](
            q_batched,
            k_batched,
            v_batched,
            sparse_kv_indices_batched,
            out_batched,
            lse_batched,
            q_heads,
            seqlen_q,
            seqlen_kv,
            scale,
            causal=causal,
            block_q=block_q,
            block_sparse_sub=effective_block_sparse_sub,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            block_dim_qk=block_dim_qk,
            block_dim_v=block_dim_v,
            group_size=group_size,
            num_kv_heads=num_kv_heads,
            num_sparse_kv=num_sparse_kv,
            num_warps=effective_num_warps,
            num_stages=launch_num_stages,
        )

    if squeeze_output:
        return out_batched.squeeze(0), lse_batched.squeeze(0)
    return out_batched, lse_batched
