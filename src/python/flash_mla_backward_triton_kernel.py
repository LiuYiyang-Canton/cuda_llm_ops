# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-01
# Purpose: Triton implementation for Flash MLA backward kernel.
# ==============================================================================

import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def rowwise_dot_o_grad_o_kernel(
    o_ptr,
    grad_o_ptr,
    out_ptr,
    head_dim_v: tl.constexpr,
    head_block: tl.constexpr,
):
    """
    Compute the dot product between O and grad_o for every query row.

    Main feature:
        Loads bf16 O and grad_o blocks and accumulates their dot product in fp32.

    Inputs:
        o_ptr: pointer to bf16 tensor of shape [rows, head_dim_v]
        grad_o_ptr: pointer to bf16 tensor of shape [rows, head_dim_v]
        out_ptr: pointer to float32 tensor of shape [rows]
        head_dim_v: constexpr int, size of the value/output head dimension
        head_block: constexpr int, number of head elements processed per iteration

    Outputs:
        None. The computed dot products are stored in out_ptr.
    """
    row_id = tl.program_id(axis=0)
    row_offset = row_id * head_dim_v
    accum = tl.zeros((), dtype=tl.float32)
    for head_start in range(0, head_dim_v, head_block):
        head_indices = head_start + tl.arange(0, head_block)
        mask = head_indices < head_dim_v
        o_slice = tl.load(o_ptr + row_offset + head_indices, mask=mask, other=0.0).to(tl.float32)
        grad_slice = tl.load(grad_o_ptr + row_offset + head_indices, mask=mask, other=0.0).to(tl.float32)
        accum += tl.sum(o_slice * grad_slice)
    tl.store(out_ptr + row_id, accum)


def compute_rowwise_dot_o_grad_o(o_tensor, grad_o_tensor):
    """
    Launch the Triton kernel that computes O @ grad_o per row.

    Main feature:
        Flattens the batched tensors into rows and reshapes the result after the kernel finishes.

    Inputs:
        o_tensor: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]
        grad_o_tensor: torch.Tensor bf16 matching o_tensor shape

    Outputs:
        torch.Tensor float32 of shape [batch, q_heads, seqlen_q] containing row-wise dots.
    """
    o_tensor = o_tensor.contiguous()
    grad_o_tensor = grad_o_tensor.contiguous()
    batch, q_heads, seqlen_q, head_dim_v = o_tensor.shape
    rows = batch * q_heads * seqlen_q
    rowwise_flat = torch.empty(rows, device=o_tensor.device, dtype=torch.float32)
    if rows > 0:
        grid = (rows,)
        rowwise_dot_o_grad_o_kernel[grid](
            o_tensor.view(-1),
            grad_o_tensor.view(-1),
            rowwise_flat,
            head_dim_v=head_dim_v,
            head_block=128,
        )
    return rowwise_flat.view(batch, q_heads, seqlen_q)


@triton.jit
def float32_to_bf16_multi_kernel(
    input0_ptr,
    input1_ptr,
    input2_ptr,
    output0_ptr,
    output1_ptr,
    output2_ptr,
    size0,
    size1,
    size2,
    block_size: tl.constexpr,
):
    """
    Cast three float32 arrays to bfloat16 in a single kernel launch.

    Main feature:
        Maps a single linear index space onto three tensors and casts each slice to bf16.

    Inputs:
        input0_ptr: pointer to float32 tensor of shape [size0]
        input1_ptr: pointer to float32 tensor of shape [size1]
        input2_ptr: pointer to float32 tensor of shape [size2]
        output0_ptr: pointer to bf16 tensor of shape [size0]
        output1_ptr: pointer to bf16 tensor of shape [size1]
        output2_ptr: pointer to bf16 tensor of shape [size2]
        size0: int32 scalar, number of elements for tensor 0
        size1: int32 scalar, number of elements for tensor 1
        size2: int32 scalar, number of elements for tensor 2
        block_size: constexpr int defining the block width

    Outputs:
        None. The bf16 results are written into output pointers.
    """
    block_start = tl.program_id(axis=0) * block_size
    offsets = block_start + tl.arange(0, block_size)
    total = size0 + size1 + size2
    mask = offsets < total

    mask0 = offsets < size0
    mask1 = (offsets >= size0) & (offsets < size0 + size1)
    mask2 = offsets >= size0 + size1

    offsets0 = offsets
    offsets1 = offsets - size0
    offsets2 = offsets - size0 - size1

    values0 = tl.load(input0_ptr + offsets0, mask=mask & mask0, other=0.0)
    values1 = tl.load(input1_ptr + offsets1, mask=mask & mask1, other=0.0)
    values2 = tl.load(input2_ptr + offsets2, mask=mask & mask2, other=0.0)

    tl.store(output0_ptr + offsets0, values0.to(tl.bfloat16), mask=mask & mask0)
    tl.store(output1_ptr + offsets1, values1.to(tl.bfloat16), mask=mask & mask1)
    tl.store(output2_ptr + offsets2, values2.to(tl.bfloat16), mask=mask & mask2)


def cast_float32_tensor_to_bf16(float_tensor0, float_tensor1, float_tensor2):
    """
    Convert three contiguous float32 tensors to bf16 via a single Triton kernel.

    Main feature:
        Uses one block-wise Triton kernel to cast three tensors in a single launch.

    Inputs:
        float_tensor0: torch.Tensor float32 of arbitrary shape
        float_tensor1: torch.Tensor float32 of arbitrary shape
        float_tensor2: torch.Tensor float32 of arbitrary shape

    Outputs:
        output0: torch.Tensor bf16 with the same shape and device as float_tensor0
        output1: torch.Tensor bf16 with the same shape and device as float_tensor1
        output2: torch.Tensor bf16 with the same shape and device as float_tensor2
    """
    float_tensor0 = float_tensor0.contiguous()
    float_tensor1 = float_tensor1.contiguous()
    float_tensor2 = float_tensor2.contiguous()
    output0 = torch.empty_like(float_tensor0, dtype=torch.bfloat16)
    output1 = torch.empty_like(float_tensor1, dtype=torch.bfloat16)
    output2 = torch.empty_like(float_tensor2, dtype=torch.bfloat16)
    size0 = float_tensor0.numel()
    size1 = float_tensor1.numel()
    size2 = float_tensor2.numel()
    total = size0 + size1 + size2
    if total == 0:
        return output0, output1, output2
    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    float32_to_bf16_multi_kernel[grid](
        float_tensor0,
        float_tensor1,
        float_tensor2,
        output0,
        output1,
        output2,
        size0,
        size1,
        size2,
        block_size=block_size,
    )
    return output0, output1, output2


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


@triton.jit
def flash_mla_backward_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    grad_o_ptr,
    rowwise_dot_o_grad_o_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    q_heads,
    seqlen_q,
    seqlen_kv,
    scale,
    causal: tl.constexpr,
    block_q_sub: tl.constexpr,
    block_kv: tl.constexpr,
    head_dim_qk: tl.constexpr,
    head_dim_v: tl.constexpr,
    block_dim_qk: tl.constexpr,
    block_dim_v: tl.constexpr,
    group_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    use_q_mask: tl.constexpr,
    use_kv_mask: tl.constexpr,
    fast_causal: tl.constexpr,
):
    """
    Prototype Flash MLA backward kernel (bf16 inputs, fp32 accumulation).

    Main feature:
        Processes a single query head per program and reuses K/V blocks per KV head.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_v]
        grad_o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_v]
        rowwise_dot_o_grad_o_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        grad_q_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k_ptr: pointer to float32 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v_ptr: pointer to float32 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        lse_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        q_heads: int32 scalar, number of query heads
        seqlen_q: int32 scalar, query sequence length
        seqlen_kv: int32 scalar, key/value sequence length
        scale: float32 scalar, attention scaling factor
        causal: constexpr bool, apply causal mask if True
        block_q_sub: constexpr int32 scalar, query sub-block per iteration
        block_kv: constexpr int32 scalar, kv sub-block per iteration
        head_dim_qk: constexpr int32 scalar, head dimension for Q/K
        head_dim_v: constexpr int32 scalar, head dimension for V/output
        block_dim_qk: constexpr int32 scalar, power-of-two block for head_dim_qk
        block_dim_v: constexpr int32 scalar, power-of-two block for head_dim_v
        group_size: constexpr int32 scalar, query heads per kv head
        num_kv_heads: constexpr int32 scalar, number of kv heads
        use_q_mask: constexpr bool, apply q mask for tail blocks when True
        use_kv_mask: constexpr bool, apply kv mask for tail blocks when True
        fast_causal: constexpr bool, skip causal masking for full q blocks when True

    Outputs:
        None
    """
    pid_h = tl.program_id(axis=0)
    pid_kv = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    q_head_idx = pid_h
    kv_head_idx = q_head_idx // group_size
    if use_kv_mask:
        kv_mask = pid_kv * block_kv + tl.arange(0, block_kv) < seqlen_kv
    else:
        kv_mask = None

    d_offsets_qk = tl.arange(0, block_dim_qk)
    d_offsets_v = tl.arange(0, block_dim_v)
    dim_mask_qk = d_offsets_qk < head_dim_qk
    dim_mask_qk_tail = (d_offsets_qk + block_dim_qk) < head_dim_qk
    dim_mask_qk_tail2 = (d_offsets_qk + 2 * block_dim_qk) < head_dim_qk
    dim_mask_v = d_offsets_v < head_dim_v

    dK = tl.zeros((block_kv, block_dim_qk), dtype=tl.float32)
    if head_dim_qk > block_dim_qk:
        dK_tail = tl.zeros((block_kv, block_dim_qk), dtype=tl.float32)
    if head_dim_qk > 2 * block_dim_qk:
        dK_tail2 = tl.zeros((block_kv, block_dim_qk), dtype=tl.float32)
    dV = tl.zeros((block_kv, block_dim_v), dtype=tl.float32)

    # load k (block_kv, head_dim_qk) and v (block_kv, head_dim_v)
    kv_offsets = pid_kv * block_kv + tl.arange(0, block_kv)
    k_ptrs = k_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_qk + \
                (kv_head_idx * seqlen_kv * head_dim_qk) + \
                kv_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
    if use_kv_mask:
        k_mask = kv_offsets[:, None] < seqlen_kv
        k_block = tl.load(k_ptrs, mask=k_mask & dim_mask_qk[None, :], other=0.0)
    else:
        k_mask = None
        k_block = tl.load(k_ptrs, mask=dim_mask_qk[None, :], other=0.0)
    if head_dim_qk > block_dim_qk:
        k_ptrs_tail = k_ptrs + block_dim_qk
        if use_kv_mask:
            k_block_tail = tl.load(
                k_ptrs_tail,
                mask=k_mask & dim_mask_qk_tail[None, :],
                other=0.0,
            )
        else:
            k_block_tail = tl.load(k_ptrs_tail, mask=dim_mask_qk_tail[None, :], other=0.0)
    if head_dim_qk > 2 * block_dim_qk:
        k_ptrs_tail2 = k_ptrs + 2 * block_dim_qk
        if use_kv_mask:
            k_block_tail2 = tl.load(
                k_ptrs_tail2,
                mask=k_mask & dim_mask_qk_tail2[None, :],
                other=0.0,
            )
        else:
            k_block_tail2 = tl.load(k_ptrs_tail2, mask=dim_mask_qk_tail2[None, :], other=0.0)
    v_ptrs = v_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_v + \
                (kv_head_idx * seqlen_kv * head_dim_v) + \
                kv_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
    if use_kv_mask:
        v_block = tl.load(v_ptrs, mask=k_mask & dim_mask_v[None, :], other=0.0)
    else:
        v_block = tl.load(v_ptrs, mask=dim_mask_v[None, :], other=0.0)

    if causal:
        seq_len_q_start = pid_kv * block_kv
    else:
        seq_len_q_start = 0
    if fast_causal:
        q_causal_end = seq_len_q_start + block_kv
        q_causal_end = tl.minimum(q_causal_end, seqlen_q)
        for q_start in tl.range(seq_len_q_start, q_causal_end, block_q_sub):
            q_offsets = q_start + tl.arange(0, block_q_sub)
            if use_q_mask:
                q_mask = q_offsets[:, None] < seqlen_q
            else:
                q_mask = None
            q_ptrs = q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                     q_head_idx * seqlen_q * head_dim_qk + \
                     q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            if use_q_mask:
                q_block = tl.load(q_ptrs, mask=q_mask & dim_mask_qk[None, :], other=0.0)
            else:
                q_block = tl.load(q_ptrs, mask=dim_mask_qk[None, :], other=0.0)
            if head_dim_qk > block_dim_qk:
                q_ptrs_tail = q_ptrs + block_dim_qk
                if use_q_mask:
                    q_block_tail = tl.load(q_ptrs_tail, mask=q_mask & dim_mask_qk_tail[None, :], other=0.0)
                else:
                    q_block_tail = tl.load(q_ptrs_tail, mask=dim_mask_qk_tail[None, :], other=0.0)
            if head_dim_qk > 2 * block_dim_qk:
                q_ptrs_tail2 = q_ptrs + 2 * block_dim_qk
                if use_q_mask:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=q_mask & dim_mask_qk_tail2[None, :], other=0.0)
                else:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=dim_mask_qk_tail2[None, :], other=0.0)

            lse_ptrs = lse_ptr + pid_b * q_heads * seqlen_q + \
                        q_head_idx * seqlen_q + \
                        q_offsets
            if use_q_mask:
                lse_block = tl.load(lse_ptrs, mask=q_offsets < seqlen_q, other=0.0)
            else:
                lse_block = tl.load(lse_ptrs)

            p = tl.dot(q_block, k_block.trans(), out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                p += tl.dot(q_block_tail, k_block_tail.trans(), out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                p += tl.dot(q_block_tail2, k_block_tail2.trans(), out_dtype=tl.float32)
            p *= scale
            attn_mask = q_offsets[:, None] >= kv_offsets[None, :]
            p = tl.where(attn_mask, p, -float("inf"))
            p = tl.exp(p - lse_block[:, None])
            p_bf16 = p.to(tl.bfloat16)

            grad_o_ptrs = grad_o_ptr + pid_b * q_heads * seqlen_q * head_dim_v + \
                           q_head_idx * seqlen_q * head_dim_v + \
                           q_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
            if use_q_mask:
                grad_o_block = tl.load(grad_o_ptrs, mask=q_mask & dim_mask_v[None, :], other=0.0)
            else:
                grad_o_block = tl.load(grad_o_ptrs, mask=dim_mask_v[None, :], other=0.0)

            dV += tl.dot(p_bf16.trans(), grad_o_block, out_dtype=tl.float32)

            dP = tl.dot(grad_o_block, v_block.trans(), out_dtype=tl.float32)

            rowwise_dot_o_grad_o_ptrs = rowwise_dot_o_grad_o_ptr + pid_b * q_heads * seqlen_q + \
                                        q_head_idx * seqlen_q + \
                                        q_offsets
            if use_q_mask:
                rowwise_dot_o_grad_o_block = tl.load(
                    rowwise_dot_o_grad_o_ptrs,
                    mask=q_offsets < seqlen_q,
                    other=0.0,
                )
            else:
                rowwise_dot_o_grad_o_block = tl.load(rowwise_dot_o_grad_o_ptrs)

            dS = p * (dP - rowwise_dot_o_grad_o_block[:, None]) * scale
            dS = dS.to(tl.bfloat16)

            dK = tl.dot(dS.trans(), q_block, acc=dK, out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                dK_tail = tl.dot(dS.trans(), q_block_tail, acc=dK_tail, out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                dK_tail2 = tl.dot(dS.trans(), q_block_tail2, acc=dK_tail2, out_dtype=tl.float32)

            grad_q_ptrs = grad_q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                           q_head_idx * seqlen_q * head_dim_qk + \
                           q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            dQ = tl.dot(dS, k_block, out_dtype=tl.float32)
            if use_q_mask:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=q_mask & dim_mask_qk[None, :], sem="relaxed")
            else:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=dim_mask_qk[None, :], sem="relaxed")
            if head_dim_qk > block_dim_qk:
                grad_q_ptrs_tail = grad_q_ptrs + block_dim_qk
                dQ_tail = tl.dot(dS, k_block_tail, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
            if head_dim_qk > 2 * block_dim_qk:
                grad_q_ptrs_tail2 = grad_q_ptrs + 2 * block_dim_qk
                dQ_tail2 = tl.dot(dS, k_block_tail2, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )

        for q_start in tl.range(q_causal_end, seqlen_q, block_q_sub):
            q_offsets = q_start + tl.arange(0, block_q_sub)
            if use_q_mask:
                q_mask = q_offsets[:, None] < seqlen_q
            else:
                q_mask = None
            q_ptrs = q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                     q_head_idx * seqlen_q * head_dim_qk + \
                     q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            if use_q_mask:
                q_block = tl.load(q_ptrs, mask=q_mask & dim_mask_qk[None, :], other=0.0)
            else:
                q_block = tl.load(q_ptrs, mask=dim_mask_qk[None, :], other=0.0)
            if head_dim_qk > block_dim_qk:
                q_ptrs_tail = q_ptrs + block_dim_qk
                if use_q_mask:
                    q_block_tail = tl.load(q_ptrs_tail, mask=q_mask & dim_mask_qk_tail[None, :], other=0.0)
                else:
                    q_block_tail = tl.load(q_ptrs_tail, mask=dim_mask_qk_tail[None, :], other=0.0)
            if head_dim_qk > 2 * block_dim_qk:
                q_ptrs_tail2 = q_ptrs + 2 * block_dim_qk
                if use_q_mask:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=q_mask & dim_mask_qk_tail2[None, :], other=0.0)
                else:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=dim_mask_qk_tail2[None, :], other=0.0)

            lse_ptrs = lse_ptr + pid_b * q_heads * seqlen_q + \
                        q_head_idx * seqlen_q + \
                        q_offsets
            if use_q_mask:
                lse_block = tl.load(lse_ptrs, mask=q_offsets < seqlen_q, other=0.0)
            else:
                lse_block = tl.load(lse_ptrs)

            p = tl.dot(q_block, k_block.trans(), out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                p += tl.dot(q_block_tail, k_block_tail.trans(), out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                p += tl.dot(q_block_tail2, k_block_tail2.trans(), out_dtype=tl.float32)
            p *= scale
            p = tl.exp(p - lse_block[:, None])
            p_bf16 = p.to(tl.bfloat16)

            grad_o_ptrs = grad_o_ptr + pid_b * q_heads * seqlen_q * head_dim_v + \
                           q_head_idx * seqlen_q * head_dim_v + \
                           q_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
            if use_q_mask:
                grad_o_block = tl.load(grad_o_ptrs, mask=q_mask & dim_mask_v[None, :], other=0.0)
            else:
                grad_o_block = tl.load(grad_o_ptrs, mask=dim_mask_v[None, :], other=0.0)

            dV += tl.dot(p_bf16.trans(), grad_o_block, out_dtype=tl.float32)

            dP = tl.dot(grad_o_block, v_block.trans(), out_dtype=tl.float32)

            rowwise_dot_o_grad_o_ptrs = rowwise_dot_o_grad_o_ptr + pid_b * q_heads * seqlen_q + \
                                        q_head_idx * seqlen_q + \
                                        q_offsets
            if use_q_mask:
                rowwise_dot_o_grad_o_block = tl.load(
                    rowwise_dot_o_grad_o_ptrs,
                    mask=q_offsets < seqlen_q,
                    other=0.0,
                )
            else:
                rowwise_dot_o_grad_o_block = tl.load(rowwise_dot_o_grad_o_ptrs)

            dS = p * (dP - rowwise_dot_o_grad_o_block[:, None]) * scale
            dS = dS.to(tl.bfloat16)

            dK = tl.dot(dS.trans(), q_block, acc=dK, out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                dK_tail = tl.dot(dS.trans(), q_block_tail, acc=dK_tail, out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                dK_tail2 = tl.dot(dS.trans(), q_block_tail2, acc=dK_tail2, out_dtype=tl.float32)

            grad_q_ptrs = grad_q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                           q_head_idx * seqlen_q * head_dim_qk + \
                           q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            dQ = tl.dot(dS, k_block, out_dtype=tl.float32)
            if use_q_mask:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=q_mask & dim_mask_qk[None, :], sem="relaxed")
            else:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=dim_mask_qk[None, :], sem="relaxed")
            if head_dim_qk > block_dim_qk:
                grad_q_ptrs_tail = grad_q_ptrs + block_dim_qk
                dQ_tail = tl.dot(dS, k_block_tail, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
            if head_dim_qk > 2 * block_dim_qk:
                grad_q_ptrs_tail2 = grad_q_ptrs + 2 * block_dim_qk
                dQ_tail2 = tl.dot(dS, k_block_tail2, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )
    else:
        for q_start in tl.range(seq_len_q_start, seqlen_q, block_q_sub):
            q_offsets = q_start + tl.arange(0, block_q_sub)
            if use_q_mask:
                q_mask = q_offsets[:, None] < seqlen_q
            else:
                q_mask = None
            q_ptrs = q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                     q_head_idx * seqlen_q * head_dim_qk + \
                     q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            if use_q_mask:
                q_block = tl.load(q_ptrs, mask=q_mask & dim_mask_qk[None, :], other=0.0)
            else:
                q_block = tl.load(q_ptrs, mask=dim_mask_qk[None, :], other=0.0)
            if head_dim_qk > block_dim_qk:
                q_ptrs_tail = q_ptrs + block_dim_qk
                if use_q_mask:
                    q_block_tail = tl.load(q_ptrs_tail, mask=q_mask & dim_mask_qk_tail[None, :], other=0.0)
                else:
                    q_block_tail = tl.load(q_ptrs_tail, mask=dim_mask_qk_tail[None, :], other=0.0)
            if head_dim_qk > 2 * block_dim_qk:
                q_ptrs_tail2 = q_ptrs + 2 * block_dim_qk
                if use_q_mask:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=q_mask & dim_mask_qk_tail2[None, :], other=0.0)
                else:
                    q_block_tail2 = tl.load(q_ptrs_tail2, mask=dim_mask_qk_tail2[None, :], other=0.0)

            lse_ptrs = lse_ptr + pid_b * q_heads * seqlen_q + \
                        q_head_idx * seqlen_q + \
                        q_offsets
            if use_q_mask:
                lse_block = tl.load(lse_ptrs, mask=q_offsets < seqlen_q, other=0.0)
            else:
                lse_block = tl.load(lse_ptrs)

            p = tl.dot(q_block, k_block.trans(), out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                p += tl.dot(q_block_tail, k_block_tail.trans(), out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                p += tl.dot(q_block_tail2, k_block_tail2.trans(), out_dtype=tl.float32)
            p *= scale
            if causal and q_start < pid_kv * block_kv + block_kv - 1:
                attn_mask = q_offsets[:, None] >= kv_offsets[None, :]
                p = tl.where(attn_mask, p, -float("inf"))
            if use_kv_mask:
                p = tl.where(kv_mask[None, :], p, -float("inf"))
            p = tl.exp(p - lse_block[:, None])
            p_bf16 = p.to(tl.bfloat16)

            grad_o_ptrs = grad_o_ptr + pid_b * q_heads * seqlen_q * head_dim_v + \
                           q_head_idx * seqlen_q * head_dim_v + \
                           q_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
            if use_q_mask:
                grad_o_block = tl.load(grad_o_ptrs, mask=q_mask & dim_mask_v[None, :], other=0.0)
            else:
                grad_o_block = tl.load(grad_o_ptrs, mask=dim_mask_v[None, :], other=0.0)

            dV += tl.dot(p_bf16.trans(), grad_o_block, out_dtype=tl.float32)

            dP = tl.dot(grad_o_block, v_block.trans(), out_dtype=tl.float32)

            rowwise_dot_o_grad_o_ptrs = rowwise_dot_o_grad_o_ptr + pid_b * q_heads * seqlen_q + \
                                        q_head_idx * seqlen_q + \
                                        q_offsets
            if use_q_mask:
                rowwise_dot_o_grad_o_block = tl.load(
                    rowwise_dot_o_grad_o_ptrs,
                    mask=q_offsets < seqlen_q,
                    other=0.0,
                )
            else:
                rowwise_dot_o_grad_o_block = tl.load(rowwise_dot_o_grad_o_ptrs)

            dS = p * (dP - rowwise_dot_o_grad_o_block[:, None]) * scale
            dS = dS.to(tl.bfloat16)

            dK = tl.dot(dS.trans(), q_block, acc=dK, out_dtype=tl.float32)
            if head_dim_qk > block_dim_qk:
                dK_tail = tl.dot(dS.trans(), q_block_tail, acc=dK_tail, out_dtype=tl.float32)
            if head_dim_qk > 2 * block_dim_qk:
                dK_tail2 = tl.dot(dS.trans(), q_block_tail2, acc=dK_tail2, out_dtype=tl.float32)

            grad_q_ptrs = grad_q_ptr + pid_b * q_heads * seqlen_q * head_dim_qk + \
                           q_head_idx * seqlen_q * head_dim_qk + \
                           q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
            dQ = tl.dot(dS, k_block, out_dtype=tl.float32)
            if use_q_mask:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=q_mask & dim_mask_qk[None, :], sem="relaxed")
            else:
                tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=dim_mask_qk[None, :], sem="relaxed")
            if head_dim_qk > block_dim_qk:
                grad_q_ptrs_tail = grad_q_ptrs + block_dim_qk
                dQ_tail = tl.dot(dS, k_block_tail, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail,
                        dQ_tail.to(tl.float32),
                        mask=dim_mask_qk_tail[None, :],
                        sem="relaxed",
                    )
            if head_dim_qk > 2 * block_dim_qk:
                grad_q_ptrs_tail2 = grad_q_ptrs + 2 * block_dim_qk
                dQ_tail2 = tl.dot(dS, k_block_tail2, out_dtype=tl.float32)
                if use_q_mask:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=q_mask & dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_q_ptrs_tail2,
                        dQ_tail2.to(tl.float32),
                        mask=dim_mask_qk_tail2[None, :],
                        sem="relaxed",
                    )
    
    # write back
    grad_k_ptrs = grad_k_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_qk + \
                    (kv_head_idx * seqlen_kv * head_dim_qk) + \
                   (pid_kv * block_kv + tl.arange(0, block_kv))[:, None] * head_dim_qk + d_offsets_qk[None, :]
    if group_size == 1:
        if use_kv_mask:
            tl.store(grad_k_ptrs, dK, mask=k_mask & dim_mask_qk[None, :])
        else:
            tl.store(grad_k_ptrs, dK, mask=dim_mask_qk[None, :])
        if head_dim_qk > block_dim_qk:
            grad_k_ptrs_tail = grad_k_ptrs + block_dim_qk
            if use_kv_mask:
                tl.store(
                    grad_k_ptrs_tail,
                    dK_tail,
                    mask=k_mask & dim_mask_qk_tail[None, :],
                )
            else:
                tl.store(
                    grad_k_ptrs_tail,
                    dK_tail,
                    mask=dim_mask_qk_tail[None, :],
                )
        if head_dim_qk > 2 * block_dim_qk:
            grad_k_ptrs_tail2 = grad_k_ptrs + 2 * block_dim_qk
            if use_kv_mask:
                tl.store(
                    grad_k_ptrs_tail2,
                    dK_tail2,
                    mask=k_mask & dim_mask_qk_tail2[None, :],
                )
            else:
                tl.store(
                    grad_k_ptrs_tail2,
                    dK_tail2,
                    mask=dim_mask_qk_tail2[None, :],
                )
    else:
        if use_kv_mask:
            tl.atomic_add(grad_k_ptrs, dK, mask=k_mask & dim_mask_qk[None, :], sem="relaxed")
        else:
            tl.atomic_add(grad_k_ptrs, dK, mask=dim_mask_qk[None, :], sem="relaxed")
        if head_dim_qk > block_dim_qk:
            grad_k_ptrs_tail = grad_k_ptrs + block_dim_qk
            if use_kv_mask:
                tl.atomic_add(
                    grad_k_ptrs_tail,
                    dK_tail,
                    mask=k_mask & dim_mask_qk_tail[None, :],
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_k_ptrs_tail,
                    dK_tail,
                    mask=dim_mask_qk_tail[None, :],
                    sem="relaxed",
                )
        if head_dim_qk > 2 * block_dim_qk:
            grad_k_ptrs_tail2 = grad_k_ptrs + 2 * block_dim_qk
            if use_kv_mask:
                tl.atomic_add(
                    grad_k_ptrs_tail2,
                    dK_tail2,
                    mask=k_mask & dim_mask_qk_tail2[None, :],
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_k_ptrs_tail2,
                    dK_tail2,
                    mask=dim_mask_qk_tail2[None, :],
                    sem="relaxed",
                )
    grad_v_ptrs = grad_v_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_v + \
                    (kv_head_idx * seqlen_kv * head_dim_v) + \
                   (pid_kv * block_kv + tl.arange(0, block_kv))[:, None] * head_dim_v + d_offsets_v[None, :]
    if group_size == 1:
        if use_kv_mask:
            tl.store(grad_v_ptrs, dV, mask=k_mask & dim_mask_v[None, :])
        else:
            tl.store(grad_v_ptrs, dV, mask=dim_mask_v[None, :])
    else:
        if use_kv_mask:
            tl.atomic_add(grad_v_ptrs, dV, mask=k_mask & dim_mask_v[None, :], sem="relaxed")
        else:
            tl.atomic_add(grad_v_ptrs, dV, mask=dim_mask_v[None, :], sem="relaxed")

def launch_flash_mla_backward_bf16_kernel(
    q,
    k,
    v,
    o,
    lse,
    grad_o,
    scale=None,
    causal=True,
    block_kv=64,
    num_warps=8,
    num_stages=3,
):
    """
    Launch the prototype Flash MLA backward kernel.

    Main feature:
        Validates inputs, allocates gradients, and dispatches Triton.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
           [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q]
        grad_o: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
                [batch, q_heads, seqlen_q, head_dim_v]
        scale: optional float scalar, attention scaling factor
        causal: bool scalar, apply causal mask if True
        block_kv: int32 scalar, kv sub-block per flash MLA iteration
        num_warps: int32 scalar, Triton num_warps launch hint
        num_stages: int32 scalar, Triton num_stages launch hint

    Outputs:
        grad_q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
                [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
                [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
                [batch, num_kv_heads, seqlen_kv, head_dim_v]
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or o.dtype != torch.bfloat16
        or grad_o.dtype != torch.bfloat16
    ):
        raise ValueError("Flash MLA backward expects bf16 inputs and grad_o")
    if lse.dtype != torch.float32:
        raise ValueError("Flash MLA backward expects lse to be float32")
    if q.dim() not in (3, 4):
        raise ValueError("Q must be 3D or 4D")
    if k.dim() != q.dim() or v.dim() != q.dim() or o.dim() != q.dim() or grad_o.dim() != q.dim():
        raise ValueError("Q, K, V, O, and grad_o must have matching ranks")

    squeeze_output = q.dim() == 3
    if squeeze_output:
        q_batched = q.contiguous().unsqueeze(0)
        k_batched = k.contiguous().unsqueeze(0)
        v_batched = v.contiguous().unsqueeze(0)
        o_batched = o.contiguous().unsqueeze(0)
        grad_o_batched = grad_o.contiguous().unsqueeze(0)
    else:
        q_batched = q.contiguous()
        k_batched = k.contiguous()
        v_batched = v.contiguous()
        o_batched = o.contiguous()
        grad_o_batched = grad_o.contiguous()

    batch, q_heads, seqlen_q, head_dim_qk = q_batched.shape
    _, num_kv_heads, seqlen_kv, k_head_dim = k_batched.shape
    _, v_heads, v_seqlen, head_dim_v = v_batched.shape
    _, o_heads, o_seqlen, o_head_dim = o_batched.shape
    _, grad_o_heads, grad_o_seqlen, grad_o_head_dim = grad_o_batched.shape
    if seqlen_kv != v_seqlen:
        raise ValueError("K and V must share sequence length")
    if seqlen_q != o_seqlen or seqlen_q != grad_o_seqlen:
        raise ValueError("Q, O, and grad_o must share sequence length")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    if o_head_dim != head_dim_v or grad_o_head_dim != head_dim_v:
        raise ValueError("O and grad_o must share head_dim_v with V")
    if v_heads != num_kv_heads:
        raise ValueError("K and V must share number of KV heads")
    if o_heads != q_heads or grad_o_heads != q_heads:
        raise ValueError("Q, O, and grad_o must share number of query heads")
    if q_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")

    if squeeze_output:
        if lse.dim() != 2:
            raise ValueError("LSE must be 2D when Q is 3D")
        lse_batched = lse.contiguous().unsqueeze(0)
    else:
        if lse.dim() != 3:
            raise ValueError("LSE must be 3D when Q is 4D")
        lse_batched = lse.contiguous()
    if lse_batched.shape != (batch, q_heads, seqlen_q):
        raise ValueError("LSE must have shape [batch, q_heads, seqlen_q]")

    group_size = q_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)

    if 128 < head_dim_qk <= 192:
        block_dim_qk = 64
    elif 192 < head_dim_qk <= 256:
        block_dim_qk = 128
    else:
        block_dim_qk = _next_power_of_2(head_dim_qk)
    block_dim_v = _next_power_of_2(head_dim_v)

    grad_q_batched = torch.zeros_like(q_batched, dtype = torch.float)
    grad_k_batched = torch.zeros_like(k_batched, dtype = torch.float)
    grad_v_batched = torch.zeros_like(v_batched, dtype = torch.float)

    launch_block_kv = block_kv
    block_q_sub = 32
    launch_num_warps = num_warps
    launch_num_stages = num_stages
    if block_kv >= 64 or block_dim_qk > 128 or block_dim_v > 128:
        launch_num_stages = min(num_stages, 2)
    if block_kv > 64 or block_dim_qk > 128 or block_dim_v > 128:
        launch_block_kv = min(block_kv, 32)

    grid = (q_heads, triton.cdiv(seqlen_kv, launch_block_kv), batch)

    rowwise_dot_o_grad_o = compute_rowwise_dot_o_grad_o(o_batched, grad_o_batched)

    flash_mla_backward_bf16_kernel[grid](
        q_batched,
        k_batched,
        v_batched,
        o_batched,
        lse_batched,
        grad_o_batched,
        rowwise_dot_o_grad_o,
        grad_q_batched,
        grad_k_batched,
        grad_v_batched,
        q_heads,
        seqlen_q,
        seqlen_kv,
        scale,
        causal=causal,
        block_q_sub=block_q_sub,
        block_kv=launch_block_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        block_dim_qk=block_dim_qk,
        block_dim_v=block_dim_v,
        group_size=group_size,
        num_kv_heads=num_kv_heads,
        use_q_mask=seqlen_q % block_q_sub != 0,
        use_kv_mask=seqlen_kv % launch_block_kv != 0,
        fast_causal=causal and (seqlen_kv % launch_block_kv == 0 and seqlen_kv >= seqlen_q),
        num_warps=launch_num_warps,
        num_stages=launch_num_stages,
    )

    grad_q_batched, grad_k_batched, grad_v_batched = cast_float32_tensor_to_bf16(
        grad_q_batched,
        grad_k_batched,
        grad_v_batched,
    )

    if squeeze_output:
        return (
            grad_q_batched.squeeze(0),
            grad_k_batched.squeeze(0),
            grad_v_batched.squeeze(0),
        )
    return grad_q_batched, grad_k_batched, grad_v_batched
