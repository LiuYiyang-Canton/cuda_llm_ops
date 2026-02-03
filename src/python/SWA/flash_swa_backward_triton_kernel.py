# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-03
# Purpose: Triton implementation for Flash SWA backward kernel.
# ==============================================================================

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
def float32_to_bf16_kernel(
    input_ptr,
    output_ptr,
    element_count,
    block_size: tl.constexpr,
):
    """
    Cast a float32 array to bfloat16 in block-aligned chunks.

    Main feature:
        Copies each block of float32 elements and casts to bf16 inside Triton to avoid host-side remapping.

    Inputs:
        input_ptr: pointer to float32 tensor of shape [element_count]
        output_ptr: pointer to bf16 tensor of shape [element_count]
        element_count: number of elements to cast
        block_size: constexpr int defining the block width

    Outputs:
        None. The bf16 results are written into output_ptr.
    """
    block_start = tl.program_id(axis=0) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values.to(tl.bfloat16), mask=mask)


def cast_float32_tensor_to_bf16(float_tensor):
    """
    Convert a contiguous float32 tensor to bf16 via a Triton kernel.

    Main feature:
        Uses a block-wise Triton kernel to avoid Python-level `.to(torch.bfloat16)` overhead.

    Inputs:
        float_tensor: torch.Tensor float32 of arbitrary shape

    Outputs:
        torch.Tensor bf16 with the same shape and device as float_tensor.
    """
    float_tensor = float_tensor.contiguous()
    output_tensor = torch.empty_like(float_tensor, dtype=torch.bfloat16)
    element_count = float_tensor.numel()
    if element_count == 0:
        return output_tensor
    block_size = 256
    grid = (triton.cdiv(element_count, block_size),)
    float32_to_bf16_kernel[grid](
        float_tensor,
        output_tensor,
        element_count,
        block_size=block_size,
    )
    return output_tensor


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
def flash_swa_backward_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
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
    block_q_sub: tl.constexpr,
    block_kv: tl.constexpr,
    head_dim_qk: tl.constexpr,
    head_dim_v: tl.constexpr,
    block_dim_qk: tl.constexpr,
    block_dim_v: tl.constexpr,
    group_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    window_size: tl.constexpr,
):
    """
    Prototype Flash SWA backward kernel (bf16 inputs, fp32 accumulation).

    Main feature:
        Applies sliding-window attention with a causal mask and uses sink-adjusted log-sum-exp.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
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
        block_kv: constexpr int32 scalar, kv sub-block per iteration
        head_dim_qk: constexpr int32 scalar, head dimension for Q/K
        head_dim_v: constexpr int32 scalar, head dimension for V/output
        block_dim_qk: constexpr int32 scalar, power-of-two block for head_dim_qk
        block_dim_v: constexpr int32 scalar, power-of-two block for head_dim_v
        group_size: constexpr int32 scalar, query heads per kv head
        num_kv_heads: constexpr int32 scalar, number of kv heads
        window_size: constexpr int32 scalar, sliding-window size

    Outputs:
        None
    """
    pid_h = tl.program_id(axis=0)
    pid_kv = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    q_head_idx = pid_h
    kv_head_idx = q_head_idx // group_size
    kv_offsets = pid_kv * block_kv + tl.arange(0, block_kv)
    kv_mask = kv_offsets < seqlen_kv

    d_offsets_qk = tl.arange(0, block_dim_qk)
    d_offsets_v = tl.arange(0, block_dim_v)
    dim_mask_qk = d_offsets_qk < head_dim_qk
    dim_mask_v = d_offsets_v < head_dim_v

    dK = tl.zeros((block_kv, block_dim_qk), dtype=tl.float32)
    dV = tl.zeros((block_kv, block_dim_v), dtype=tl.float32)

    # load k (block_kv, head_dim_qk) and v (block_kv, head_dim_v)
    kv_start = pid_kv * block_kv
    k_mask = kv_offsets[:, None] < seqlen_kv
    k_base = (pid_b * num_kv_heads + kv_head_idx) * seqlen_kv * head_dim_qk
    v_base = (pid_b * num_kv_heads + kv_head_idx) * seqlen_kv * head_dim_v
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_base,
        shape=(seqlen_kv, head_dim_qk),
        strides=(head_dim_qk, 1),
        offsets=(kv_start, 0),
        block_shape=(block_kv, block_dim_qk),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_base,
        shape=(seqlen_kv, head_dim_v),
        strides=(head_dim_v, 1),
        offsets=(kv_start, 0),
        block_shape=(block_kv, block_dim_v),
        order=(1, 0),
    )
    k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

    q_start_min = kv_start
    q_start_max = kv_start + block_kv + window_size - 1
    q_start_max = tl.minimum(q_start_max, seqlen_q)
    q_base_qk = (pid_b * q_heads + q_head_idx) * seqlen_q * head_dim_qk
    q_base_v = (pid_b * q_heads + q_head_idx) * seqlen_q * head_dim_v
    q_base_row = (pid_b * q_heads + q_head_idx) * seqlen_q
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_base_qk,
        shape=(seqlen_q, head_dim_qk),
        strides=(head_dim_qk, 1),
        offsets=(q_start_min, 0),
        block_shape=(block_q_sub, block_dim_qk),
        order=(1, 0),
    )
    for q_start in tl.range(q_start_min, q_start_max, block_q_sub):
        # load q (block_q_sub, head_dim_qk)
        q_offsets = q_start + tl.arange(0, block_q_sub)
        q_mask = q_offsets[:, None] < seqlen_q
        q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # load lse (block_q_sub,)
        lse_ptrs = lse_ptr + q_base_row + q_offsets
        lse_block = tl.load(lse_ptrs, mask=q_offsets < seqlen_q, other=0.0)

        # recompute p = softmax(q @ k^T) (block_q_sub, block_kv)
        p = tl.dot(q_block, k_block.trans(), out_dtype=tl.float32) * scale
        window_mask = kv_offsets[None, :] >= (q_offsets[:, None] - (window_size - 1))
        causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
        attn_mask = kv_mask[None, :] & window_mask & causal_mask
        p = tl.where(attn_mask, p, -float("inf"))
        p = tl.exp2((p - lse_block[:, None]) * 1.4426950408889634)
        p_bf16 = p.to(tl.bfloat16)

        # load grad_o (block_q_sub, head_dim_v)
        grad_o_ptrs = grad_o_ptr + q_base_v + q_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
        grad_o_block = tl.load(grad_o_ptrs, mask=q_mask & dim_mask_v[None, :], other=0.0)

        # dV = p^T @ grad_o (block_kv, head_dim_v)
        dV += tl.dot(p_bf16.trans(), grad_o_block, out_dtype=tl.float32)

        # dP = grad_o @ v^T (block_q_sub, block_kv)
        dP = tl.dot(grad_o_block, v_block.trans(), out_dtype=tl.float32)

        # load rowwise_dot_o_grad_o (block_q_sub,)
        rowwise_dot_o_grad_o_ptrs = rowwise_dot_o_grad_o_ptr + q_base_row + q_offsets
        rowwise_dot_o_grad_o_block = tl.load(rowwise_dot_o_grad_o_ptrs, mask=q_offsets < seqlen_q, other=0.0)

        # dS = p * (dP - rowwise_dot_o_grad_o) (block_q_sub, block_kv)
        dS = p * (dP - rowwise_dot_o_grad_o_block[:, None]) * scale
        dS = dS.to(tl.bfloat16)

        dK = tl.dot(dS.trans(), q_block, acc=dK, out_dtype=tl.float32)

        # dQ += dS @ k_block
        grad_q_ptrs = grad_q_ptr + q_base_qk + q_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
        dQ = tl.dot(dS, k_block, out_dtype=tl.float32)
        tl.atomic_add(grad_q_ptrs, dQ.to(tl.float32), mask=q_mask & dim_mask_qk[None, :], sem="relaxed")
        q_block_ptr = tl.advance(q_block_ptr, (block_q_sub, 0))
    
    # write back
    grad_k_ptrs = grad_k_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_qk + \
                    (kv_head_idx * seqlen_kv * head_dim_qk) + \
                   kv_offsets[:, None] * head_dim_qk + d_offsets_qk[None, :]
    tl.atomic_add(grad_k_ptrs, dK, mask=k_mask & dim_mask_qk[None, :], sem="relaxed")
    grad_v_ptrs = grad_v_ptr + pid_b * num_kv_heads * seqlen_kv * head_dim_v + \
                    (kv_head_idx * seqlen_kv * head_dim_v) + \
                   kv_offsets[:, None] * head_dim_v + d_offsets_v[None, :]
    tl.atomic_add(grad_v_ptrs, dV, mask=k_mask & dim_mask_v[None, :], sem="relaxed")

def launch_flash_swa_backward_bf16_kernel(
    q,
    k,
    v,
    o,
    lse,
    grad_o,
    scale=None,
    window_size=None,
    block_kv=64,
    num_warps=8,
    num_stages=3,
):
    """
    Launch the prototype Flash SWA backward kernel.

    Main feature:
        Validates inputs, allocates gradients, and dispatches Triton with sliding-window causal masks.

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
             [batch, q_heads, seqlen_q], computed with the sink term
        grad_o: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
                [batch, q_heads, seqlen_q, head_dim_v]
        scale: optional float scalar, attention scaling factor
        window_size: int32 scalar, sliding-window size
        block_kv: int32 scalar, kv sub-block per flash SWA iteration
        num_warps: int32 scalar, Triton num_warps launch hint
        num_stages: int32 scalar, Triton num_stages launch hint

    Outputs:
        grad_q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
                [batch, q_heads, seqlen_q, head_dim_qk] (computed in float32 then cast)
        grad_k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
                [batch, num_kv_heads, seqlen_kv, head_dim_qk] (computed in float32 then cast)
        grad_v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
                [batch, num_kv_heads, seqlen_kv, head_dim_v] (computed in float32 then cast)
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or o.dtype != torch.bfloat16
        or grad_o.dtype != torch.bfloat16
    ):
        raise ValueError("Flash SWA backward expects bf16 inputs and grad_o")
    if lse.dtype != torch.float32:
        raise ValueError("Flash SWA backward expects lse to be float32")
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
    if window_size is None:
        raise ValueError("window_size must be provided")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)

    block_dim_qk = _next_power_of_2(head_dim_qk)
    block_dim_v = _next_power_of_2(head_dim_v)

    grad_q_batched = torch.zeros_like(q_batched, dtype=torch.float)
    grad_k_batched = torch.zeros_like(k_batched, dtype=torch.float)
    grad_v_batched = torch.zeros_like(v_batched, dtype=torch.float)

    launch_block_kv = block_kv
    launch_num_stages = num_stages
    if block_dim_qk > 128 or block_dim_v > 128:
        launch_block_kv = min(block_kv, 32)
        launch_num_stages = min(num_stages, 2)

    grid = (q_heads, triton.cdiv(seqlen_kv, launch_block_kv), batch)

    rowwise_dot_o_grad_o = compute_rowwise_dot_o_grad_o(o_batched, grad_o_batched)

    flash_swa_backward_bf16_kernel[grid](
        q_batched,
        k_batched,
        v_batched,
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
        block_q_sub=32,
        block_kv=launch_block_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        block_dim_qk=block_dim_qk,
        block_dim_v=block_dim_v,
        group_size=group_size,
        num_kv_heads=num_kv_heads,
        window_size=window_size,
        num_warps=num_warps,
        num_stages=launch_num_stages,
    )

    grad_q_batched = cast_float32_tensor_to_bf16(grad_q_batched)
    grad_k_batched = cast_float32_tensor_to_bf16(grad_k_batched)
    grad_v_batched = cast_float32_tensor_to_bf16(grad_v_batched)

    if squeeze_output:
        return (
            grad_q_batched.squeeze(0),
            grad_k_batched.squeeze(0),
            grad_v_batched.squeeze(0),
        )
    return grad_q_batched, grad_k_batched, grad_v_batched
