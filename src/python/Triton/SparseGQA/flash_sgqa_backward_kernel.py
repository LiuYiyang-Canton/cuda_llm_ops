# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-06
# Purpose: Triton implementation for Flash Sparse GQA backward kernel.
# ==============================================================================

import math

import torch
import triton
import triton.language as tl


@triton.jit
def rowwise_dot_o_grad_o_kernel(
    o_ptr,
    grad_o_ptr,
    rowwise_dot_ptr,
    head_dim_v: tl.constexpr,
    block_dim_v: tl.constexpr,
):
    """
    Compute rowwise dot products between O and grad_o.

    Main feature:
        Loads one flattened [head_dim_v] row for O and grad_o and stores an fp32 rowwise dot product.

    Inputs:
        o_ptr: pointer to bf16 tensor of shape [rows, head_dim_v]
        grad_o_ptr: pointer to bf16 tensor of shape [rows, head_dim_v]
        rowwise_dot_ptr: pointer to float32 tensor of shape [rows]
        head_dim_v: constexpr int32 scalar, value head dimension
        block_dim_v: constexpr int32 scalar, power-of-two block for head_dim_v

    Outputs:
        None
    """
    row_id = tl.program_id(axis=0)
    dim_offsets = tl.arange(0, block_dim_v)
    dim_mask = dim_offsets < head_dim_v
    row_base = row_id * head_dim_v

    o_vals = tl.load(o_ptr + row_base + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    grad_o_vals = tl.load(grad_o_ptr + row_base + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    rowwise_dot = tl.sum(o_vals * grad_o_vals)
    tl.store(rowwise_dot_ptr + row_id, rowwise_dot)


@triton.jit
def float32_to_bf16_kernel(
    input_ptr,
    output_ptr,
    element_count,
    block_size: tl.constexpr,
):
    """
    Cast float32 values to bfloat16 with contiguous block copies.

    Main feature:
        Converts large fp32 gradient buffers to bf16 using a Triton kernel.

    Inputs:
        input_ptr: pointer to float32 tensor of shape [element_count]
        output_ptr: pointer to bf16 tensor of shape [element_count]
        element_count: int32 scalar, element count
        block_size: constexpr int32 scalar, elements per program

    Outputs:
        None
    """
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values.to(tl.bfloat16), mask=mask)


@triton.jit
def flash_sgqa_backward_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    grad_o_ptr,
    rowwise_dot_ptr,
    sparse_kv_indices_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
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
    Compute sparse Flash SGQA backward gradients for one KV head group per program.

    Main feature:
        Each program processes all query heads in one KV-head group for one query token,
        reusing sparse KV indices across grouped query heads.

    Inputs:
        q_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        k_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v_ptr: pointer to bf16 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        lse_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        grad_o_ptr: pointer to bf16 with shape [batch, q_heads, seqlen_q, head_dim_v]
        rowwise_dot_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q]
        sparse_kv_indices_ptr: pointer to int32 with shape [batch, num_kv_heads, seqlen_q, num_sparse_kv]
        grad_q_ptr: pointer to float32 with shape [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k_ptr: pointer to float32 with shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v_ptr: pointer to float32 with shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        q_heads: int32 scalar, number of query heads
        seqlen_q: int32 scalar, query sequence length
        seqlen_kv: int32 scalar, key/value sequence length
        scale: float32 scalar, attention scaling factor
        causal: constexpr bool, apply causal mask if True
        block_q: constexpr int32 scalar, query tokens per block (launcher sets this to 1)
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

    grad_o_base = (pid_b * q_heads + head_idx) * seqlen_q * head_dim_v
    grad_o_ptrs = grad_o_ptr + grad_o_base[:, None] + q_pos * head_dim_v + d_offsets_v[None, :]
    grad_o_group_bf16 = tl.load(grad_o_ptrs, mask=q_valid & dim_mask_v[None, :], other=0.0)
    grad_o_group = grad_o_group_bf16.to(tl.float32)

    lse_ptrs = lse_ptr + (pid_b * q_heads + head_idx) * seqlen_q + q_pos
    lse_group = tl.load(lse_ptrs, mask=q_valid, other=-float("inf"))
    lse_finite = lse_group > -1.0e19
    lse_safe = tl.where(lse_finite, lse_group, 0.0)

    rowwise_dot_ptrs = rowwise_dot_ptr + (pid_b * q_heads + head_idx) * seqlen_q + q_pos
    rowwise_dot = tl.load(rowwise_dot_ptrs, mask=q_valid, other=0.0)

    d_q_group = tl.zeros((group_size, block_dim_qk), dtype=tl.float32)
    log2e = 1.4426950408889634

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
        p = tl.math.exp2((scores - lse_safe[:, None]) * log2e)
        p = tl.where(valid_mask[None, :] & lse_finite[:, None], p, 0.0)

        d_p = tl.dot(grad_o_group_bf16, tl.trans(v_block), out_dtype=tl.float32)
        d_s = p * (d_p - rowwise_dot[:, None]) * scale

        d_q_group += tl.dot(tl.cast(d_s, tl.bfloat16), k_block, out_dtype=tl.float32)
        d_k = tl.sum(d_s[:, :, None] * q_group[:, None, :].to(tl.float32), axis=0)
        d_v = tl.sum(p[:, :, None] * grad_o_group[:, None, :], axis=0)

        grad_k_ptrs = grad_k_ptr + k_base + kv_idx[:, None] * head_dim_qk + d_offsets_qk[None, :]
        grad_v_ptrs = grad_v_ptr + v_base + kv_idx[:, None] * head_dim_v + d_offsets_v[None, :]
        tl.atomic_add(grad_k_ptrs, d_k, mask=valid_mask[:, None] & dim_mask_qk[None, :], sem="relaxed")
        tl.atomic_add(grad_v_ptrs, d_v, mask=valid_mask[:, None] & dim_mask_v[None, :], sem="relaxed")

    grad_q_ptrs = grad_q_ptr + q_base[:, None] + q_pos * head_dim_qk + d_offsets_qk[None, :]
    tl.store(grad_q_ptrs, d_q_group, mask=q_valid & dim_mask_qk[None, :])


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


def _validate_optional_positive_int(value, name):
    """
    Validate optional positive integer launch parameters.

    Main feature:
        Ensures launch override values are None or positive Python ints.

    Inputs:
        value: optional int32 scalar, launch parameter candidate
        name: str scalar, launch parameter name

    Outputs:
        None
    """
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise ValueError(f"{name} must be None or a positive int")


def _select_sgqa_backward_launch_config(
    seqlen_q,
    q_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
):
    """
    Select deterministic launch defaults for SGQA backward.

    Main feature:
        Returns a fixed launch configuration for block_sparse_sub, num_warps, and num_stages.

    Inputs:
        seqlen_q: int32 scalar, query sequence length
        q_heads: int32 scalar, number of query heads
        num_kv_heads: int32 scalar, number of KV heads
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output

    Outputs:
        launch_config: tuple[int32, int32, int32], (block_sparse_sub, num_warps, num_stages)
    """
    _ = seqlen_q
    _ = q_heads
    _ = num_kv_heads
    _ = head_dim_qk
    _ = head_dim_v
    return 16, 2, 3


def _compute_rowwise_dot_o_grad_o(o_batched, grad_o_batched):
    """
    Compute rowwise dot product sum(o * grad_o, dim=head_dim_v) using Triton.

    Main feature:
        Flattens [batch, q_heads, seqlen_q] rows and writes fp32 rowwise reductions.

    Inputs:
        o_batched: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]
        grad_o_batched: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]

    Outputs:
        rowwise_dot: torch.Tensor float32 of shape [batch, q_heads, seqlen_q]
    """
    batch, q_heads, seqlen_q, head_dim_v = o_batched.shape
    row_count = batch * q_heads * seqlen_q
    block_dim_v = _next_power_of_2(head_dim_v)
    rowwise_dot = torch.empty((batch, q_heads, seqlen_q), device=o_batched.device, dtype=torch.float32)
    if row_count > 0:
        grid = (row_count,)
        rowwise_dot_o_grad_o_kernel[grid](
            o_batched,
            grad_o_batched,
            rowwise_dot,
            head_dim_v=head_dim_v,
            block_dim_v=block_dim_v,
            num_warps=2,
            num_stages=1,
        )
    return rowwise_dot


def _cast_float32_to_bf16(input_tensor):
    """
    Cast a float32 tensor to bfloat16 through a Triton conversion kernel.

    Main feature:
        Converts contiguous fp32 buffers with one dedicated cast launch.

    Inputs:
        input_tensor: torch.Tensor float32 of shape [N]

    Outputs:
        output_tensor: torch.Tensor bf16 of shape [N]
    """
    contiguous_input = input_tensor.contiguous()
    output_tensor = torch.empty_like(contiguous_input, dtype=torch.bfloat16)
    element_count = contiguous_input.numel()
    if element_count == 0:
        return output_tensor
    block_size = 1024
    grid = (triton.cdiv(element_count, block_size),)
    float32_to_bf16_kernel[grid](
        contiguous_input,
        output_tensor,
        element_count,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output_tensor


def launch_flash_sgqa_backward_bf16_kernel(
    q,
    k,
    v,
    o,
    lse,
    grad_o,
    sparse_kv_indices,
    num_sparse_kv,
    scale=None,
    causal=True,
    block_sparse_sub=None,
    num_warps=None,
    num_stages=None,
):
    """
    Launch the Flash SGQA backward kernel.

    Main feature:
        Validates inputs, computes rowwise dot prepass, resolves launch config, and dispatches Triton backward.

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
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, num_sparse_kv]
        num_sparse_kv: int32 scalar, sparse KV list length
        scale: optional float scalar, attention scaling factor
        causal: bool scalar, apply causal mask if True
        block_sparse_sub: optional int32 scalar, sparse indices processed per sub-iteration
        num_warps: optional int32 scalar, Triton num_warps launch hint
        num_stages: optional int32 scalar, Triton num_stages launch hint

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
        raise ValueError("Flash SGQA backward expects bf16 inputs and grad_o")
    if lse.dtype != torch.float32:
        raise ValueError("Flash SGQA backward expects lse to be float32")
    if sparse_kv_indices.dtype != torch.int32:
        raise ValueError("sparse_kv_indices must be int32")
    if sparse_kv_indices.dim() != 4:
        raise ValueError("sparse_kv_indices must be 4D [batch, num_kv_heads, seqlen_q, num_sparse_kv]")
    if not isinstance(num_sparse_kv, int):
        raise ValueError("num_sparse_kv must be an int")
    if num_sparse_kv < 1:
        raise ValueError("num_sparse_kv must be >= 1")

    _validate_optional_positive_int(block_sparse_sub, "block_sparse_sub")
    _validate_optional_positive_int(num_warps, "num_warps")
    _validate_optional_positive_int(num_stages, "num_stages")

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

    block_dim_qk = _next_power_of_2(head_dim_qk)
    block_dim_v = _next_power_of_2(head_dim_v)

    (
        auto_block_sparse_sub,
        auto_num_warps,
        auto_num_stages,
    ) = _select_sgqa_backward_launch_config(
        seqlen_q=seqlen_q,
        q_heads=q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
    )
    effective_block_sparse_sub = (
        block_sparse_sub if block_sparse_sub is not None else auto_block_sparse_sub
    )
    effective_num_warps = num_warps if num_warps is not None else auto_num_warps
    effective_num_stages = num_stages if num_stages is not None else auto_num_stages

    launch_num_stages = effective_num_stages
    if block_dim_qk > 128 or block_dim_v > 128:
        launch_num_stages = min(effective_num_stages, 2)

    rowwise_dot_batched = _compute_rowwise_dot_o_grad_o(o_batched, grad_o_batched)
    grad_q_batched = torch.zeros_like(q_batched, dtype=torch.float32)
    grad_k_batched = torch.zeros_like(k_batched, dtype=torch.float32)
    grad_v_batched = torch.zeros_like(v_batched, dtype=torch.float32)

    if seqlen_q > 0:
        grid = (num_kv_heads, triton.cdiv(seqlen_q, block_q), batch)
        flash_sgqa_backward_bf16_kernel[grid](
            q_batched,
            k_batched,
            v_batched,
            lse_batched,
            grad_o_batched,
            rowwise_dot_batched,
            sparse_kv_indices_batched,
            grad_q_batched,
            grad_k_batched,
            grad_v_batched,
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

    grad_q_out = _cast_float32_to_bf16(grad_q_batched)
    grad_k_out = _cast_float32_to_bf16(grad_k_batched)
    grad_v_out = _cast_float32_to_bf16(grad_v_batched)

    if squeeze_output:
        return (
            grad_q_out.squeeze(0),
            grad_k_out.squeeze(0),
            grad_v_out.squeeze(0),
        )
    return grad_q_out, grad_k_out, grad_v_out
