# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Triton SSD kernels and launch wrappers for Mamba-2.
# ==============================================================================
"""Triton SSD kernels and launch wrappers for Mamba-2."""

import functools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_head": 8}, num_stages=2, num_warps=4),
    ],
    key=["head_dim", "state_dim"],
)
@triton.jit
def ssd_intra_chunk_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    out_y_ptr,
    out_a_cumsum_last_ptr,
    out_state_decay_ptr,
    out_states_ptr,
    seq_len,
    num_heads,
    num_groups,
    stride_x_b,
    stride_x_l,
    stride_x_h,
    stride_x_d,
    stride_a_b,
    stride_a_l,
    stride_a_h,
    stride_b_b,
    stride_b_l,
    stride_b_g,
    stride_b_d,
    stride_c_b,
    stride_c_l,
    stride_c_g,
    stride_c_d,
    stride_out_y_b,
    stride_out_y_l,
    stride_out_y_h,
    stride_out_y_d,
    stride_out_a_cumsum_b,
    stride_out_a_cumsum_h,
    stride_out_a_cumsum_c,
    stride_out_state_decay_b,
    stride_out_state_decay_h,
    stride_out_state_decay_c,
    stride_out_state_decay_l,
    stride_out_states_b,
    stride_out_states_c,
    stride_out_states_h,
    stride_out_states_d,
    stride_out_states_k,
    state_dim: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    block_head: tl.constexpr,
):
    """
    Compute intra-chunk SSD contributions for Mamba-2.

    Main feature:
        Produces diagonal outputs, per-step decays, and chunk boundary states.

    Inputs:
        x_ptr: pointer to bf16 with shape [batch, seq_len, num_heads, head_dim]
        a_ptr: pointer to float32 with shape [batch, seq_len, num_heads]
        b_ptr: pointer to bf16 with shape [batch, seq_len, num_groups, state_dim]
        c_ptr: pointer to bf16 with shape [batch, seq_len, num_groups, state_dim]
        out_y_ptr: pointer to float32 with shape [batch, seq_len, num_heads, head_dim]
        out_a_cumsum_last_ptr: pointer to float32 with shape [batch, num_heads, 1 + num_chunks]
        out_state_decay_ptr: pointer to float32 with shape [batch, num_heads, num_chunks, chunk_size]
        out_states_ptr: pointer to float32 with shape [batch, 1 + num_chunks, num_heads, head_dim, state_dim]
        seq_len: int32 scalar
        num_heads: int32 scalar
        num_groups: int32 scalar
        stride_*: int32 scalars for tensor strides
        state_dim: constexpr int32 scalar
        head_dim: constexpr int32 scalar
        chunk_size: constexpr int32 scalar
        block_head: constexpr int32 scalar

    Outputs:
        None
    """
    chunk_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    chunk_start = chunk_idx * chunk_size

    seq_offsets = tl.arange(0, chunk_size)
    head_offsets = tl.arange(0, head_dim)
    valid_mask = (chunk_start + seq_offsets) < seq_len
    state_offsets = tl.arange(0, state_dim)

    b_ptrs = (
        b_ptr
        + batch_idx * stride_b_b
        + chunk_start * stride_b_l
        + ((head_block_idx * block_head) % num_groups) * stride_b_g
    )
    b_vals = tl.load(
        b_ptrs + seq_offsets[:, None] * stride_b_l + state_offsets[None, :] * stride_b_d,
        mask=valid_mask[:, None],
        other=0.0,
    )

    c_ptrs = (
        c_ptr
        + batch_idx * stride_c_b
        + chunk_start * stride_c_l
        + ((head_block_idx * block_head) % num_groups) * stride_c_g
    )
    c_vals = tl.load(
        c_ptrs + seq_offsets[:, None] * stride_c_l + state_offsets[None, :] * stride_c_d,
        mask=valid_mask[:, None],
        other=0.0,
    )

    ctb = tl.dot(c_vals, tl.trans(b_vals), out_dtype=tl.float32)

    rows = tl.arange(0, chunk_size)
    cols = tl.arange(0, chunk_size)

    for head_idx in range(head_block_idx * block_head, (head_block_idx + 1) * block_head):
        x_ptrs = x_ptr + batch_idx * stride_x_b + chunk_start * stride_x_l + head_idx * stride_x_h
        x_vals = tl.load(
            x_ptrs + seq_offsets[:, None] * stride_x_l + head_offsets[None, :] * stride_x_d,
            mask=valid_mask[:, None],
            other=0.0,
        )

        a_ptrs = a_ptr + batch_idx * stride_a_b + chunk_start * stride_a_l + head_idx * stride_a_h
        a_vals = tl.load(a_ptrs + seq_offsets * stride_a_l, mask=valid_mask, other=0.0)

        a_cumsum = tl.cumsum(a_vals, axis=0)
        l_vals = tl.exp(a_cumsum[:, None] - a_cumsum[None, :])
        l_vals = tl.where(rows[:, None] >= cols[None, :], l_vals, 0.0)

        state_decay = tl.exp(a_cumsum)

        l_ctb = l_vals * ctb

        y_diag = tl.dot(l_ctb.to(tl.bfloat16), x_vals, out_dtype=tl.float32)

        out_y_ptrs = out_y_ptr + batch_idx * stride_out_y_b + chunk_start * stride_out_y_l + head_idx * stride_out_y_h
        tl.store(
            out_y_ptrs + seq_offsets[:, None] * stride_out_y_l + head_offsets[None, :] * stride_out_y_d,
            y_diag,
            mask=valid_mask[:, None],
        )

        a_cumsum_last = tl.sum(tl.where(seq_offsets == chunk_size - 1, a_cumsum, 0.0))
        decay_states = tl.exp(a_cumsum_last - a_cumsum)

        b_decay = b_vals * decay_states[:, None]

        states = tl.dot(tl.trans(b_decay.to(tl.bfloat16)), x_vals, out_dtype=tl.float32)

        state_decay_ptrs = (
            out_state_decay_ptr
            + batch_idx * stride_out_state_decay_b
            + head_idx * stride_out_state_decay_h
            + chunk_idx * stride_out_state_decay_c
        )
        tl.store(state_decay_ptrs + seq_offsets * stride_out_state_decay_l, state_decay, mask=valid_mask)

        out_states_ptrs = (
            out_states_ptr
            + batch_idx * stride_out_states_b
            + (1 + chunk_idx) * stride_out_states_c
            + head_idx * stride_out_states_h
            + head_offsets[:, None] * stride_out_states_d
            + state_offsets[None, :] * stride_out_states_k
        )
        tl.store(out_states_ptrs, tl.trans(states), mask=True)

        out_a_cumsum_last_ptrs = (
            out_a_cumsum_last_ptr
            + batch_idx * stride_out_a_cumsum_b
            + head_idx * stride_out_a_cumsum_h
            + (1 + chunk_idx) * stride_out_a_cumsum_c
        )
        tl.store(out_a_cumsum_last_ptrs, a_cumsum_last, mask=True)


@triton.autotune(
    configs=[
        triton.Config({"block_headdim": 16}),
    ],
    key=["head_dim", "state_dim"],
)
@triton.jit
def ssd_inter_chunk_scan_linear_kernel(
    a_intra_sum_ptr,
    old_states_ptr,
    c_ptr,
    state_decay_ptr,
    diag_y_ptr,
    final_state_ptr,
    out_y_ptr,
    num_chunks,
    state_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    num_groups,
    stride_a_b,
    stride_a_h,
    stride_a_c,
    stride_old_states_b,
    stride_old_states_c,
    stride_old_states_h,
    stride_old_states_d,
    stride_old_states_k,
    stride_c_b,
    stride_c_c,
    stride_c_l,
    stride_c_h,
    stride_c_n,
    stride_state_decay_b,
    stride_state_decay_h,
    stride_state_decay_c,
    stride_state_decay_l,
    stride_final_b,
    stride_final_h,
    stride_final_d,
    stride_final_k,
    stride_out_y_b,
    stride_out_y_c,
    stride_out_y_l,
    stride_out_y_h,
    stride_out_y_p,
    block_headdim: tl.constexpr,
):
    """
    Scan across chunks to update states and produce full SSD output.

    Main feature:
        Applies inter-chunk decay/accumulation and mixes diagonal outputs.

    Inputs:
        a_intra_sum_ptr: pointer to float32 with shape [batch, num_heads, 1 + num_chunks]
        old_states_ptr: pointer to float32 with shape [batch, 1 + num_chunks, num_heads, head_dim, state_dim]
        c_ptr: pointer to bf16 with shape [batch, num_chunks, chunk_size, num_groups, state_dim]
        state_decay_ptr: pointer to float32 with shape [batch, num_heads, num_chunks, chunk_size]
        diag_y_ptr: pointer to float32 with shape [batch, num_chunks, chunk_size, num_heads, head_dim]
        final_state_ptr: pointer to float32 with shape [batch, num_heads, head_dim, state_dim]
        out_y_ptr: pointer to bf16 with shape [batch, num_chunks, chunk_size, num_heads, head_dim]
        num_chunks: int32 scalar
        state_dim: constexpr int32 scalar
        chunk_size: constexpr int32 scalar
        num_groups: int32 scalar
        stride_*: int32 scalars for tensor strides
        block_headdim: constexpr int32 scalar

    Outputs:
        None
    """
    head_dim_idx = tl.program_id(0) * block_headdim
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    chunk_offsets = tl.arange(0, chunk_size)
    head_dim_offsets = tl.arange(0, block_headdim)
    state_offsets = tl.arange(0, state_dim)

    ptr_a = a_intra_sum_ptr + batch_idx * stride_a_b + head_idx * stride_a_h

    base_offset = (
        batch_idx * stride_old_states_b
        + head_idx * stride_old_states_h
        + head_dim_idx * stride_old_states_d
    )

    old_states_ptrs = old_states_ptr + base_offset
    final_offset = (
        batch_idx * stride_final_b
        + head_idx * stride_final_h
        + head_dim_idx * stride_final_d
    )
    final_ptrs = final_state_ptr + final_offset

    c_offset = batch_idx * stride_c_b + (head_idx % num_groups) * stride_c_h
    c_ptrs = c_ptr + c_offset

    state_decay_offset = batch_idx * stride_state_decay_b + head_idx * stride_state_decay_h
    state_decay_ptrs = state_decay_ptr + state_decay_offset

    out_y_offset = batch_idx * stride_out_y_b + head_idx * stride_out_y_h + head_dim_idx * stride_out_y_p
    diag_y_ptrs = diag_y_ptr + out_y_offset
    out_y_ptrs = out_y_ptr + out_y_offset

    running_state = tl.zeros((block_headdim, state_dim), dtype=tl.float32)

    for chunk_idx in range(num_chunks):
        curr_a_val = tl.load(ptr_a + chunk_idx * stride_a_c)
        decay = tl.exp(curr_a_val)
        running_state = running_state * decay

        ptr_input_chunk = (
            old_states_ptrs
            + (chunk_idx * stride_old_states_c)
            + (head_dim_offsets[:, None] * stride_old_states_d)
            + (state_offsets[None, :] * stride_old_states_k)
        )
        input_state = tl.load(ptr_input_chunk, mask=True, other=0.0)
        running_state = running_state + input_state

        c_ptrs_chunk = (
            c_ptrs
            + (chunk_idx * stride_c_c)
            + (chunk_offsets[:, None] * stride_c_l)
            + (state_offsets[None, :] * stride_c_n)
        )
        c_vals = tl.load(c_ptrs_chunk, mask=True, other=0.0)

        state_decay_ptrs_chunk = (
            state_decay_ptrs
            + (chunk_idx * stride_state_decay_c)
            + (chunk_offsets * stride_state_decay_l)
        )
        state_decay_vals = tl.load(state_decay_ptrs_chunk, mask=True, other=0.0)

        c_state_decay = (
            tl.dot(c_vals, tl.trans(running_state.to(tl.bfloat16)), out_dtype=tl.float32)
            * state_decay_vals[:, None]
        )

        diag_y_ptrs_chunk = (
            diag_y_ptrs
            + (chunk_idx * stride_out_y_c)
            + (chunk_offsets[:, None] * stride_out_y_l)
            + (head_dim_offsets[None, :] * stride_out_y_p)
        )
        diag_y_chunk = tl.load(diag_y_ptrs_chunk, mask=True, other=0.0)

        out_y_ptrs_chunk = (
            out_y_ptrs
            + (chunk_idx * stride_out_y_c)
            + (chunk_offsets[:, None] * stride_out_y_l)
            + (head_dim_offsets[None, :] * stride_out_y_p)
        )
        tl.store(out_y_ptrs_chunk, (c_state_decay + diag_y_chunk).to(tl.bfloat16), mask=True)

    curr_a_val = tl.load(ptr_a + num_chunks * stride_a_c)
    decay = tl.exp(curr_a_val)
    running_state = running_state * decay

    ptr_input_chunk = (
        old_states_ptrs
        + (num_chunks * stride_old_states_c)
        + (head_dim_offsets[:, None] * stride_old_states_d)
        + (state_offsets[None, :] * stride_old_states_k)
    )
    input_state = tl.load(ptr_input_chunk, mask=True, other=0.0)

    running_state = running_state + input_state

    ptr_final_chunk = (
        final_ptrs
        + (head_dim_offsets[:, None] * stride_final_d)
        + (state_offsets[None, :] * stride_final_k)
    )
    tl.store(ptr_final_chunk, running_state, mask=True)


def ssd_intra_chunk_grid(meta, seq_len, chunk_size, num_heads, batch_size):
    """
    Compute the Triton grid for the intra-chunk kernel.

    Main feature:
        Maps chunks, head blocks, and batches to a 3D grid.

    Inputs:
        meta: dict with key "block_head" (int32)
        seq_len: int32 scalar
        chunk_size: int32 scalar
        num_heads: int32 scalar
        batch_size: int32 scalar

    Outputs:
        grid: tuple[int32, int32, int32] with shape [3]
    """
    return (
        triton.cdiv(seq_len, chunk_size),
        num_heads // meta["block_head"],
        batch_size,
    )


def ssd_inter_chunk_grid(meta, head_dim, num_heads, batch_size):
    """
    Compute the Triton grid for the inter-chunk scan kernel.

    Main feature:
        Maps head-dimension tiles, heads, and batches to a 3D grid.

    Inputs:
        meta: dict with key "block_headdim" (int32)
        head_dim: int32 scalar
        num_heads: int32 scalar
        batch_size: int32 scalar

    Outputs:
        grid: tuple[int32, int32, int32] with shape [3]
    """
    return (
        head_dim // meta["block_headdim"],
        num_heads,
        batch_size,
    )


def launch_ssd_intra_chunk_kernel(
    x,
    a,
    b,
    c,
    diag_y,
    intra_chunk_cumsum,
    state_decay,
    chunk_states,
    chunk_size,
):
    """
    Launch the Triton intra-chunk SSD kernel.

    Main feature:
        Produces diagonal outputs and intermediate states for SSD.

    Inputs:
        x: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, seq_len, num_heads]
        b: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        c: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        diag_y: torch.Tensor float32 of shape [batch, seq_len, num_heads, head_dim]
        intra_chunk_cumsum: torch.Tensor float32 of shape [batch, num_heads, 1 + num_chunks]
        state_decay: torch.Tensor float32 of shape [batch, num_heads, num_chunks, chunk_size]
        chunk_states: torch.Tensor float32 of shape [batch, 1 + num_chunks, num_heads, head_dim, state_dim]
        chunk_size: int32 scalar

    Outputs:
        None
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape (batch, seq_len, num_heads, head_dim); got {x.shape}")
    batch_size, seq_len, num_heads, head_dim = x.shape
    if a.shape != (batch_size, seq_len, num_heads):
        raise ValueError(f"a must be shape (batch, seq_len, num_heads); got {a.shape}")

    num_groups = b.shape[2]
    state_dim = b.shape[3]
    if b.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"b must have leading shape (batch, seq_len); got {b.shape[:2]}")
    if c.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"c must have leading shape (batch, seq_len); got {c.shape[:2]}")
    if b.shape[2:] != c.shape[2:]:
        raise ValueError(f"b and c must share (num_groups, state_dim); got {b.shape[2:]} vs {c.shape[2:]}")

    for tensor_name, tensor in (
        ("x", x),
        ("a", a),
        ("b", b),
        ("c", c),
        ("diag_y", diag_y),
        ("intra_chunk_cumsum", intra_chunk_cumsum),
        ("state_decay", state_decay),
        ("chunk_states", chunk_states),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous")

    stride_x_b, stride_x_l, stride_x_h, stride_x_d = x.stride()
    stride_a_b, stride_a_l, stride_a_h = a.stride()
    stride_b_b, stride_b_l, stride_b_g, stride_b_d = b.stride()
    stride_c_b, stride_c_l, stride_c_g, stride_c_d = c.stride()
    stride_diag_y_b, stride_diag_y_l, stride_diag_y_h, stride_diag_y_d = diag_y.stride()
    stride_a_cumsum_b, stride_a_cumsum_h, stride_a_cumsum_c = intra_chunk_cumsum.stride()
    stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l = state_decay.stride()
    stride_out_states_b, stride_out_states_c, stride_out_states_h, stride_out_states_d, stride_out_states_k = chunk_states.stride()

    grid = functools.partial(
        ssd_intra_chunk_grid,
        seq_len=seq_len,
        chunk_size=chunk_size,
        num_heads=num_heads,
        batch_size=batch_size,
    )

    ssd_intra_chunk_kernel[grid](
        x,
        a,
        b,
        c,
        diag_y,
        intra_chunk_cumsum,
        state_decay,
        chunk_states,
        seq_len,
        num_heads,
        num_groups,
        stride_x_b,
        stride_x_l,
        stride_x_h,
        stride_x_d,
        stride_a_b,
        stride_a_l,
        stride_a_h,
        stride_b_b,
        stride_b_l,
        stride_b_g,
        stride_b_d,
        stride_c_b,
        stride_c_l,
        stride_c_g,
        stride_c_d,
        stride_diag_y_b,
        stride_diag_y_l,
        stride_diag_y_h,
        stride_diag_y_d,
        stride_a_cumsum_b,
        stride_a_cumsum_h,
        stride_a_cumsum_c,
        stride_state_decay_b,
        stride_state_decay_h,
        stride_state_decay_c,
        stride_state_decay_l,
        stride_out_states_b,
        stride_out_states_c,
        stride_out_states_h,
        stride_out_states_d,
        stride_out_states_k,
        state_dim,
        head_dim,
        chunk_size,
    )


def launch_ssd_inter_chunk_scan_linear_kernel(
    intra_chunk_cumsum,
    chunk_states,
    c,
    state_decay,
    diag_y,
    final_states,
    out_y,
    num_chunks,
    chunk_size,
    num_groups,
):
    """
    Launch the Triton inter-chunk scan kernel.

    Main feature:
        Scans chunks to compute full SSD output and final state.

    Inputs:
        intra_chunk_cumsum: torch.Tensor float32 of shape [batch, num_heads, 1 + num_chunks]
        chunk_states: torch.Tensor float32 of shape [batch, 1 + num_chunks, num_heads, head_dim, state_dim]
        c: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        state_decay: torch.Tensor float32 of shape [batch, num_heads, num_chunks, chunk_size]
        diag_y: torch.Tensor float32 of shape [batch, seq_len, num_heads, head_dim]
        final_states: torch.Tensor float32 of shape [batch, num_heads, head_dim, state_dim]
        out_y: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        num_chunks: int32 scalar
        chunk_size: int32 scalar
        num_groups: int32 scalar

    Outputs:
        None
    """
    batch_size, num_heads, head_dim = intra_chunk_cumsum.shape[0], intra_chunk_cumsum.shape[1], final_states.shape[2]
    state_dim = final_states.shape[3]

    for tensor_name, tensor in (
        ("intra_chunk_cumsum", intra_chunk_cumsum),
        ("chunk_states", chunk_states),
        ("c", c),
        ("state_decay", state_decay),
        ("diag_y", diag_y),
        ("final_states", final_states),
        ("out_y", out_y),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous")

    stride_a_b, stride_a_h, stride_a_c = intra_chunk_cumsum.stride()
    stride_old_states_b, stride_old_states_c, stride_old_states_h, stride_old_states_d, stride_old_states_k = chunk_states.stride()
    stride_c_b, stride_c_l, stride_c_g, stride_c_d = c.stride()
    stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l = state_decay.stride()
    stride_final_b, stride_final_h, stride_final_d, stride_final_k = final_states.stride()
    stride_out_y_b, stride_out_y_l, stride_out_y_h, stride_out_y_p = out_y.stride()

    stride_c_c = chunk_size * stride_c_l
    stride_c_h = stride_c_g
    stride_c_n = stride_c_d
    stride_out_y_c = chunk_size * stride_out_y_l

    grid = functools.partial(
        ssd_inter_chunk_grid,
        head_dim=head_dim,
        num_heads=num_heads,
        batch_size=batch_size,
    )

    ssd_inter_chunk_scan_linear_kernel[grid](
        intra_chunk_cumsum,
        chunk_states,
        c,
        state_decay,
        diag_y,
        final_states,
        out_y,
        num_chunks,
        state_dim,
        chunk_size,
        num_groups,
        stride_a_b,
        stride_a_h,
        stride_a_c,
        stride_old_states_b,
        stride_old_states_c,
        stride_old_states_h,
        stride_old_states_d,
        stride_old_states_k,
        stride_c_b,
        stride_c_c,
        stride_c_l,
        stride_c_h,
        stride_c_n,
        stride_state_decay_b,
        stride_state_decay_h,
        stride_state_decay_c,
        stride_state_decay_l,
        stride_final_b,
        stride_final_h,
        stride_final_d,
        stride_final_k,
        stride_out_y_b,
        stride_out_y_c,
        stride_out_y_l,
        stride_out_y_h,
        stride_out_y_p,
    )


def launch_ssd_mamba2(
    x,
    a,
    b,
    c,
    out_y,
    chunk_size,
):
    """
    Launch the full Triton SSD pipeline for Mamba-2.

    Main feature:
        Runs intra-chunk and inter-chunk kernels to produce SSD outputs.

    Inputs:
        x: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, seq_len, num_heads]
        b: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        c: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        out_y: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        chunk_size: int32 scalar

    Outputs:
        final_states: torch.Tensor float32 of shape [batch, num_heads, head_dim, state_dim]
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape (batch, seq_len, num_heads, head_dim); got {x.shape}")
    if a.shape != (x.shape[0], x.shape[1], x.shape[2]):
        raise ValueError(f"a must be shape (batch, seq_len, num_heads); got {a.shape}")

    for tensor_name, tensor in (("x", x), ("a", a), ("b", b), ("c", c), ("out_y", out_y)):
        if not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous")

    batch_size, seq_len, num_heads, head_dim = x.shape
    if out_y.shape != x.shape or out_y.dtype != x.dtype:
        raise ValueError(
            f"out_y must match x shape {x.shape} and dtype {x.dtype}; got {out_y.shape} and {out_y.dtype}"
        )

    if seq_len % chunk_size != 0:
        raise ValueError(f"Sequence length {seq_len} must be divisible by chunk size {chunk_size}")
    num_chunks = seq_len // chunk_size

    num_groups = b.shape[2]
    state_dim = b.shape[3]
    if b.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"b must have leading shape (batch, seq_len)={(batch_size, seq_len)}; got {b.shape[:2]}")
    if c.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"c must have leading shape (batch, seq_len)={(batch_size, seq_len)}; got {c.shape[:2]}")
    if b.shape[2:] != c.shape[2:]:
        raise ValueError(f"b and c must share (num_groups, state_dim); got {b.shape[2:]} vs {c.shape[2:]}")

    diag_y = torch.empty_like(x, dtype=torch.float32)
    state_decay = torch.empty((batch_size, num_heads, num_chunks, chunk_size), dtype=torch.float32, device=x.device)
    chunk_states = torch.zeros(
        batch_size,
        1 + num_chunks,
        num_heads,
        head_dim,
        state_dim,
        dtype=torch.float32,
        device=x.device,
    )
    intra_chunk_cumsum = torch.zeros(
        batch_size,
        num_heads,
        1 + num_chunks,
        dtype=torch.float32,
        device=x.device,
    )

    launch_ssd_intra_chunk_kernel(
        x,
        a,
        b,
        c,
        diag_y,
        intra_chunk_cumsum,
        state_decay,
        chunk_states,
        chunk_size,
    )

    final_states = torch.empty(
        (batch_size, num_heads, head_dim, state_dim),
        device=x.device,
        dtype=torch.float32,
    )

    launch_ssd_inter_chunk_scan_linear_kernel(
        intra_chunk_cumsum,
        chunk_states,
        c,
        state_decay,
        diag_y,
        final_states,
        out_y,
        num_chunks,
        chunk_size,
        num_groups,
    )

    return final_states
