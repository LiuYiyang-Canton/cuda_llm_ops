"""Triton kernels and Python wrappers for GDN chunked forward-pass primitives."""

import torch
import triton
import triton.language as tl


@triton.jit
def local_cumsum_kernel(
    alpha_ptr,
    g_ptr,
    stride_alpha_b,
    stride_alpha_n,
    stride_alpha_c,
    stride_alpha_h,
    stride_g_b,
    stride_g_n,
    stride_g_c,
    stride_g_h,
    chunk_size: tl.constexpr,
):
    """
    Compute per-chunk cumulative sum of alpha along the chunk dimension.

    Args:
        alpha_ptr (pointer, input): [B, N, C, H] float32 alpha values.
        g_ptr (pointer, output): [B, N, C, H] float32 cumulative sums.
        stride_alpha_* (int, input): Strides for alpha in elements.
        stride_g_* (int, input): Strides for g in elements.
        chunk_size (int, input): Chunk length (C).
    """
    head_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    seq_offsets = tl.arange(0, chunk_size)
    alpha_ptrs = (
        alpha_ptr
        + batch_idx * stride_alpha_b
        + chunk_idx * stride_alpha_n
        + seq_offsets * stride_alpha_c
        + head_idx * stride_alpha_h
    )
    alpha_vals = tl.load(alpha_ptrs)
    g_vals = tl.cumsum(alpha_vals, axis=0)

    g_ptrs = (
        g_ptr
        + batch_idx * stride_g_b
        + chunk_idx * stride_g_n
        + seq_offsets * stride_g_c
        + head_idx * stride_g_h
    )
    tl.store(g_ptrs, g_vals)


@triton.jit
def compute_scaled_kkt_kernel(
    g_ptr,
    beta_ptr,
    k_ptr,
    A_ptr,
    stride_g_b,
    stride_g_n,
    stride_g_c,
    stride_g_h,
    stride_beta_b,
    stride_beta_n,
    stride_beta_c,
    stride_beta_h,
    stride_k_b,
    stride_k_n,
    stride_k_c,
    stride_k_h,
    stride_k_d,
    stride_A_b,
    stride_A_s,
    stride_A_h,
    stride_A_t,
    D_qk: tl.constexpr,
    num_heads,
    num_chunks,
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute scaled strictly-lower-triangular beta * (k k^T) with gamma from g cumsums.

    Args:
        g_ptr (pointer, input): [B, N, C, H] float32 cumulative log-decays.
        beta_ptr (pointer, input): [B, N, C, H] float32 beta values.
        k_ptr (pointer, input): [B, N, C, H, D_qk] bfloat16 key vectors.
        A_ptr (pointer, output): [B, N*C, H, C] float32 scaled lower-triangular blocks.
        stride_g_* (int, input): Strides for g in elements.
        stride_beta_* (int, input): Strides for beta in elements.
        stride_k_* (int, input): Strides for k in elements.
        stride_A_* (int, input): Strides for A in elements.
        D_qk (int, input): Key dimension.
        num_heads (int, input): Number of attention heads (H).
        num_chunks (int, input): Number of chunks (N).
        chunk_size (int, input): Chunk length (C).
        BLOCK_* (int, input): Tile sizes for the matmul.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bhn = tl.program_id(2)

    chunk_idx = pid_bhn % num_chunks
    pid_bh = pid_bhn // num_chunks
    pid_h = pid_bh % num_heads
    pid_b = pid_bh // num_heads

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = row_offsets < chunk_size
    mask_col = col_offsets < chunk_size

    g_base = g_ptr + pid_b * stride_g_b + chunk_idx * stride_g_n + pid_h * stride_g_h
    beta_base = beta_ptr + pid_b * stride_beta_b + chunk_idx * stride_beta_n + pid_h * stride_beta_h
    k_base = k_ptr + pid_b * stride_k_b + chunk_idx * stride_k_n + pid_h * stride_k_h
    A_base = A_ptr + pid_b * stride_A_b + (chunk_idx * chunk_size) * stride_A_s + pid_h * stride_A_h

    g_row = tl.load(g_base + row_offsets * stride_g_c, mask=mask_row, other=0.0)
    g_col = tl.load(g_base + col_offsets * stride_g_c, mask=mask_col, other=0.0)
    beta_row = tl.load(beta_base + row_offsets * stride_beta_c, mask=mask_row, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.static_range(0, D_qk, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D_qk

        k_row_ptrs = k_base + row_offsets[:, None] * stride_k_c + k_offsets[None, :] * stride_k_d
        k_col_ptrs = k_base + col_offsets[:, None] * stride_k_c + k_offsets[None, :] * stride_k_d
        k_row = tl.load(k_row_ptrs, mask=mask_row[:, None] & mask_k[None, :], other=0.0)
        k_col = tl.load(k_col_ptrs, mask=mask_col[:, None] & mask_k[None, :], other=0.0)

        acc += tl.dot(k_row, tl.trans(k_col), out_dtype=tl.float32, allow_tf32=False)

    gamma = tl.exp(g_row[:, None] - g_col[None, :])
    acc = acc * beta_row[:, None] * gamma

    tri_mask = row_offsets[:, None] > col_offsets[None, :]
    acc = tl.where(tri_mask, acc, 0.0)

    A_ptrs = A_base + row_offsets[:, None] * stride_A_s + col_offsets[None, :] * stride_A_t
    tl.store(A_ptrs, acc, mask=mask_row[:, None] & mask_col[None, :])


@triton.jit
def compute_w_u_kernel(
    A_ptr,
    g_ptr,
    beta_ptr,
    k_ptr,
    v_ptr,
    w_ptr,
    u_ptr,
    stride_A_b,
    stride_A_s,
    stride_A_h,
    stride_A_t,
    stride_g_b,
    stride_g_n,
    stride_g_c,
    stride_g_h,
    stride_beta_b,
    stride_beta_n,
    stride_beta_c,
    stride_beta_h,
    stride_k_b,
    stride_k_n,
    stride_k_c,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_n,
    stride_v_c,
    stride_v_h,
    stride_v_d,
    stride_w_b,
    stride_w_n,
    stride_w_c,
    stride_w_h,
    stride_w_d,
    stride_u_b,
    stride_u_n,
    stride_u_c,
    stride_u_h,
    stride_u_d,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    num_heads,
    num_chunks,
    chunk_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Compute W and U intermediates per chunk using results fromm solve_tril.

    Args:
        A_ptr (pointer, input): [B, N*C, H, C] float32 lower-triangular blocks.
        g_ptr (pointer, input): [B, N, C, H] float32 cumulative log-decays.
        beta_ptr (pointer, input): [B, N, C, H] float32 beta values.
        k_ptr (pointer, input): [B, N, C, H, D_qk] bfloat16 key vectors.
        v_ptr (pointer, input): [B, N, C, H, D_v] bfloat16 value vectors.
        w_ptr (pointer, output): [B, N, C, H, D_qk] float32 W blocks.
        u_ptr (pointer, output): [B, N, C, H, D_v] float32 U blocks.
        stride_* (int, input): Strides for the corresponding tensors in elements.
        D_qk (int, input): Key dimension.
        D_v (int, input): Value dimension.
        num_heads (int, input): Number of attention heads (H).
        num_chunks (int, input): Number of chunks (N).
        chunk_size (int, input): Chunk length (C).
        BLOCK_* (int, input): Tile sizes for the matmuls.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bhn = tl.program_id(2)

    chunk_idx = pid_bhn % num_chunks
    pid_bh = pid_bhn // num_chunks
    pid_h = pid_bh % num_heads
    pid_b = pid_bh // num_heads

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = row_offsets < chunk_size
    mask_col_w = col_offsets < D_qk
    mask_col_u = col_offsets < D_v

    A_base = (
        A_ptr
        + pid_b * stride_A_b
        + pid_h * stride_A_h
        + (chunk_idx * chunk_size) * stride_A_s
    )
    g_base = g_ptr + pid_b * stride_g_b + chunk_idx * stride_g_n + pid_h * stride_g_h
    beta_base = beta_ptr + pid_b * stride_beta_b + chunk_idx * stride_beta_n + pid_h * stride_beta_h
    k_base = k_ptr + pid_b * stride_k_b + chunk_idx * stride_k_n + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + chunk_idx * stride_v_n + pid_h * stride_v_h
    w_base = w_ptr + pid_b * stride_w_b + chunk_idx * stride_w_n + pid_h * stride_w_h
    u_base = u_ptr + pid_b * stride_u_b + chunk_idx * stride_u_n + pid_h * stride_u_h

    acc_w = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.static_range(0, chunk_size, BLOCK_C):
        c_offsets = k_start + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < chunk_size

        a_ptrs = A_base + row_offsets[:, None] * stride_A_s + c_offsets[None, :] * stride_A_t
        a_vals = tl.load(a_ptrs, mask=mask_row[:, None] & mask_c[None, :], other=0.0).to(tl.bfloat16)

        g_ptrs = g_base + c_offsets * stride_g_c
        beta_ptrs = beta_base + c_offsets * stride_beta_c
        g_vals = tl.load(g_ptrs, mask=mask_c, other=0.0)
        beta_vals = tl.load(beta_ptrs, mask=mask_c, other=0.0)

        scale_w = tl.exp(g_vals) * beta_vals

        k_ptrs = k_base + c_offsets[:, None] * stride_k_c + col_offsets[None, :] * stride_k_d
        k_vals = tl.load(k_ptrs, mask=mask_c[:, None] & mask_col_w[None, :], other=0.0)
        k_vals = k_vals.to(tl.float32) * scale_w[:, None]

        v_ptrs = v_base + c_offsets[:, None] * stride_v_c + col_offsets[None, :] * stride_v_d
        v_vals = tl.load(v_ptrs, mask=mask_c[:, None] & mask_col_u[None, :], other=0.0)
        v_vals = v_vals.to(tl.float32) * beta_vals[:, None]

        acc_w += tl.dot(a_vals, k_vals.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)
        acc_u += tl.dot(a_vals, v_vals.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)

    w_ptrs = w_base + row_offsets[:, None] * stride_w_c + col_offsets[None, :] * stride_w_d
    u_ptrs = u_base + row_offsets[:, None] * stride_u_c + col_offsets[None, :] * stride_u_d
    tl.store(w_ptrs, acc_w, mask=mask_row[:, None] & mask_col_w[None, :])
    tl.store(u_ptrs, acc_u, mask=mask_row[:, None] & mask_col_u[None, :])


@triton.jit
def compute_states_kernel(
    u_ptr,
    w_ptr,
    g_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    stride_u_b,
    stride_u_n,
    stride_u_c,
    stride_u_h,
    stride_u_v,
    stride_w_b,
    stride_w_n,
    stride_w_c,
    stride_w_h,
    stride_w_k,
    stride_g_b,
    stride_g_n,
    stride_g_c,
    stride_g_h,
    stride_k_b,
    stride_k_n,
    stride_k_c,
    stride_k_h,
    stride_k_k,
    stride_vn_b,
    stride_vn_n,
    stride_vn_c,
    stride_vn_h,
    stride_vn_v,
    stride_h_b,
    stride_h_n,
    stride_h_h,
    stride_h_k,
    stride_h_v,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    num_heads,
    chunk_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Compute v_new and recurrent states h across chunks.

    Args:
        u_ptr (pointer, input): [B, N, C, H, D_v] float32 U blocks.
        w_ptr (pointer, input): [B, N, C, H, D_qk] float32 W blocks.
        g_ptr (pointer, input): [B, N, C, H] float32 cumulative log-decays.
        k_ptr (pointer, input): [B, N, C, H, D_qk] bfloat16 key vectors.
        v_new_ptr (pointer, output): [B, N, C, H, D_v] bfloat16 updated values.
        h_ptr (pointer, output): [B, N, H, D_qk, D_v] float32 recurrent states.
        stride_* (int, input): Strides for the corresponding tensors in elements.
        D_qk (int, input): Key dimension.
        D_v (int, input): Value dimension.
        num_heads (int, input): Number of attention heads (H).
        chunk_size (int, input): Chunk length (C).
        num_chunks (int, input): Number of chunks (N).
        BLOCK_K (int, input): Tile size for D_qk.
        BLOCK_V (int, input): Tile size for D_v.
    """
    pid_v = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_bh = tl.program_id(2)
    pid_h = pid_bh % num_heads
    pid_b = pid_bh // num_heads

    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = tl.arange(0, BLOCK_V)

    u_base = u_ptr + pid_b * stride_u_b + pid_h * stride_u_h + pid_v * BLOCK_V * stride_u_v
    w_base = w_ptr + pid_b * stride_w_b + pid_h * stride_w_h + pid_k * BLOCK_K * stride_w_k
    g_base = g_ptr + pid_b * stride_g_b + pid_h * stride_g_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h + pid_k * BLOCK_K * stride_k_k
    v_new_base = v_new_ptr + pid_b * stride_vn_b + pid_h * stride_vn_h + pid_v * BLOCK_V * stride_vn_v
    h_base = h_ptr + pid_b * stride_h_b + pid_h * stride_h_h + pid_k * BLOCK_K * stride_h_k + pid_v * BLOCK_V * stride_h_v

    seq_offsets = tl.arange(0, chunk_size)

    # initialize old_states
    old_states = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk_idx in tl.range(0, num_chunks):
        u_ptrs = u_base + chunk_idx * stride_u_n + seq_offsets[:, None] * stride_u_c + v_offsets[None, :] * stride_u_v
        u_curr = tl.load(u_ptrs, mask = True, other = 0.) # (chunk_size, BLOCK_V)

        w_ptrs = w_base + chunk_idx * stride_w_n + seq_offsets[:, None] * stride_w_c + k_offsets[None, :] * stride_w_k
        w_curr = tl.load(w_ptrs, mask = True, other = 0.).to(tl.bfloat16) # (chunk_size, D_qk)

        # v_new = u - w @ s
        v_new_ptrs = v_new_base + chunk_idx * stride_vn_n + seq_offsets[:, None] * stride_vn_c + v_offsets[None, :] * stride_vn_v
        u_curr -= tl.dot(w_curr, old_states.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32 = False)
        tl.store(v_new_ptrs, u_curr.to(tl.bfloat16), mask = True)

        g_ptrs = g_base + chunk_idx * stride_g_n + seq_offsets * stride_g_c
        g_curr = tl.load(g_ptrs, mask = True) # (chunk_size,)
        g_last_ptr = g_base + chunk_idx * stride_g_n + (chunk_size - 1) * stride_g_c
        g_last_curr = tl.load(g_last_ptr) # (1,)
        g_last_exp = tl.exp(g_last_curr)
        
        k_ptrs = k_base + chunk_idx * stride_k_n + seq_offsets[:, None] * stride_k_c + k_offsets[None, :] * stride_k_k
        k_curr = tl.load(k_ptrs, mask = True) # (chunk_size, D_qk)
        k_decay_curr = tl.exp(g_last_curr - g_curr)[:, None] * k_curr

        # compute new states
        old_states *= g_last_exp # (D_qk, BLOCK_V)
        old_states += tl.dot(tl.trans(k_decay_curr.to(tl.bfloat16)), u_curr.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False) # (D_qk, BLOCK_V)

        h_ptrs = h_base + chunk_idx * stride_h_n + k_offsets[:, None] * stride_h_k + v_offsets[None, :] * stride_h_v
        tl.store(h_ptrs, old_states, mask=True)

@triton.jit
def compute_output_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    g_ptr,
    o_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_c,
    stride_q_h,
    stride_q_k,
    stride_k_b,
    stride_k_n,
    stride_k_c,
    stride_k_h,
    stride_k_k,
    stride_vn_b,
    stride_vn_n,
    stride_vn_c,
    stride_vn_h,
    stride_vn_v,
    stride_h_b,
    stride_h_n,
    stride_h_h,
    stride_h_k,
    stride_h_v,
    stride_g_b,
    stride_g_n,
    stride_g_c,
    stride_g_h,
    stride_o_b,
    stride_o_n,
    stride_o_c,
    stride_o_h,
    stride_o_v,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    num_heads,
    chunk_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Compute output for each chunk using local attention and previous state.

    Args:
        q_ptr (pointer, input): [B, N, C, H, D_qk] bfloat16 query vectors.
        k_ptr (pointer, input): [B, N, C, H, D_qk] bfloat16 key vectors.
        v_new_ptr (pointer, input): [B, N, C, H, D_v] bfloat16 updated values.
        h_ptr (pointer, input): [B, N, H, D_qk, D_v] float32 recurrent states.
        g_ptr (pointer, input): [B, N, C, H] float32 cumulative log-decays.
        o_ptr (pointer, output): [B, N*C, H, D_v] bfloat16 outputs.
        stride_* (int, input): Strides for the corresponding tensors in elements.
        D_qk (int, input): Key dimension.
        D_v (int, input): Value dimension.
        num_heads (int, input): Number of attention heads (H).
        chunk_size (int, input): Chunk length (C).
        BLOCK_V (int, input): Tile size for D_v.
    """
    pid_v = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_bh = tl.program_id(2)
    pid_h = pid_bh % num_heads
    pid_b = pid_bh // num_heads

    seq_offsets = tl.arange(0, chunk_size)
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, D_qk)
    mask_v = v_offsets < D_v

    q_base = q_ptr + pid_b * stride_q_b + chunk_idx * stride_q_n + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + chunk_idx * stride_k_n + pid_h * stride_k_h
    v_base = v_new_ptr + pid_b * stride_vn_b + chunk_idx * stride_vn_n + pid_h * stride_vn_h
    g_base = g_ptr + pid_b * stride_g_b + chunk_idx * stride_g_n + pid_h * stride_g_h
    o_base = o_ptr + pid_b * stride_o_b + chunk_idx * stride_o_n + pid_h * stride_o_h

    q_ptrs = q_base + seq_offsets[:, None] * stride_q_c + k_offsets[None, :] * stride_q_k
    k_ptrs = k_base + seq_offsets[:, None] * stride_k_c + k_offsets[None, :] * stride_k_k
    q_vals = tl.load(q_ptrs, mask=True)
    k_vals = tl.load(k_ptrs, mask=True)

    g_vals = tl.load(g_base + seq_offsets * stride_g_c, mask=True)

    qk = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32, allow_tf32=False)
    gamma = tl.exp(g_vals[:, None] - g_vals[None, :])
    gamma = tl.where(seq_offsets[:, None] >= seq_offsets[None, :], gamma, 0.0)
    qk *= gamma

    v_ptrs = v_base + seq_offsets[:, None] * stride_vn_c + v_offsets[None, :] * stride_vn_v
    v_vals = tl.load(v_ptrs, mask=mask_v[None, :], other=0.0)
    o_vals = tl.dot(qk.to(tl.bfloat16), v_vals, out_dtype=tl.float32, allow_tf32=False)

    if chunk_idx > 0:
        q_decay = tl.exp(g_vals)[:, None] * q_vals.to(tl.float32)
        h_base = h_ptr + pid_b * stride_h_b + (chunk_idx - 1) * stride_h_n + pid_h * stride_h_h
        h_ptrs = h_base + k_offsets[:, None] * stride_h_k + v_offsets[None, :] * stride_h_v
        h_vals = tl.load(h_ptrs, mask=mask_v[None, :], other=0.0)
        o_vals += tl.dot(q_decay.to(tl.bfloat16), h_vals.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)

    o_ptrs = o_base + seq_offsets[:, None] * stride_o_c + v_offsets[None, :] * stride_o_v
    tl.store(o_ptrs, o_vals.to(tl.bfloat16), mask=mask_v[None, :])


def compute_local_cumsum(alpha_chunks: torch.Tensor) -> torch.Tensor:
    """
    Compute per-chunk cumulative sums of alpha using Triton.

    Args:
        alpha_chunks (torch.Tensor): [B, N, C, H] float32 contiguous CUDA tensor.

    Returns:
        torch.Tensor: [B, N, C, H] float32 cumulative sums.
    """
    if alpha_chunks.ndim != 4:
        raise ValueError(f"alpha_chunks must be 4D, got {alpha_chunks.ndim}D")
    if not alpha_chunks.is_contiguous():
        raise ValueError("alpha_chunks must be contiguous")
    if not alpha_chunks.is_cuda:
        raise ValueError("alpha_chunks must be a CUDA tensor")

    batch, num_chunks, chunk_size, num_heads = alpha_chunks.shape
    g = torch.empty_like(alpha_chunks)

    grid = (num_heads, num_chunks, batch)
    local_cumsum_kernel[grid](
        alpha_chunks,
        g,
        *alpha_chunks.stride(),
        *g.stride(),
        chunk_size=chunk_size,
        num_warps=4,
        num_stages=3,
    )
    return g


def compute_scaled_kkt(
    g: torch.Tensor,
    beta_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute scaled lower-triangular KKT blocks in Triton.

    Args:
        g (torch.Tensor): [B, N, C, H] float32 cumulative log-decays.
        beta_chunks (torch.Tensor): [B, N, C, H] float32 beta values.
        k_chunks (torch.Tensor): [B, N, C, H, D_qk] bfloat16 key vectors.
        chunk_size (int): Chunk length (C).

    Returns:
        torch.Tensor: [B, N*C, H, C] float32 scaled lower-triangular blocks.
    """
    if g.ndim != 4:
        raise ValueError(f"g must be 4D, got {g.ndim}D")
    if beta_chunks.ndim != 4:
        raise ValueError(f"beta_chunks must be 4D, got {beta_chunks.ndim}D")
    if k_chunks.ndim != 5:
        raise ValueError(f"k_chunks must be 5D, got {k_chunks.ndim}D")
    if not (g.is_cuda and beta_chunks.is_cuda and k_chunks.is_cuda):
        raise ValueError("g, beta_chunks, and k_chunks must be CUDA tensors")

    B, num_chunks, local_chunk, H = g.shape
    if local_chunk != chunk_size:
        raise ValueError(f"chunk_size mismatch: got {chunk_size}, expected {local_chunk}")
    if beta_chunks.shape != g.shape:
        raise ValueError(f"beta_chunks shape mismatch: {beta_chunks.shape} vs {g.shape}")
    if k_chunks.shape[:4] != (B, num_chunks, chunk_size, H):
        raise ValueError(f"k_chunks shape mismatch: {k_chunks.shape}")

    D_qk = k_chunks.shape[-1]
    A = torch.empty((B, num_chunks * chunk_size, H, chunk_size), device=g.device, dtype=torch.float32)

    block_m = 64
    block_n = 64
    block_k = 64

    grid = (triton.cdiv(chunk_size, block_m), triton.cdiv(chunk_size, block_n), B * H * num_chunks)
    compute_scaled_kkt_kernel[grid](
        g,
        beta_chunks,
        k_chunks,
        A,
        *g.stride(),
        *beta_chunks.stride(),
        *k_chunks.stride(),
        *A.stride(),
        D_qk=D_qk,
        num_heads=H,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )

    return A


def compute_w_u(
    A: torch.Tensor,
    g: torch.Tensor,
    beta_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    v_chunks: torch.Tensor,
    chunk_size: int,
):
    """
    Compute W and U intermediates for each chunk using Triton.

    Args:
        A (torch.Tensor): [B, N*C, H, C] float32 lower-triangular blocks.
        g (torch.Tensor): [B, N, C, H] float32 cumulative log-decays.
        beta_chunks (torch.Tensor): [B, N, C, H] float32 beta values.
        k_chunks (torch.Tensor): [B, N, C, H, D_qk] bfloat16 key vectors.
        v_chunks (torch.Tensor): [B, N, C, H, D_v] bfloat16 value vectors.
        chunk_size (int): Chunk length (C).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            w: [B, N, C, H, D_qk] float32.
            u: [B, N, C, H, D_v] float32.
    """
    if A.ndim != 4:
        raise ValueError(f"A must be 4D, got {A.ndim}D")
    if g.ndim != 4:
        raise ValueError(f"g must be 4D, got {g.ndim}D")
    if beta_chunks.ndim != 4:
        raise ValueError(f"beta_chunks must be 4D, got {beta_chunks.ndim}D")
    if k_chunks.ndim != 5:
        raise ValueError(f"k_chunks must be 5D, got {k_chunks.ndim}D")
    if v_chunks.ndim != 5:
        raise ValueError(f"v_chunks must be 5D, got {v_chunks.ndim}D")
    if not (A.is_cuda and g.is_cuda and beta_chunks.is_cuda and k_chunks.is_cuda and v_chunks.is_cuda):
        raise ValueError("All inputs to compute_w_u must be CUDA tensors")

    B, num_chunks, local_chunk, H = g.shape
    if local_chunk != chunk_size:
        raise ValueError(f"chunk_size mismatch: got {chunk_size}, expected {local_chunk}")
    if beta_chunks.shape != g.shape:
        raise ValueError(f"beta_chunks shape mismatch: {beta_chunks.shape} vs {g.shape}")
    if k_chunks.shape[:4] != (B, num_chunks, chunk_size, H):
        raise ValueError(f"k_chunks shape mismatch: {k_chunks.shape}")
    if v_chunks.shape[:4] != (B, num_chunks, chunk_size, H):
        raise ValueError(f"v_chunks shape mismatch: {v_chunks.shape}")

    expected_s = num_chunks * chunk_size
    if A.shape != (B, expected_s, H, chunk_size):
        raise ValueError(f"A shape mismatch: {A.shape}, expected {(B, expected_s, H, chunk_size)}")

    D_qk = k_chunks.shape[-1]
    D_v = v_chunks.shape[-1]

    w = torch.empty((B, num_chunks, chunk_size, H, D_qk), device=A.device, dtype=torch.float32)
    u = torch.empty((B, num_chunks, chunk_size, H, D_v), device=A.device, dtype=torch.float32)

    block_m = 64
    block_n = 64
    block_c = 64

    max_d = D_qk if D_qk >= D_v else D_v
    grid = (triton.cdiv(chunk_size, block_m), triton.cdiv(max_d, block_n), B * H * num_chunks)
    compute_w_u_kernel[grid](
        A,
        g,
        beta_chunks,
        k_chunks,
        v_chunks,
        w,
        u,
        *A.stride(),
        *g.stride(),
        *beta_chunks.stride(),
        *k_chunks.stride(),
        *v_chunks.stride(),
        *w.stride(),
        *u.stride(),
        D_qk=D_qk,
        D_v=D_v,
        num_heads=H,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=3,
    )

    return w, u


def compute_states(
    u: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    k_chunks: torch.Tensor,
    chunk_size: int,
):
    """
    Compute v_new and recurrent states h across chunks using Triton.

    Args:
        u (torch.Tensor): [B, N, C, H, D_v] float32 U blocks.
        w (torch.Tensor): [B, N, C, H, D_qk] float32 W blocks.
        g (torch.Tensor): [B, N, C, H] float32 cumulative log-decays.
        k_chunks (torch.Tensor): [B, N, C, H, D_qk] bfloat16 key vectors.
        chunk_size (int): Chunk length (C).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h: [B, N, H, D_qk, D_v] float32 recurrent states.
            v_new: [B, N, C, H, D_v] bfloat16 updated values.
            final_state: [B, H, D_qk, D_v] float32 final state from last chunk.
    """
    if u.ndim != 5:
        raise ValueError(f"u must be 5D, got {u.ndim}D")
    if w.ndim != 5:
        raise ValueError(f"w must be 5D, got {w.ndim}D")
    if g.ndim != 4:
        raise ValueError(f"g must be 4D, got {g.ndim}D")
    if k_chunks.ndim != 5:
        raise ValueError(f"k_chunks must be 5D, got {k_chunks.ndim}D")
    if not (u.is_cuda and w.is_cuda and g.is_cuda and k_chunks.is_cuda):
        raise ValueError("u, w, g, and k_chunks must be CUDA tensors")

    B, num_chunks, _, H, D_v = u.shape
    D_qk = w.shape[-1]

    v_new = torch.empty(B, num_chunks, chunk_size, H, D_v, device=u.device, dtype=torch.bfloat16)
    h = torch.empty(
        (B, num_chunks, H, D_qk, D_v),
        device=u.device,
        dtype=torch.float32,
    )

    # make sure that D_v is multiples of 16
    assert D_v % 16 == 0, "D_v must be a multiple of 16"

    # make sure that chunk_size and D_qk is multiples of 16
    assert chunk_size % 16 == 0, "chunk_size must be a multiple of 16"
    assert D_qk % 16 == 0, "D_qk must be a multiple of 16"

    block_v = 16
    block_k = D_qk

    assert D_qk == block_k, "D_qk must be equal to block_k"

    grid = (triton.cdiv(D_v, block_v), triton.cdiv(D_qk, block_k), B * H)
    compute_states_kernel[grid](
        u,
        w,
        g,
        k_chunks,
        v_new,
        h,
        *u.stride(),
        *w.stride(),
        *g.stride(),
        *k_chunks.stride(),
        *v_new.stride(),
        *h.stride(),
        D_qk=D_qk,
        D_v=D_v,
        num_heads=H,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=3,
    )

    final_state = h[:, -1]
    return h, v_new, final_state


def compute_output(
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute the final output tensor using Triton.

    Args:
        q_chunks (torch.Tensor): [B, N, C, H, D_qk] bfloat16 query vectors.
        k_chunks (torch.Tensor): [B, N, C, H, D_qk] bfloat16 key vectors.
        v_new (torch.Tensor): [B, N, C, H, D_v] bfloat16 updated values.
        h (torch.Tensor): [B, N, H, D_qk, D_v] float32 recurrent states.
        g (torch.Tensor): [B, N, C, H] float32 cumulative log-decays.
        chunk_size (int): Chunk length (C).

    Returns:
        torch.Tensor: [B, N*C, H, D_v] bfloat16 outputs.
    """
    if q_chunks.ndim != 5:
        raise ValueError(f"q_chunks must be 5D, got {q_chunks.ndim}D")
    if k_chunks.ndim != 5:
        raise ValueError(f"k_chunks must be 5D, got {k_chunks.ndim}D")
    if v_new.ndim != 5:
        raise ValueError(f"v_new must be 5D, got {v_new.ndim}D")
    if h.ndim != 5:
        raise ValueError(f"h must be 5D, got {h.ndim}D")
    if g.ndim != 4:
        raise ValueError(f"g must be 4D, got {g.ndim}D")
    if not (q_chunks.is_cuda and k_chunks.is_cuda and v_new.is_cuda and h.is_cuda and g.is_cuda):
        raise ValueError("All inputs to compute_output must be CUDA tensors")

    B, num_chunks, local_chunk, H, D_qk = q_chunks.shape
    if local_chunk != chunk_size:
        raise ValueError(f"chunk_size mismatch: got {chunk_size}, expected {local_chunk}")
    D_v = v_new.shape[-1]

    assert chunk_size % 16 == 0, "chunk_size must be a multiple of 16"
    assert D_qk % 16 == 0, "D_qk must be a multiple of 16"

    block_v = 64
    o = torch.empty((B, num_chunks * chunk_size, H, D_v), device=q_chunks.device, dtype=torch.bfloat16)
    o_chunks = o.view(B, num_chunks, chunk_size, H, D_v)

    grid = (triton.cdiv(D_v, block_v), num_chunks, B * H)
    compute_output_kernel[grid](
        q_chunks,
        k_chunks,
        v_new,
        h,
        g,
        o,
        *q_chunks.stride(),
        *k_chunks.stride(),
        *v_new.stride(),
        *h.stride(),
        *g.stride(),
        *o_chunks.stride(),
        D_qk=D_qk,
        D_v=D_v,
        num_heads=H,
        chunk_size=chunk_size,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=3,
    )

    return o
