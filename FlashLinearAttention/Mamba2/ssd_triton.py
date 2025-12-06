"""Triton implementation of Mamba-2 SSD (state space duality) linear attention."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from einops import rearrange, repeat


# Copyright (c) 2024, Albert Gu and Tri Dao. Copied from the Mamba repository (https://github.com/state-spaces/mamba).
# Reference-only: kept for correctness verification, not used in the Triton implementation below.
def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


# Copyright (c) 2024, Albert Gu and Tri Dao. Copied from the Mamba repository (https://github.com/state-spaces/mamba).
# Reference-only: kept for correctness verification, not used in the Triton implementation below.
def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HEAD": 8}, num_stages=2, num_warps=4),
    ],
    key=["head_dim", "state_dim"],
)
@triton.jit
def ssd_intra_chunk_kernel(
    x_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    out_y_ptr,
    out_A_cumsum_last_ptr,
    out_state_decay_ptr,
    out_states_ptr,
    seq_len,
    num_heads,
    num_groups,
    # Strides
    stride_x_b, stride_x_l, stride_x_h, stride_x_d,
    stride_A_b, stride_A_l, stride_A_h,
    stride_B_b, stride_B_l, stride_B_g, stride_B_d,
    stride_C_b, stride_C_l, stride_C_g, stride_C_d,
    stride_out_y_b, stride_out_y_l, stride_out_y_h, stride_out_y_d,
    stride_out_A_cumsum_b, stride_out_A_cumsum_h, stride_out_A_cumsum_c,
    stride_out_state_decay_b, stride_out_state_decay_h, stride_out_state_decay_c, stride_out_state_decay_l,
    stride_out_states_b, stride_out_states_c, stride_out_states_h, stride_out_states_d, stride_out_states_k,
    state_dim: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """
    Compute intra-chunk SSD contributions: diagonal outputs, per-step decays, and chunk boundary states.

    Args:
        x_ptr (pointer, input): [batch, seq_len, num_heads, head_dim] input tensor.
        A_ptr (pointer, input): [batch, seq_len, num_heads] decay values in log space.
        B_ptr (pointer, input): [batch, seq_len, num_groups, state_dim] state-projection matrix.
        C_ptr (pointer, input): [batch, seq_len, num_groups, state_dim] output-projection matrix.
        out_y_ptr (pointer, output): [batch, seq_len, num_heads, head_dim] intra-chunk output accumulator.
        out_A_cumsum_last_ptr (pointer, output): [batch, num_heads, 1 + num_chunks] last cumulative decay per chunk.
        out_state_decay_ptr (pointer, output): [batch, num_heads, num_chunks, chunk_size] per-step state decay.
        out_states_ptr (pointer, output): [batch, 1 + num_chunks, num_heads, head_dim, state_dim] chunk boundary states.
        seq_len (int, input): Sequence length.
        num_heads (int, input): Total attention heads.
        num_groups (int, input): Group count for shared B/C projections.
        stride_* (int, input): Strides for the corresponding tensors.
        state_dim (int, input): Dimension of the recurrent state.
        head_dim (int, input): Dimension of each attention head.
        chunk_size (int, input): Number of timesteps per chunk.
        BLOCK_HEAD (int, input): Number of heads processed per Triton program.
    """
    chunk_idx = tl.program_id(0)  # chunk along sequence
    head_block_idx = tl.program_id(1)  # head block
    batch_idx = tl.program_id(2)  # batch index

    # chunk start index
    chunk_start = chunk_idx * chunk_size

    seq_offsets = tl.arange(0, chunk_size)
    head_offsets = tl.arange(0, head_dim)
    valid_mask = (chunk_start + seq_offsets) < seq_len
    state_offsets = tl.arange(0, state_dim)

    # Load B, shape is (chunk_size, state_dim)
    B_ptrs = B_ptr + batch_idx * stride_B_b + chunk_start * stride_B_l + ((head_block_idx * BLOCK_HEAD) % num_groups) * stride_B_g
    B = tl.load(
        B_ptrs + seq_offsets[:, None] * stride_B_l + state_offsets[None, :] * stride_B_d,
        mask=valid_mask[:, None],
        other=0.0,
    )

    # load C, shape is (chunk_size, state_dim)
    C_ptrs = C_ptr + batch_idx * stride_C_b + chunk_start * stride_C_l + ((head_block_idx * BLOCK_HEAD) % num_groups) * stride_C_g
    C = tl.load(
        C_ptrs + seq_offsets[:, None] * stride_C_l + state_offsets[None, :] * stride_C_d,
        mask=valid_mask[:, None],
        other=0.0,
    )

    # compute C^T B, bf16 matmul
    CTB = tl.dot(C, tl.trans(B), out_dtype=tl.float32, allow_tf32=False)  # (chunk_size, chunk_size)

    rows = tl.arange(0, chunk_size)
    cols = tl.arange(0, chunk_size)

    for head_idx in range(head_block_idx * BLOCK_HEAD, (head_block_idx + 1) * BLOCK_HEAD):
        # load x, shape is (chunk_size, head_dim)
        x_ptrs = x_ptr + batch_idx * stride_x_b + chunk_start * stride_x_l + head_idx * stride_x_h
        x = tl.load(
            x_ptrs + seq_offsets[:, None] * stride_x_l + head_offsets[None, :] * stride_x_d,
            mask=valid_mask[:, None],
            other=0.0,
        )

        # load A, shape is (chunk_size,)
        A_ptrs = A_ptr + batch_idx * stride_A_b + chunk_start * stride_A_l + head_idx * stride_A_h
        A = tl.load(A_ptrs + seq_offsets * stride_A_l, mask=valid_mask, other=0.0)

        A_cumsum = tl.cumsum(A, axis=0)
        L = tl.exp(A_cumsum[:, None] - A_cumsum[None, :])
        L = tl.where(rows[:, None] >= cols[None, :], L, 0.0)

        state_decay = tl.exp(A_cumsum)

        # elementwise multiply L and CTB
        L_CTB = L * CTB  # (chunk_size, chunk_size)

        # compute Y_diag = L_CTB @ X, bf16 matmul
        Y_diag = tl.dot(L_CTB.to(tl.bfloat16), x, out_dtype=tl.float32) # (chunk_size, head_dim)

        # store Y_diag
        out_y_ptrs = out_y_ptr + batch_idx * stride_out_y_b + chunk_start * stride_out_y_l + head_idx * stride_out_y_h
        tl.store(
            out_y_ptrs + seq_offsets[:, None] * stride_out_y_l + head_offsets[None, :] * stride_out_y_d,
            Y_diag,
            mask=valid_mask[:, None],
        )

        # compute A_cumsum[-1:] - A_cumsum
        A_cumsum_last = tl.sum(tl.where(seq_offsets == chunk_size - 1, A_cumsum, 0.0)) # scalar
        decay_states = tl.exp(A_cumsum_last - A_cumsum)  # (chunk_size,)

        # compute B * decay_states, elementwise multiply
        B_decay = B * decay_states[:, None]  # (chunk_size, state_dim)

        # compute states = B_decay @ X, bf16 matmul, output shape is (state_dim, head_dim)
        states = tl.dot(tl.trans(B_decay.to(tl.bfloat16)), x, out_dtype=tl.float32)

        # store state_decay (global shape is (batch, num_heads, num_chunks, chunk_size))
        state_decay_ptrs = (
            out_state_decay_ptr
            + batch_idx * stride_out_state_decay_b
            + head_idx * stride_out_state_decay_h
            + chunk_idx * stride_out_state_decay_c
        )
        tl.store(state_decay_ptrs + seq_offsets * stride_out_state_decay_l, state_decay, mask=valid_mask)

        # store states to out_states_ptr, global shape is (batch, num_chunks, num_heads, head_dim, state_dim)
        # shift pid_chunk by one to account for initial zero state
        out_states_ptrs = (
            out_states_ptr
            + batch_idx * stride_out_states_b
            + (1 + chunk_idx) * stride_out_states_c
            + head_idx * stride_out_states_h
            + head_offsets[:, None] * stride_out_states_d
            + state_offsets[None, :] * stride_out_states_k
        )
        tl.store(out_states_ptrs, tl.trans(states), mask=True)

        # store last element of A_cumsum to out_A_cumsum_last_ptr (global shape is (batch, num_heads, 1 + num_chunks))
        out_A_cumsum_last_ptrs = (
            out_A_cumsum_last_ptr
            + batch_idx * stride_out_A_cumsum_b
            + head_idx * stride_out_A_cumsum_h
            + (1 + chunk_idx) * stride_out_A_cumsum_c
        )
        tl.store(out_A_cumsum_last_ptrs, A_cumsum_last, mask=True) 

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HEADDIM": 16}),
    ],
    key=["head_dim", "state_dim"],
)
@triton.jit
def ssd_inter_chunk_scan_linear_kernel(
    A_intra_sum_ptr,      # (batch, num_heads, 1 + num_chunks)
    old_states_ptr,       # (batch, 1 + num_chunks, num_heads, head_dim, state_dim)
    C_ptr,                # (batch, num_chunks, chunk_size, num_groups, state_dim)
    state_decay_ptr,      # (batch, num_heads, num_chunks, chunk_size)
    diag_y_ptr,           # (batch, num_chunks, chunk_size, num_heads, head_dim)
    final_state_ptr,      # (batch, num_heads, head_dim, state_dim)
    out_y_ptr,            # (batch, num_chunks, chunk_size, num_heads, head_dim)
    num_chunks,
    state_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    num_groups,
    # Strides
    stride_A_b, stride_A_h, stride_A_c,
    stride_old_states_b, stride_old_states_c, stride_old_states_h, stride_old_states_d, stride_old_states_k,
    stride_C_b, stride_C_c, stride_C_l, stride_C_h, stride_C_n,
    stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l,
    stride_final_b, stride_final_h, stride_final_d, stride_final_k,
    stride_out_y_b, stride_out_y_c, stride_out_y_l, stride_out_y_h, stride_out_y_p, 
    # BLOCKS
    BLOCK_HEADDIM: tl.constexpr
):
    """
    Scan across chunks to decay and accumulate states, then mix with intra-chunk outputs to form the full SSD output and final state.

    Args:
        A_intra_sum_ptr (pointer, input): [batch, num_heads, 1 + num_chunks] cumulative decay per chunk.
        old_states_ptr (pointer, input): [batch, 1 + num_chunks, num_heads, head_dim, state_dim] states from intra-kernel.
        C_ptr (pointer, input): [batch, num_chunks, chunk_size, num_groups, state_dim] projection matrix C.
        state_decay_ptr (pointer, input): [batch, num_heads, num_chunks, chunk_size] per-step state decay.
        diag_y_ptr (pointer, input): [batch, num_chunks, chunk_size, num_heads, head_dim] intra-chunk outputs.
        final_state_ptr (pointer, output): [batch, num_heads, head_dim, state_dim] final recurrent state.
        out_y_ptr (pointer, output): [batch, num_chunks, chunk_size, num_heads, head_dim] full SSD output.
        num_chunks (int, input): Number of chunks in the sequence.
        state_dim (int, input): Dimension of the recurrent state.
        chunk_size (int, input): Number of steps per chunk.
        num_groups (int, input): Group count for shared B/C projections.
        stride_* (int, input): Strides for the corresponding tensors.
        BLOCK_HEADDIM (int, input): Number of head_dim elements processed per program.
    """
    # Map program ID to a specific (Batch, Head, HeadDim) stream
    head_dim_idx = tl.program_id(0) * BLOCK_HEADDIM  # head_dim offset
    head_idx = tl.program_id(1)  # head index
    batch_idx = tl.program_id(2)  # batch index

    # offsets for chunk_size, head_dim, and state_dim
    chunk_offsets = tl.arange(0, chunk_size)
    head_dim_offsets = tl.arange(0, BLOCK_HEADDIM)
    state_offsets = tl.arange(0, state_dim)

    # Initial Pointer setup
    # Point to the specific sequence in A_cumsum (batch, head)
    ptr_A = A_intra_sum_ptr + batch_idx * stride_A_b + head_idx * stride_A_h
    
    # Point to the start of the sequence in states/new_states (batch, 0, head, head_dim, 0)
    # We will increment the chunk pointer manually in the loop
    base_offset = (batch_idx * stride_old_states_b) + \
                  (head_idx * stride_old_states_h) + \
                  (head_dim_idx * stride_old_states_d)
    
    old_states_ptrs = old_states_ptr + base_offset
    final_offset = (batch_idx * stride_final_b) + \
                  (head_idx * stride_final_h) + \
                  (head_dim_idx * stride_final_d)
    final_ptrs = final_state_ptr + final_offset

    C_offset = (batch_idx * stride_C_b) + ((head_idx % num_groups) * stride_C_h)
    C_ptrs = C_ptr + C_offset

    state_decay_offset = (batch_idx * stride_state_decay_b) + \
                         (head_idx * stride_state_decay_h)
    state_decay_ptrs = state_decay_ptr + state_decay_offset

    out_y_offset = (batch_idx * stride_out_y_b) + (head_idx * stride_out_y_h) + \
                   (head_dim_idx * stride_out_y_p)
    diag_y_ptrs = diag_y_ptr + out_y_offset
    out_y_ptrs = out_y_ptr + out_y_offset

    # Accumulator for the state (registers)
    running_state = tl.zeros((BLOCK_HEADDIM, state_dim), dtype=tl.float32)
    
    # Linear Scan Loop over Chunks
    for c in range(num_chunks):
        # 1. Load A_cumsum for current chunk (scalar)
        curr_A_val = tl.load(ptr_A + c * stride_A_c)

        # 2. Compute decay = exp(A[t] - A[t-1])
        decay = tl.exp(curr_A_val)
        
        # 3. Apply decay to running state
        running_state = running_state * decay
        
        # 4. Load input state (B @ X) for this chunk
        ptr_input_chunk = old_states_ptrs + (c * stride_old_states_c) + \
                          (head_dim_offsets[:, None] * stride_old_states_d) + \
                          (state_offsets[None, :] * stride_old_states_k)
        input_state = tl.load(ptr_input_chunk, mask=True, other=0.0)

        # 5. Add input to running state
        running_state = running_state + input_state

        # 6. Load C, shape is (chunk_size, state_dim)
        C_ptrs_chunk = C_ptrs + (c * stride_C_c) + (chunk_offsets[:, None] * stride_C_l) + \
                       (state_offsets[None, :] * stride_C_n)
        C = tl.load(C_ptrs_chunk, mask=True, other=0.0)

        # 7. Load state_decay, shape is (chunk_size,)
        state_decay_ptrs_chunk = state_decay_ptrs + (c * stride_state_decay_c) + \
                                 (chunk_offsets * stride_state_decay_l)
        state_decay = tl.load(state_decay_ptrs_chunk, mask=True, other=0.0)

        # 8. compute (C @ running_state) * (chunk_size,)
        # bf16 matmul (chunk_size, state_dim) @ (BLOCK_HEADDIM, state_dim) then
        # broadcasted elementwise multiply (chunk_size, BLOCK_HEADDIM) * (chunk_size,)
        C_state_decay = tl.dot(C, tl.trans(running_state.to(tl.bfloat16)), out_dtype=tl.float32) * state_decay[:, None]

        # 9. Load intra-chunk output contribution
        diag_y_ptrs_chunk = diag_y_ptrs + (c * stride_out_y_c) + (chunk_offsets[:, None] * stride_out_y_l) + \
                            (head_dim_offsets[None, :] * stride_out_y_p)
        diag_y_chunk = tl.load(diag_y_ptrs_chunk, mask=True, other=0.0)

        # 10. store to Y
        out_y_ptrs_chunk = out_y_ptrs + (c * stride_out_y_c) + (chunk_offsets[:, None] * stride_out_y_l) + \
                           (head_dim_offsets[None, :] * stride_out_y_p)
        tl.store(out_y_ptrs_chunk, (C_state_decay + diag_y_chunk).to(tl.bfloat16), mask=True)

    # 1. Load A_cumsum for current chunk (scalar)
    curr_A_val = tl.load(ptr_A + num_chunks * stride_A_c)

    # 2. Compute decay = exp(A[t] - A[t-1])
    decay = tl.exp(curr_A_val)
    
    # 3. Apply decay to running state
    running_state = running_state * decay

    # 4. Load input state (B @ X) for this chunk
    ptr_input_chunk = old_states_ptrs + (num_chunks * stride_old_states_c) + \
                      (head_dim_offsets[:, None] * stride_old_states_d) + \
                      (state_offsets[None, :] * stride_old_states_k)
    input_state = tl.load(ptr_input_chunk, mask=True, other=0.0)

    # 5. Add input to running state
    running_state = running_state + input_state

    # store to final_state
    ptr_final_chunk = final_ptrs + \
                      (head_dim_offsets[:, None] * stride_final_d) + \
                      (state_offsets[None, :] * stride_final_k) 
    tl.store(ptr_final_chunk, running_state, mask = True)
    return


def ssd_mamba2_torch(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """
    Compute the reference SSD output using the PyTorch implementation.

    Args:
        x (torch.Tensor, input): [batch, seq_len, num_heads, head_dim] input features.
        A (torch.Tensor, input): [batch, seq_len, num_heads] decay values in log space.
        B (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] state-projection weights.
        C (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] output-projection weights.
        chunk_size (int, input): Number of timesteps per chunk.

    Returns:
        y (torch.Tensor, output): [batch, seq_len, num_heads, head_dim] SSD output.
        final_states (torch.Tensor, output): [batch, num_heads, head_dim, state_dim] final recurrent state.
    """
    y, final_states = ssd_minimal_discrete(x, A, B, C, chunk_size)
    return y, final_states


def launch_ssd_mamba2(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_y: torch.Tensor,
    chunk_size: int,
):
    """
    Launch the Triton SSD kernels (intra-chunk + inter-chunk scan).

    Args:
        x (torch.Tensor, input): [batch, seq_len, num_heads, head_dim] input features (contiguous).
        A (torch.Tensor, input): [batch, seq_len, num_heads] decay values in log space (contiguous).
        B (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] state-projection weights (contiguous).
        C (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] output-projection weights (contiguous).
        out_y (torch.Tensor, output): [batch, seq_len, num_heads, head_dim] buffer for Triton output (contiguous, preallocated).
        chunk_size (int, input): Number of timesteps per chunk; must divide seq_len.

    Returns:
        final_states (torch.Tensor, output): [batch, num_heads, head_dim, state_dim] final recurrent state after the scan.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape (batch, seq_len, num_heads, head_dim); got {x.shape}")
    if A.shape != (x.shape[0], x.shape[1], x.shape[2]):
        raise ValueError(f"A must be shape (batch, seq_len, num_heads); got {A.shape}")

    for tensor_name, tensor in (("x", x), ("A", A), ("B", B), ("C", C), ("out_y", out_y)):
        if not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous")

    batch_size, seq_len, num_heads, head_dim = x.shape
    if out_y.shape != x.shape or out_y.dtype != x.dtype:
        raise ValueError(f"out_y must match x shape {x.shape} and dtype {x.dtype}; got {out_y.shape} and {out_y.dtype}")

    if seq_len % chunk_size != 0:
        raise ValueError(f"Sequence length {seq_len} must be divisible by chunk size {chunk_size}")
    num_chunks = seq_len // chunk_size

    num_groups = B.shape[2]
    state_dim = B.shape[3]
    if B.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"B must have leading shape (batch, seq_len)={(batch_size, seq_len)}; got {B.shape[:2]}")
    if C.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"C must have leading shape (batch, seq_len)={(batch_size, seq_len)}; got {C.shape[:2]}")
    if B.shape[2:] != C.shape[2:]:
        raise ValueError(f"B and C must share (num_groups, state_dim); got {B.shape[2:]} vs {C.shape[2:]}")

    def grid(meta):
        return (triton.cdiv(seq_len, chunk_size), num_heads // meta["BLOCK_HEAD"], batch_size)

    diag_y = torch.empty_like(x, dtype=torch.float32)
    state_decay = torch.empty((batch_size, num_heads, num_chunks, chunk_size), dtype=torch.float32, device=x.device)
    # one more chunk added for initial zero state
    chunk_states = torch.zeros(batch_size, 1 + num_chunks, num_heads, head_dim, state_dim, dtype=torch.float32, device=x.device)
    intra_chunk_cumsum = torch.zeros(batch_size, num_heads, 1 + num_chunks, dtype=torch.float32, device=x.device)
    stride_x_b, stride_x_l, stride_x_h, stride_x_d = x.stride()
    stride_A_in_b, stride_A_in_l, stride_A_in_h = A.stride()
    stride_B_b, stride_B_l, stride_B_g, stride_B_d = B.stride()
    stride_C_b, stride_C_l, stride_C_g, stride_C_d = C.stride()
    stride_diag_y_b, stride_diag_y_l, stride_diag_y_h, stride_diag_y_d = diag_y.stride()
    stride_A_cumsum_b, stride_A_cumsum_h, stride_A_cumsum_c = intra_chunk_cumsum.stride()
    stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l = state_decay.stride()
    stride_out_states_b, stride_out_states_c, stride_out_states_h, stride_out_states_d, stride_out_states_k = chunk_states.stride()
    ssd_intra_chunk_kernel[grid](
        x,
        A,
        B,
        C,
        diag_y,
        intra_chunk_cumsum,
        state_decay,
        chunk_states,
        seq_len,
        num_heads,
        num_groups,
        stride_x_b, stride_x_l, stride_x_h, stride_x_d,
        stride_A_in_b, stride_A_in_l, stride_A_in_h,
        stride_B_b, stride_B_l, stride_B_g, stride_B_d,
        stride_C_b, stride_C_l, stride_C_g, stride_C_d,
        stride_diag_y_b, stride_diag_y_l, stride_diag_y_h, stride_diag_y_d,
        stride_A_cumsum_b, stride_A_cumsum_h, stride_A_cumsum_c,
        stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l,
        stride_out_states_b, stride_out_states_c, stride_out_states_h, stride_out_states_d, stride_out_states_k,
        state_dim,
        head_dim,
        chunk_size,
    )

    # compute new_states = decay_chunk @ out_states, reduce over chunk dimension
    def grid_scan(meta):
        return (head_dim // meta["BLOCK_HEADDIM"], num_heads, batch_size)
    
    final_states = torch.empty([batch_size, num_heads, head_dim, state_dim], device=x.device, dtype=torch.float32)
    stride_A_b, stride_A_h, stride_A_c = intra_chunk_cumsum.stride()
    stride_old_states_b, stride_old_states_c, stride_old_states_h, stride_old_states_d, stride_old_states_k = chunk_states.stride()
    stride_final_b, stride_final_h, stride_final_d, stride_final_k = final_states.stride()
    
    stride_C_c = chunk_size * stride_C_l
    stride_C_h = stride_C_g
    stride_C_n = stride_C_d
    stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l = state_decay.stride()
    stride_out_y_b, stride_out_y_l, stride_out_y_h, stride_out_y_p = out_y.stride()
    stride_out_y_c = chunk_size * stride_out_y_l
    ssd_inter_chunk_scan_linear_kernel[grid_scan](
        intra_chunk_cumsum,
        chunk_states,        # Input tensor
        C,
        state_decay,
        diag_y,
        final_states,
        out_y,
        num_chunks,
        state_dim,
        chunk_size,
        num_groups,
        stride_A_b, stride_A_h, stride_A_c,
        stride_old_states_b, stride_old_states_c, stride_old_states_h, stride_old_states_d, stride_old_states_k,
        stride_C_b, stride_C_c, stride_C_l, stride_C_h, stride_C_n,
        stride_state_decay_b, stride_state_decay_h, stride_state_decay_c, stride_state_decay_l,
        stride_final_b, stride_final_h, stride_final_d, stride_final_k,
        stride_out_y_b, stride_out_y_c, stride_out_y_l, stride_out_y_h, stride_out_y_p,
    )

    return final_states


def benchmark_torch_ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    warmup: int,
    rep: int,
) -> float:
    """
    Benchmark the PyTorch reference SSD implementation.

    Args:
        x (torch.Tensor, input): [batch, seq_len, num_heads, head_dim] input features.
        A (torch.Tensor, input): [batch, seq_len, num_heads] decay values in log space.
        B (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] state-projection weights.
        C (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] output-projection weights.
        chunk_size (int, input): Number of timesteps per chunk for the reference implementation.
        warmup (int, input): Number of warmup iterations.
        rep (int, input): Number of timed repetitions.

    Returns:
        float (output): Kernel latency in milliseconds.
    """
    def op():
        ssd_mamba2_torch(x, A, B, C, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_y: torch.Tensor,
    chunk_size: int,
    warmup: int,
    rep: int,
) -> float:
    """
    Benchmark the Triton SSD kernel.

    Args:
        x (torch.Tensor, input): [batch, seq_len, num_heads, head_dim] input features.
        A (torch.Tensor, input): [batch, seq_len, num_heads] decay values in log space.
        B (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] state-projection weights.
        C (torch.Tensor, input): [batch, seq_len, num_groups, state_dim] output-projection weights.
        out_y (torch.Tensor, output): [batch, seq_len, num_heads, head_dim] output buffer for Triton kernel.
        chunk_size (int, input): Number of timesteps per chunk.
        warmup (int, input): Number of warmup iterations.
        rep (int, input): Number of timed repetitions.

    Returns:
        float (output): Kernel latency in milliseconds.
    """
    def op():
        launch_ssd_mamba2(x, A, B, C, out_y, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """
    Generate synthetic inputs, compare Triton vs PyTorch SSD outputs, and print throughput metrics.

    Args:
        None (input): Uses fixed configuration for shapes and datatypes.

    Returns:
        None

    Config:
        batch_size (int): 1
        seq_len (int): 128 * 1024
        num_heads (int): 8
        head_dim (int): 64
        num_groups (int): 1
        state_dim (int): 64
        dtype (torch.dtype): torch.bfloat16
        chunk_size (int): 128
        warmup_iters (int): 10
        repetition_iters (int): 10
    """
    batch_size = 1
    seq_len = 128 * 1024
    head_dim = 64
    num_heads = 8
    num_groups = 1
    state_dim = 64
    device = "cuda"
    dtype = torch.bfloat16
    chunk_size = 128
    warmup_iters = 10
    repetition_iters = 10
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)  # V in transformer view
    A = torch.log(torch.rand(batch_size, seq_len, num_heads, device=device, dtype=torch.float32))  # decay (log space)
    B = torch.randn(batch_size, seq_len, num_groups, state_dim, device=device, dtype=dtype)  # K in transformer view
    C = torch.randn(batch_size, seq_len, num_groups, state_dim, device=device, dtype=dtype)  # Q in transformer view
    y_triton = torch.empty_like(x, dtype=dtype)

    y_torch, final_states_torch = ssd_mamba2_torch(
        x.to(torch.float32), A, B.to(torch.float32), C.to(torch.float32), chunk_size=chunk_size
    )
    y_torch = y_torch.to(dtype)
    torch_ms = benchmark_torch_ssd(
        x.to(torch.float32),
        A,
        B.to(torch.float32),
        C.to(torch.float32),
        chunk_size=chunk_size,
        warmup=warmup_iters,
        rep=repetition_iters,
    )
    bytes_transferred = x.element_size() * x.numel() + \
                        A.element_size() * A.numel() + \
                        B.element_size() * B.numel() + \
                        C.element_size() * C.numel() + \
                        y_torch.element_size() * y_torch.numel() + \
                        final_states_torch.element_size() * final_states_torch.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4)
    print(f"Torch time: {torch_ms :.3f} ms | bandwidth: {torch_bw} TB/s")

    num_chunks = seq_len // chunk_size
    final_states_triton = launch_ssd_mamba2(x, A, B, C, y_triton, chunk_size=chunk_size)
    error_y = relative_error(y_torch, y_triton)
    print(f"Y relative error: {error_y:.3e}")
    error_final_state = relative_error(final_states_torch, final_states_triton)
    print(f"Final state relative error: {error_final_state:.3e}")

    triton_ms = benchmark_triton_ssd(
        x, A, B, C, y_triton, chunk_size=chunk_size, warmup=warmup_iters, rep=repetition_iters
    )

    num_chunks = seq_len // chunk_size
    triton_flops = (
        2 * batch_size * num_groups * num_chunks * (chunk_size * chunk_size * state_dim) +  # intra-chunk C^T B
        2 * batch_size * num_heads * num_chunks * (chunk_size * chunk_size * head_dim) +  # intra-chunk L_CTB X
        2 * batch_size * num_heads * num_chunks * (chunk_size * state_dim * head_dim) +  # intra-chunk B_decay X
        2 * batch_size * num_heads * ((num_chunks + 1) * state_dim * head_dim) + # inter-chunk decay_chunk @ states
        2 * batch_size * num_heads * num_chunks * (chunk_size * state_dim * head_dim)  # inter-chunk C states
    )

    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    triton_tflops = triton_flops / (triton_ms * 1e-3) / 1e12
    print(f"Triton time: {triton_ms :.3f} ms | bandwidth: {triton_bw:.3f} TB/s | TFLOPS: {triton_tflops:.3f}")


def relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """
    Compute the relative error between two tensors of identical shape and dtype.

    Args:
        ref (torch.Tensor, input): Reference tensor; arbitrary shape, dtype matches approx.
        approx (torch.Tensor, input): Approximated tensor; same shape/dtype as ref.

    Returns:
        float (output): Relative error defined as ||ref - approx|| / ||ref||.
    """
    if ref.shape != approx.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {approx.shape}")
    if ref.dtype != approx.dtype:
        raise ValueError(f"Data type mismatch: {ref.dtype} vs {approx.dtype}")
    denom = torch.linalg.norm(ref.to(torch.float32))
    if denom == 0:
        return float(torch.linalg.norm(ref.to(torch.float32) - approx.to(torch.float32)))
    return float(torch.linalg.norm(ref.to(torch.float32) - approx.to(torch.float32)) / denom)


if __name__ == "__main__":
    test()
