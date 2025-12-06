"""Triton implementation of Mamba-2 SSD (state space duality) linear attention."""

import torch
import triton
import triton.language as tl
import triton.testing
from ssd_minimal import ssd_minimal_discrete
from torch.profiler import profile, record_function, ProfilerActivity


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_DSTATE": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_DSTATE": 64}, num_warps=8),
    ],
    key=["seqlen", "dstate"],
)
@triton.jit
def ssd_mamba2_kernel(
    x_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    out_y_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_feat,
    stride_A_batch,
    stride_A_seq,
    stride_A_head,
    stride_B_batch,
    stride_B_seq,
    stride_B_group,
    stride_B_state,
    stride_C_batch,
    stride_C_seq,
    stride_C_group,
    stride_C_state,
    batch,
    seqlen,
    nheads,
    ngroups,
    dstate,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pass


@triton.autotune(
    configs=[
        triton.Config({}),
    ],
    key=["headdim", "dstate"],
)
@triton.jit
def ssd_intra_chunk_kernel(
    x_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    out_y_ptr,
    batch,
    seqlen,
    nheads,
    ngroups,
    dstate: tl.constexpr,
    headdim: tl.constexpr,
    chunk_size: tl.constexpr,
):
    pid_chunk = tl.program_id(0)  # chunk along sequence
    pid_h = tl.program_id(1)  # head
    pid_b = tl.program_id(2)  # batch

    # chunk start index
    chunk_start = pid_chunk * chunk_size

    # load x, shape is (chunk_size, headdim)
    x_ptrs = x_ptr + pid_b * seqlen * nheads * headdim + chunk_start * nheads * headdim + pid_h * headdim
    offsets_seqlen = tl.arange(0, chunk_size)
    offsets_headdim = tl.arange(0, headdim)
    mask = (chunk_start + offsets_seqlen) < seqlen
    x = tl.load(x_ptrs + offsets_seqlen[:, None] * nheads * headdim + offsets_headdim[None, :], mask=mask[:, None], other=0.0)

    # load A, shape is (chunk_size,)
    A_ptrs = A_ptr + pid_b * seqlen * nheads + chunk_start * nheads + pid_h
    A = tl.load(A_ptrs + offsets_seqlen * nheads, mask=mask, other=0.0)

    # load B, shape is (chunk_size, dstate)
    B_ptrs = B_ptr + pid_b * seqlen * ngroups * dstate + chunk_start * ngroups * dstate + (pid_h % ngroups) * dstate
    offsets_dstate = tl.arange(0, dstate)
    B = tl.load(B_ptrs + offsets_seqlen[:, None] * ngroups * dstate + offsets_dstate[None, :], mask=mask[:, None], other=0.0)

    # load C, shape is (chunk_size, dstate)
    C_ptrs = C_ptr + pid_b * seqlen * ngroups * dstate + chunk_start * ngroups * dstate + (pid_h % ngroups) * dstate
    C = tl.load(C_ptrs + offsets_seqlen[:, None] * ngroups * dstate + offsets_dstate[None, :], mask=mask[:, None], other=0.0)

    # compute C^T B, bf16 matmul
    CTB = tl.dot(C, tl.trans(B), out_dtype=tl.float32, allow_tf32=False)  # (chunk_size, chunk_size)

    # compute L = exp(segsum(A))
    rows = tl.arange(0, chunk_size)
    cols = tl.arange(0, chunk_size)
    A_broadcasted = tl.where(rows[:, None] > cols[None, :], A[:, None], 0.0)
    A_broadcasted = tl.cumsum(A_broadcasted, axis=0)
    L = tl.exp(A_broadcasted)  # (chunk_size, chunk_size)
    L = tl.where(rows[:, None] >= cols[None, :], L, 0)

    # elementwise multiply L and CTB
    L_CTB = L * CTB  # (chunk_size, chunk_size)

    # compute Y_diag = L_CTB @ X, bf16 matmul
    Y_diag = tl.dot(L_CTB.to(tl.bfloat16), x, out_dtype=tl.float32) # (chunk_size, headdim)

    # store Y_diag
    out_y_ptrs = out_y_ptr + pid_b * seqlen * nheads * headdim + chunk_start * nheads * headdim + pid_h * headdim
    tl.store(out_y_ptrs + offsets_seqlen[:, None] * nheads * headdim + offsets_headdim[None, :], Y_diag.to(tl.bfloat16), mask=mask[:, None])


def ssd_mamba2_torch(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, chunk_size=64) -> torch.Tensor:
    y, final_states = ssd_minimal_discrete(x, A, B, C, chunk_size)
    return y, final_states


def launch_ssd_mamba2(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_y: torch.Tensor,
    chunk_size: int
):
    """Launch the Triton SSD kernel with stride metadata populated."""
    if x.shape != out_y.shape:
        raise ValueError(f"Output shape {out.shape} must match input shape {x.shape}")
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape (batch, seqlen, nheads, headdim); got {x.shape}")
    if A.shape != (x.shape[0], x.shape[1], x.shape[2]):
        raise ValueError(f"A must be shape (batch, seqlen, nheads); got {A.shape}")

    """Ensure all inputs and outputs are contiguous"""
    x, A, B, C, out_y = [t.contiguous() for t in (x, A, B, C, out_y)]

    batch, seqlen, nheads, headdim = x.shape

    if seqlen % chunk_size != 0:
        raise ValueError(f"Sequence length {seqlen} must be divisible by chunk size {chunk_size}")

    ngroups = B.shape[2]
    dstate = B.shape[3]

    def grid(meta):
        return (triton.cdiv(seqlen, chunk_size), nheads, batch)

    ssd_intra_chunk_kernel[grid](
        x,
        A,
        B,
        C,
        out_y,
        batch,
        seqlen,
        nheads,
        ngroups,
        dstate,
        headdim,
        chunk_size
    )


def benchmark_torch_ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    warmup: int,
    rep: int,
) -> float:
    """Benchmark the PyTorch reference SSD implementation."""
    def op():
        ssd_mamba2_torch(x, A, B, C, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_y: torch.Tensor,
    chunk_size,
    warmup: int,
    rep: int,
) -> float:
    """Benchmark the Triton SSD kernel."""
    def op():
        launch_ssd_mamba2(x, A, B, C, out_y, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """Generate sample inputs and compare Triton vs PyTorch SSD once implementations are filled in."""
    batch = 1
    seqlen = 128 * 1024
    headdim = 64
    nheads = 8
    ngroups = 1
    dstate = 64
    device = "cuda"
    dtype = torch.bfloat16
    chunk_size = 64
    warmup_iters = 100
    repetition_iters = 100

    """
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
        final_states: (batch, length, n_heads, d_state)
    """
    torch.manual_seed(0)
    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype) # V in transformer view
    A = torch.log(torch.rand(batch, seqlen, nheads, device=device, dtype=torch.float32)) # decay (log space)
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) # K in transformer view
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) # Q in transformer view
    y_diag_triton = torch.empty_like(x, dtype = dtype)

    y_torch, final_states_torch = ssd_mamba2_torch(x.to(torch.float32), A, B.to(torch.float32), C.to(torch.float32), chunk_size=chunk_size)
    y_torch = y_torch.to(dtype)
    torch_ms = benchmark_torch_ssd(x.to(torch.float32), A, B.to(torch.float32), C.to(torch.float32), chunk_size=chunk_size, warmup=warmup_iters, rep=repetition_iters)
    bytes_transferred = x.element_size() * x.numel() + \
                        A.element_size() * A.numel() + \
                        B.element_size() * B.numel() + \
                        C.element_size() * C.numel() + \
                        y_torch.element_size() * y_torch.numel() + \
                        final_states_torch.element_size() * final_states_torch.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4)
    print(f"Torch time: {torch_ms :.3f} ms | bandwidth: {bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4):.3f} TB/s")

    launch_ssd_mamba2(x, A, B, C, y_diag_triton, chunk_size=chunk_size)
    error_diag = relative_error(y_torch, y_diag_triton)
    print(f"Y_diag relative error: {error_diag:.3e}")

    triton_ms = benchmark_triton_ssd(x, A, B, C, y_diag_triton, chunk_size=chunk_size, warmup=warmup_iters, rep=repetition_iters)

    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    print(f"Triton time: {triton_ms :.3f} ms | bandwidth: {triton_bw:.3f} TB/s")

    # error = relative_error(out_torch, out)
    # print(f"ssd_mamba2_kernel relative error: {error:.3e}")


def relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Return ||ref - approx|| / ||ref||."""
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
