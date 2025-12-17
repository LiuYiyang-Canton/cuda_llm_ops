"""Gated Delta Net (GDN) reference implementations, Triton wrappers, and benchmarks."""

from typing import Union

import torch
import triton
import triton.testing
from fla.ops.utils import solve_tril
from gdn_kernels import (
    compute_local_cumsum,
    compute_output,
    compute_scaled_kkt,
    compute_states,
    compute_w_u,
)


def generate_gdn_inputs(
    batch: int,
    seqlen: int,
    num_heads: int,
    headdim_qk: int,
    headdim_v: int,
    device: Union[str, torch.device] = "cuda",
    seed: int = 0,
):
    """
    Generate random GDN inputs with expected shapes and dtypes.

    Args:
        batch (int): Batch size (B).
        seqlen (int): Sequence length (S).
        num_heads (int): Number of attention heads (H).
        headdim_qk (int): Per-head dimension for q/k (D_qk).
        headdim_v (int): Per-head dimension for v (D_v).
        device (str or torch.device): Target device.
        seed (int): RNG seed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            alpha: [B, S, H] float32 log-decay values.
            beta: [B, S, H] float32 values.
            q: [B, S, H, D_qk] bfloat16 query vectors.
            k: [B, S, H, D_qk] bfloat16 key vectors.
            v: [B, S, H, D_v] bfloat16 value vectors.
    """
    torch.manual_seed(seed)
    alpha = torch.log(torch.rand((batch, seqlen, num_heads), device=device, dtype=torch.float32))
    beta = torch.rand((batch, seqlen, num_heads), device=device, dtype=torch.float32)
    q = torch.randn((batch, seqlen, num_heads, headdim_qk), device=device, dtype=torch.bfloat16)
    k = torch.randn((batch, seqlen, num_heads, headdim_qk), device=device, dtype=torch.bfloat16)
    v = torch.randn((batch, seqlen, num_heads, headdim_v), device=device, dtype=torch.bfloat16)

    k /= torch.linalg.norm(k, dim=-1, keepdim=True)
    v /= torch.linalg.norm(k, dim=-1, keepdim=True)

    return alpha, beta, q, k, v


def gated_delta_net_torch_recurrent(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    Reference recurrent GDN forward pass without chunking.

    S_t = S_{t-1} (exp(alpha_t) * (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
    o_t = S_t q_t

    Args:
        alpha (torch.Tensor): [B, S, H] float32 log-decay values.
        beta (torch.Tensor): [B, S, H] float32 values.
        q (torch.Tensor): [B, S, H, D_qk] bfloat16 query vectors.
        k (torch.Tensor): [B, S, H, D_qk] bfloat16 key vectors.
        v (torch.Tensor): [B, S, H, D_v] bfloat16 value vectors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            final_states: [B, H, D_qk, D_v] float32.
            o: [B, S, H, D_v] bfloat16.
    """
    if alpha.shape != beta.shape:
        raise ValueError(f"alpha/beta shape mismatch: {alpha.shape} vs {beta.shape}")
    if q.shape != k.shape:
        raise ValueError(f"q/k shape mismatch: {q.shape} vs {k.shape}")
    if alpha.shape[:3] != q.shape[:3]:
        raise ValueError(f"alpha/beta and q/k batch-head dims differ: {alpha.shape[:3]} vs {q.shape[:3]}")
    if v.shape[:3] != alpha.shape[:3]:
        raise ValueError(f"v batch-head dims differ: {v.shape[:3]} vs {alpha.shape[:3]}")

    batch, seqlen, num_heads, headdim_qk = q.shape
    headdim_v = v.shape[-1]

    # States are accumulated in fp32 for stability. Shape (B, H, D_qk, D_v).
    state = torch.zeros((batch, num_heads, headdim_qk, headdim_v), device=q.device, dtype=torch.float32)
    outputs = torch.empty((batch, seqlen, num_heads, headdim_v), device=q.device, dtype=torch.bfloat16)
    eye = torch.eye(headdim_qk, device=q.device, dtype=torch.float32)

    for t in range(seqlen):
        k_t = k[:, t].float()          # (B, H, D_qk)
        v_t = v[:, t].float()          # (B, H, D_v)
        alpha_t = alpha[:, t].float().view(batch, num_heads, 1, 1)
        beta_t = beta[:, t].float().view(batch, num_heads, 1, 1)

        # Update: S_t = S_{t-1} (alpha_t (I - beta_t k k^T)) + beta_t v k^T
        k_outer = torch.einsum("bhk,bhl->bhkl", k_t, k_t)                   # (B, H, D_qk, D_qk)
        update_mat = torch.exp(alpha_t) * (eye - beta_t * k_outer)                     # (B, H, D_qk, D_qk)
        state = torch.einsum("bhkv,bhkl->bhlv", state, update_mat) + beta_t * torch.einsum("bhv,bhk->bhkv", v_t, k_t) # (B, H, D_k, D_v)

        # o_t = S_t q_t
        out_t = torch.einsum("bhkv,bhk->bhv", state, q[:, t].float())
        outputs[:, t] = out_t.to(dtype=torch.bfloat16)

    return state, outputs


def gated_delta_net_torch_chunk(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
):
    """
    Torch reference for the chunked GDN forward pass.

    Args:
        alpha (torch.Tensor): [B, S, H] float32 log-decay values.
        beta (torch.Tensor): [B, S, H] float32 values.
        q (torch.Tensor): [B, S, H, D_qk] bfloat16 query vectors.
        k (torch.Tensor): [B, S, H, D_qk] bfloat16 key vectors.
        v (torch.Tensor): [B, S, H, D_v] bfloat16 value vectors.
        chunk_size (int): Chunk length (C), must divide S.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            final_state: [B, H, D_qk, D_v] float32.
            o: [B, S, H, D_v] bfloat16.
    """
    B, S, H, D_qk = q.shape
    D_v = v.shape[-1]

    # only supports the case when seqlen is multiples of chunk_size
    assert S % chunk_size == 0, "seqlen must be a multiple of chunk_size"
    num_chunks = S // chunk_size


    alpha_chunks = alpha.view(B, num_chunks, chunk_size, H)
    beta_chunks = beta.view(B, num_chunks, chunk_size, H)
    q_chunks = q.view(B, num_chunks, chunk_size, H, D_qk)
    k_chunks = k.view(B, num_chunks, chunk_size, H, D_qk)
    v_chunks = v.view(B, num_chunks, chunk_size, H, D_v)

    # We follow the roadmap of the official FLA's GDN

    # step 1, compute per-chunk cumsum of alpha
    g = torch.cumsum(alpha_chunks, dim = 2) # (B, num_chunks, chunk_size, H)

    # step 2, compute gamma * beta * (k @ K^T) (strcitly lower triangular)
    A = beta_chunks.unsqueeze(-1) * torch.einsum("bnchk, bnthk->bncht", k_chunks.float(), k_chunks.float())
    A = torch.tril(A.permute(0,1,3,2,4), diagonal=-1).permute(0,1,3,2,4)
    gamma = torch.exp(g.unsqueeze(-1) - g.transpose(-2,-1).unsqueeze(2))
    A = A * gamma # (B, num_chunks, chunk_size, H, chunk_size)
    A = A.reshape(B, S, H, chunk_size)

    if A.is_cuda:
        A = solve_tril(A, cu_seqlens=None) # (B, S, H, chunk_size)
    else:
        # CPU fallback: solve (I + A) X = I with a standard triangular solver.
        A_blocks = A.reshape(B, num_chunks, chunk_size, H, chunk_size).permute(0, 1, 3, 2, 4)
        eye = torch.eye(chunk_size, device=A.device, dtype=A.dtype)
        L = eye + A_blocks
        A_blocks = torch.linalg.solve_triangular(L, eye, upper=False)
        A = A_blocks.permute(0, 1, 3, 2, 4).reshape(B, S, H, chunk_size)

    # step 3, compute W and U
    bK_decay = torch.exp(g).unsqueeze(-1) * beta_chunks.unsqueeze(-1) * k_chunks.float() # (B, num_chunks, chunk_size, H, D_qk)
    bV = beta_chunks.unsqueeze(-1) * v_chunks.float() # (B, num_chunks, chunk_size, H, D_v)

    w = torch.einsum("bncht,bnthk->bnchk", A.view(B, num_chunks, chunk_size, H, chunk_size), bK_decay)
    u = torch.einsum("bncht,bnthv->bnchv", A.view(B, num_chunks, chunk_size, H, chunk_size), bV)

    # step 4, compute h, v_new, final_state
    h = torch.empty((B, num_chunks, H, D_qk, D_v), device=q.device, dtype = torch.float32)
    v_new = torch.empty_like(u)
    final_state = k.new_empty(B, H, D_qk, D_v, dtype=torch.float32, device=q.device)
    old_states = torch.zeros((B, H, D_qk, D_v), dtype = torch.float32, device = q.device)
    for chunk_idx in range(num_chunks):
        # v_new = u - w @ s
        u_curr = u[:, chunk_idx]
        w_curr = w[:, chunk_idx]
        v_new[:, chunk_idx] = u_curr - torch.einsum("bchk,bhkv->bchv", w_curr, old_states)

        g_curr = g[:, chunk_idx] # (B, C, H)
        g_last_curr = g[:, chunk_idx, -1] # (B, H)

        k_curr = k_chunks[:, chunk_idx].float()
        k_decay_curr = torch.exp(g_last_curr.unsqueeze(1) - g_curr).unsqueeze(-1) * k_curr # (B, C, H, K)

        # compute new states
        old_states_decay = old_states * torch.exp(g_last_curr).unsqueeze(-1).unsqueeze(-1)
        h[:, chunk_idx] = old_states_decay + torch.einsum("bchv,bchk->bhkv", v_new[:, chunk_idx], k_decay_curr)
        old_states = h[:, chunk_idx]
    
    final_state = h[:, -1]

    # step 5, compute outputs
    q_decay = torch.exp(g).unsqueeze(-1) * q_chunks.float() # (B, num_chunks, chunk_size, H, D_qk)
    qk_masked = torch.einsum("bnchk,bnlhk->bnchl", q_chunks.float(), k_chunks.float())* gamma
    qk_masked = torch.tril(qk_masked.transpose(-3,-2)).transpose(-3,-2)
    o = torch.einsum("bnchl,bnlhv->bnchv", qk_masked, v_new)
    o[:, 1:] += torch.einsum("bnchk,bnhkv->bnchv", q_decay[:, 1:], h[:, :-1]) 
    o = o.to(dtype=torch.bfloat16).reshape(B, S, H, D_v)

    return final_state, o


def benchmark_gdn_torch_chunk(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    warmup: int,
    rep: int,
) -> float:
    """
    Benchmark the chunked torch implementation.

    Args:
        alpha (torch.Tensor): [B, S, H] float32 log-decay values.
        beta (torch.Tensor): [B, S, H] float32 values.
        q (torch.Tensor): [B, S, H, D_qk] bfloat16 query vectors.
        k (torch.Tensor): [B, S, H, D_qk] bfloat16 key vectors.
        v (torch.Tensor): [B, S, H, D_v] bfloat16 value vectors.
        chunk_size (int): Chunk length (C).
        warmup (int): Warmup iterations for benchmarking.
        rep (int): Timed iterations for benchmarking.

    Returns:
        float: Average runtime in milliseconds.
    """
    def op():
        with torch.inference_mode():
            gated_delta_net_torch_chunk(alpha, beta, q, k, v, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def gated_delta_net_triton_chunk(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
):
    """
    Triton chunked GDN forward pass using custom kernels.

    Args:
        alpha (torch.Tensor): [B, S, H] float32 log-decay values.
        beta (torch.Tensor): [B, S, H] float32 values.
        q (torch.Tensor): [B, S, H, D_qk] bfloat16 query vectors.
        k (torch.Tensor): [B, S, H, D_qk] bfloat16 key vectors.
        v (torch.Tensor): [B, S, H, D_v] bfloat16 value vectors.
        chunk_size (int): Chunk length (C), must divide S.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            final_state: [B, H, D_qk, D_v] float32.
            o: [B, S, H, D_v] bfloat16.
    """
    # make sure everything is contiguous
    if not alpha.is_contiguous():
        raise ValueError("alpha must be contiguous")
    if not beta.is_contiguous():
        raise ValueError("beta must be contiguous")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if not k.is_contiguous():
        raise ValueError("k must be contiguous")
    if not v.is_contiguous():
        raise ValueError("v must be contiguous")

    B, S, H, D_qk = q.shape
    D_v = v.shape[-1]

    # only supports the case when seqlen is multiples of chunk_size
    assert S % chunk_size == 0, "seqlen must be a multiple of chunk_size"
    num_chunks = S // chunk_size


    alpha_chunks = alpha.view(B, num_chunks, chunk_size, H)
    beta_chunks = beta.view(B, num_chunks, chunk_size, H)
    q_chunks = q.view(B, num_chunks, chunk_size, H, D_qk)
    k_chunks = k.view(B, num_chunks, chunk_size, H, D_qk)
    v_chunks = v.view(B, num_chunks, chunk_size, H, D_v)

    # We follow the roadmap of the official FLA's GDN

    # step 1, compute per-chunk cumsum of alpha
    g = compute_local_cumsum(alpha_chunks) # (B, num_chunks, chunk_size, H)

    # step 2, compute gamma * beta * (k @ K^T) (strcitly lower triangular)
    A = compute_scaled_kkt(g, beta_chunks, k_chunks, chunk_size)

    A = solve_tril(A, cu_seqlens=None) # (B, S, H, chunk_size)

    # step 3, compute W and U
    w, u = compute_w_u(A, g, beta_chunks, k_chunks, v_chunks, chunk_size)

    # step 4, compute h, v_new, final_state
    h, v_new, final_state = compute_states(u, w, g, k_chunks, chunk_size)

    # step 5, compute outputs
    o = compute_output(q_chunks, k_chunks, v_new, h, g, chunk_size)

    return final_state, o


def benchmark_gdn_triton_chunk(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    warmup: int,
    rep: int,
) -> float:
    """
    Benchmark the Triton chunked implementation.

    Args:
        alpha (torch.Tensor): [B, S, H] float32 log-decay values.
        beta (torch.Tensor): [B, S, H] float32 values.
        q (torch.Tensor): [B, S, H, D_qk] bfloat16 query vectors.
        k (torch.Tensor): [B, S, H, D_qk] bfloat16 key vectors.
        v (torch.Tensor): [B, S, H, D_v] bfloat16 value vectors.
        chunk_size (int): Chunk length (C).
        warmup (int): Warmup iterations for benchmarking.
        rep (int): Timed iterations for benchmarking.

    Returns:
        float: Average runtime in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_gdn_triton_chunk requires a CUDA device.")
    def op():
        with torch.inference_mode():
            gated_delta_net_triton_chunk(alpha, beta, q, k, v, chunk_size)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Return ||ref - approx|| / ||ref||."""
    if ref.shape != approx.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {approx.shape}")
    if ref.dtype != approx.dtype:
        raise ValueError(f"Data type mismatch: {ref.dtype} vs {approx.dtype}")
    denom = torch.linalg.norm(ref)
    if denom == 0:
        return float(torch.linalg.norm(ref - approx))
    return float(torch.linalg.norm(ref - approx) / denom)


def main():
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    batch = 1
    seqlen = 128 * 1024
    num_heads = 8
    headdim_qk = 64
    headdim_v = 64

    chunk_size = 64
    warmup_iters = 10
    benchmark_reps = 10

    alpha, beta, q, k, v = generate_gdn_inputs(
        batch, seqlen, num_heads, headdim_qk, headdim_v, device=device, seed=0
    )
    final_states_shape = (batch, num_heads, headdim_qk, headdim_v)
    o_shape = (batch, seqlen, num_heads, headdim_v)

    final_states_recurrent_torch, output_recurrent_torch = gated_delta_net_torch_recurrent(alpha, beta, q, k, v)
    
    # assert output shape and dtype
    assert final_states_recurrent_torch.shape == final_states_shape
    assert output_recurrent_torch.shape == o_shape

    final_states_chunk_torch, output_chunk_torch = gated_delta_net_torch_chunk(alpha, beta, q, k, v, chunk_size=chunk_size)

    # assert output shape and dtype
    assert final_states_chunk_torch.shape == final_states_shape
    assert output_chunk_torch.shape == o_shape

    # check gated_delta_net_torch_chunk accuracy
    state_error = relative_error(final_states_recurrent_torch, final_states_chunk_torch)
    print(f"gated_delta_net_torch_chunk error final_states: {state_error:.4e}")
    output_error = relative_error(output_recurrent_torch.float(), output_chunk_torch.float())
    print(f"gated_delta_net_torch_chunk error output: {output_error:.4e}")

    if has_cuda:
        final_states_chunk_triton, output_chunk_triton = gated_delta_net_triton_chunk(
            alpha, beta, q, k, v, chunk_size=chunk_size
        )

        # assert output shape and dtype
        assert final_states_chunk_triton.shape == final_states_shape
        assert output_chunk_triton.shape == o_shape

        # check gated_delta_net_triton_chunk accuracy
        state_error = relative_error(final_states_recurrent_torch, final_states_chunk_triton)
        print(f"gated_delta_net_triton_chunk error final_states: {state_error:.4e}")
        output_error = relative_error(output_recurrent_torch.float(), output_chunk_triton.float())
        print(f"gated_delta_net_triton_chunk error output: {output_error:.4e}")

        triton_chunk_ms = benchmark_gdn_triton_chunk(
            alpha, beta, q, k, v, chunk_size, warmup_iters, benchmark_reps
        )

        num_chunks = seqlen // chunk_size
        triton_flops = (
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk) + # K @ K^T
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk) + # A @ bK_decay
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_v) + # A @ bV
            2 * batch * num_chunks * num_heads * (chunk_size ** 3 / 3) * 2 + # solve_tril(A), we count fp32 operations twice
            2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v) + # w_curr @ old_states
            2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v) + # v_new @ k_decay_curr
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk) + # q @ k^T
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_v) + # qk_masked @ v_new
            2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v) # qk_decay @ h
        )

        triton_tflops = triton_flops / (triton_chunk_ms * 1e-3) / 1e12
        print(f"Triton time: {triton_chunk_ms :.3f} ms | TFLOPS: {triton_tflops:.3f}")

    torch_chunk_ms = benchmark_gdn_torch_chunk(alpha, beta, q, k, v, chunk_size, warmup_iters, benchmark_reps)
    print(f"gated_delta_net_torch_chunk latency: {torch_chunk_ms:.3f} ms")



if __name__ == "__main__":
    main()
