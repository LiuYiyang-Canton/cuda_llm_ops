# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for GDN Triton kernels.
# ==============================================================================
"""Tests for Gated Delta Net (GDN) Triton kernels."""

import os
import sys

import torch
import triton.testing
from fla.ops.utils import solve_tril

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from gdn_triton_kernel import (  # noqa: E402
    compute_local_cumsum,
    compute_output,
    compute_scaled_kkt,
    compute_states,
    compute_w_u,
)


def generate_gdn_inputs(batch, seqlen, num_heads, headdim_qk, headdim_v, device="cuda", seed=0):
    """
    Generate random GDN inputs with expected shapes and dtypes.

    Main feature:
        Creates alpha/beta/q/k/v tensors with normalized k/v vectors.

    Inputs:
        batch: int32 scalar
        seqlen: int32 scalar
        num_heads: int32 scalar
        headdim_qk: int32 scalar
        headdim_v: int32 scalar
        device: str or torch.device scalar
        seed: int32 scalar

    Outputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
    """
    torch.manual_seed(seed)
    alpha = torch.log(torch.rand((batch, seqlen, num_heads), device=device, dtype=torch.float32))
    beta = torch.rand((batch, seqlen, num_heads), device=device, dtype=torch.float32)
    q = torch.randn((batch, seqlen, num_heads, headdim_qk), device=device, dtype=torch.bfloat16)
    k = torch.randn((batch, seqlen, num_heads, headdim_qk), device=device, dtype=torch.bfloat16)
    v = torch.randn((batch, seqlen, num_heads, headdim_v), device=device, dtype=torch.bfloat16)

    k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
    v = v / torch.linalg.norm(k, dim=-1, keepdim=True)

    return alpha, beta, q, k, v


def gated_delta_net_torch_recurrent(alpha, beta, q, k, v):
    """
    Reference recurrent GDN forward pass without chunking.

    Main feature:
        Updates recurrent state per timestep and computes outputs.

    Inputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]

    Outputs:
        final_state: torch.Tensor float32 of shape [batch, num_heads, headdim_qk, headdim_v]
        outputs: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
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

    state = torch.zeros((batch, num_heads, headdim_qk, headdim_v), device=q.device, dtype=torch.float32)
    outputs = torch.empty((batch, seqlen, num_heads, headdim_v), device=q.device, dtype=torch.bfloat16)
    eye = torch.eye(headdim_qk, device=q.device, dtype=torch.float32)

    for t in range(seqlen):
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        alpha_t = alpha[:, t].float().view(batch, num_heads, 1, 1)
        beta_t = beta[:, t].float().view(batch, num_heads, 1, 1)

        k_outer = torch.einsum("bhk,bhl->bhkl", k_t, k_t)
        update_mat = torch.exp(alpha_t) * (eye - beta_t * k_outer)
        state = torch.einsum("bhkv,bhkl->bhlv", state, update_mat) + beta_t * torch.einsum(
            "bhv,bhk->bhkv", v_t, k_t
        )

        out_t = torch.einsum("bhkv,bhk->bhv", state, q[:, t].float())
        outputs[:, t] = out_t.to(dtype=torch.bfloat16)

    return state, outputs


def gated_delta_net_torch_chunk(alpha, beta, q, k, v, chunk_size):
    """
    Torch reference for the chunked GDN forward pass.

    Main feature:
        Uses chunked KKT solve and state updates to compute outputs.

    Inputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
        chunk_size: int32 scalar

    Outputs:
        final_state: torch.Tensor float32 of shape [batch, num_heads, headdim_qk, headdim_v]
        outputs: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
    """
    batch, seqlen, num_heads, headdim_qk = q.shape
    headdim_v = v.shape[-1]

    if seqlen % chunk_size != 0:
        raise ValueError("seqlen must be a multiple of chunk_size")
    num_chunks = seqlen // chunk_size

    alpha_chunks = alpha.view(batch, num_chunks, chunk_size, num_heads)
    beta_chunks = beta.view(batch, num_chunks, chunk_size, num_heads)
    q_chunks = q.view(batch, num_chunks, chunk_size, num_heads, headdim_qk)
    k_chunks = k.view(batch, num_chunks, chunk_size, num_heads, headdim_qk)
    v_chunks = v.view(batch, num_chunks, chunk_size, num_heads, headdim_v)

    g_vals = torch.cumsum(alpha_chunks, dim=2)
    a_vals = beta_chunks.unsqueeze(-1) * torch.einsum(
        "bnchk, bnthk->bncht", k_chunks.float(), k_chunks.float()
    )
    a_vals = torch.tril(a_vals.permute(0, 1, 3, 2, 4), diagonal=-1).permute(0, 1, 3, 2, 4)
    gamma = torch.exp(g_vals.unsqueeze(-1) - g_vals.transpose(-2, -1).unsqueeze(2))
    a_vals = a_vals * gamma
    a_vals = a_vals.reshape(batch, seqlen, num_heads, chunk_size)

    if a_vals.is_cuda:
        a_vals = solve_tril(a_vals, cu_seqlens=None)
    else:
        a_blocks = a_vals.reshape(batch, num_chunks, chunk_size, num_heads, chunk_size).permute(0, 1, 3, 2, 4)
        eye = torch.eye(chunk_size, device=a_vals.device, dtype=a_vals.dtype)
        l_vals = eye + a_blocks
        a_blocks = torch.linalg.solve_triangular(l_vals, eye, upper=False)
        a_vals = a_blocks.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, num_heads, chunk_size)

    b_k_decay = torch.exp(g_vals).unsqueeze(-1) * beta_chunks.unsqueeze(-1) * k_chunks.float()
    b_v = beta_chunks.unsqueeze(-1) * v_chunks.float()

    w_vals = torch.einsum("bncht,bnthk->bnchk", a_vals.view(batch, num_chunks, chunk_size, num_heads, chunk_size), b_k_decay)
    u_vals = torch.einsum("bncht,bnthv->bnchv", a_vals.view(batch, num_chunks, chunk_size, num_heads, chunk_size), b_v)

    h_vals = torch.empty((batch, num_chunks, num_heads, headdim_qk, headdim_v), device=q.device, dtype=torch.float32)
    v_new = torch.empty_like(u_vals)
    old_states = torch.zeros((batch, num_heads, headdim_qk, headdim_v), dtype=torch.float32, device=q.device)
    for chunk_idx in range(num_chunks):
        u_curr = u_vals[:, chunk_idx]
        w_curr = w_vals[:, chunk_idx]
        v_new[:, chunk_idx] = u_curr - torch.einsum("bchk,bhkv->bchv", w_curr, old_states)

        g_curr = g_vals[:, chunk_idx]
        g_last_curr = g_vals[:, chunk_idx, -1]

        k_curr = k_chunks[:, chunk_idx].float()
        k_decay_curr = torch.exp(g_last_curr.unsqueeze(1) - g_curr).unsqueeze(-1) * k_curr

        old_states_decay = old_states * torch.exp(g_last_curr).unsqueeze(-1).unsqueeze(-1)
        h_vals[:, chunk_idx] = old_states_decay + torch.einsum("bchv,bchk->bhkv", v_new[:, chunk_idx], k_decay_curr)
        old_states = h_vals[:, chunk_idx]

    final_state = h_vals[:, -1]

    q_decay = torch.exp(g_vals).unsqueeze(-1) * q_chunks.float()
    qk_masked = torch.einsum("bnchk,bnlhk->bnchl", q_chunks.float(), k_chunks.float()) * gamma
    qk_masked = torch.tril(qk_masked.transpose(-3, -2)).transpose(-3, -2)
    outputs = torch.einsum("bnchl,bnlhv->bnchv", qk_masked, v_new)
    outputs[:, 1:] += torch.einsum("bnchk,bnhkv->bnchv", q_decay[:, 1:], h_vals[:, :-1])
    outputs = outputs.to(dtype=torch.bfloat16).reshape(batch, seqlen, num_heads, headdim_v)

    return final_state, outputs


def gated_delta_net_triton_chunk(alpha, beta, q, k, v, chunk_size):
    """
    Triton chunked GDN forward pass using custom kernels.

    Main feature:
        Uses Triton kernels to compute chunked GDN outputs.

    Inputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
        chunk_size: int32 scalar

    Outputs:
        final_state: torch.Tensor float32 of shape [batch, num_heads, headdim_qk, headdim_v]
        outputs: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
    """
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

    batch, seqlen, num_heads, headdim_qk = q.shape
    headdim_v = v.shape[-1]

    if seqlen % chunk_size != 0:
        raise ValueError("seqlen must be a multiple of chunk_size")
    num_chunks = seqlen // chunk_size

    alpha_chunks = alpha.view(batch, num_chunks, chunk_size, num_heads)
    beta_chunks = beta.view(batch, num_chunks, chunk_size, num_heads)
    q_chunks = q.view(batch, num_chunks, chunk_size, num_heads, headdim_qk)
    k_chunks = k.view(batch, num_chunks, chunk_size, num_heads, headdim_qk)
    v_chunks = v.view(batch, num_chunks, chunk_size, num_heads, headdim_v)

    g_vals = compute_local_cumsum(alpha_chunks)
    a_vals = compute_scaled_kkt(g_vals, beta_chunks, k_chunks, chunk_size)
    a_vals = solve_tril(a_vals, cu_seqlens=None)
    w_vals, u_vals = compute_w_u(a_vals, g_vals, beta_chunks, k_chunks, v_chunks, chunk_size)
    h_vals, v_new, final_state = compute_states(u_vals, w_vals, g_vals, k_chunks, chunk_size)
    outputs = compute_output(q_chunks, k_chunks, v_new, h_vals, g_vals, chunk_size)

    return final_state, outputs


def benchmark_gdn_torch_chunk(alpha, beta, q, k, v, chunk_size, warmup, rep):
    """
    Benchmark the chunked PyTorch GDN implementation.

    Main feature:
        Measures latency using Triton's benchmarking utilities.

    Inputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
        chunk_size: int32 scalar
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Run the PyTorch chunked GDN path.

        Main feature:
            Executes the chunked reference implementation.

        Inputs:
            None

        Outputs:
            None
        """
        with torch.inference_mode():
            gated_delta_net_torch_chunk(alpha, beta, q, k, v, chunk_size)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_gdn_triton_chunk(alpha, beta, q, k, v, chunk_size, warmup, rep):
    """
    Benchmark the Triton chunked GDN implementation.

    Main feature:
        Measures Triton kernel latency.

    Inputs:
        alpha: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        beta: torch.Tensor float32 of shape [batch, seqlen, num_heads]
        q: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        k: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_qk]
        v: torch.Tensor bf16 of shape [batch, seqlen, num_heads, headdim_v]
        chunk_size: int32 scalar
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_gdn_triton_chunk requires a CUDA device.")

    def op():
        """
        Run the Triton chunked GDN path.

        Main feature:
            Executes the Triton kernels via the wrapper.

        Inputs:
            None

        Outputs:
            None
        """
        with torch.inference_mode():
            gated_delta_net_triton_chunk(alpha, beta, q, k, v, chunk_size)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref, approx):
    """
    Compute ||ref - approx|| / ||ref||.

    Main feature:
        Returns a relative L2 error for two tensors.

    Inputs:
        ref: torch.Tensor float32 of shape [*]
        approx: torch.Tensor float32 of shape [*]

    Outputs:
        error: float scalar
    """
    if ref.shape != approx.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {approx.shape}")
    if ref.dtype != approx.dtype:
        raise ValueError(f"Data type mismatch: {ref.dtype} vs {approx.dtype}")
    denom = torch.linalg.norm(ref)
    if denom == 0:
        return float(torch.linalg.norm(ref - approx))
    return float(torch.linalg.norm(ref - approx) / denom)


def run_tests():
    """
    Validate GDN outputs and benchmark Triton kernels.

    Main feature:
        Compares recurrent and chunked outputs and reports performance.

    Inputs:
        None

    Outputs:
        None
    """
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

    assert final_states_recurrent_torch.shape == final_states_shape
    assert output_recurrent_torch.shape == o_shape

    final_states_chunk_torch, output_chunk_torch = gated_delta_net_torch_chunk(
        alpha, beta, q, k, v, chunk_size=chunk_size
    )

    assert final_states_chunk_torch.shape == final_states_shape
    assert output_chunk_torch.shape == o_shape

    state_error = relative_error(final_states_recurrent_torch, final_states_chunk_torch)
    print(f"gated_delta_net_torch_chunk error final_states: {state_error:.4e}")
    output_error = relative_error(output_recurrent_torch.float(), output_chunk_torch.float())
    print(f"gated_delta_net_torch_chunk error output: {output_error:.4e}")

    if has_cuda:
        final_states_chunk_triton, output_chunk_triton = gated_delta_net_triton_chunk(
            alpha, beta, q, k, v, chunk_size=chunk_size
        )

        assert final_states_chunk_triton.shape == final_states_shape
        assert output_chunk_triton.shape == o_shape

        state_error = relative_error(final_states_recurrent_torch, final_states_chunk_triton)
        print(f"gated_delta_net_triton_chunk error final_states: {state_error:.4e}")
        output_error = relative_error(output_recurrent_torch.float(), output_chunk_triton.float())
        print(f"gated_delta_net_triton_chunk error output: {output_error:.4e}")

        triton_chunk_ms = benchmark_gdn_triton_chunk(
            alpha, beta, q, k, v, chunk_size, warmup_iters, benchmark_reps
        )

        num_chunks = seqlen // chunk_size
        triton_flops = (
            2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk)
            + 2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk)
            + 2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_v)
            + 2 * batch * num_chunks * num_heads * (chunk_size ** 3 / 3) * 2
            + 2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v)
            + 2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v)
            + 2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_qk)
            + 2 * batch * num_chunks * num_heads * (chunk_size ** 2 * headdim_v)
            + 2 * batch * num_chunks * num_heads * (chunk_size * headdim_qk * headdim_v)
        )

        triton_tflops = triton_flops / (triton_chunk_ms * 1e-3) / 1e12

        sizeof_bf16 = 2
        sizeof_float32 = 4

        triton_bw = (
            batch * num_chunks * chunk_size * num_heads * sizeof_float32 * 2
            + batch * num_chunks * chunk_size * num_heads * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * chunk_size * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * chunk_size * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * chunk_size * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * chunk_size * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_bf16
            + batch * num_chunks * num_heads * headdim_qk * headdim_v * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * headdim_qk * sizeof_bf16
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_bf16
            + batch * num_chunks * num_heads * headdim_qk * headdim_v * sizeof_float32
            + batch * num_chunks * chunk_size * num_heads * headdim_v * sizeof_bf16
        )
        triton_bw = triton_bw / (2 ** 40) / (triton_chunk_ms * 1e-3)

        print(f"Triton time: {triton_chunk_ms :.3f} ms | TFLOPS: {triton_tflops:.3f} | BW: {triton_bw:.3f} TB/s")

    torch_chunk_ms = benchmark_gdn_torch_chunk(alpha, beta, q, k, v, chunk_size, warmup_iters, benchmark_reps)
    print(f"gated_delta_net_torch_chunk latency: {torch_chunk_ms:.3f} ms")


if __name__ == "__main__":
    run_tests()
