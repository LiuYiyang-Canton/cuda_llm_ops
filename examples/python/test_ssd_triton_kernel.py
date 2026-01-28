# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for the SSD Triton kernels.
# ==============================================================================
"""Tests for the SSD Triton kernels."""

import os
import sys

import torch
import torch.nn.functional as functional
import triton.testing
from einops import rearrange, repeat

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from ssd_triton_kernel import (  # noqa: E402
    launch_ssd_mamba2,
)


def segsum(x):
    """
    Compute a more stable segment sum for SSD reference.

    Main feature:
        Builds a lower-triangular cumulative sum with masked values.

    Inputs:
        x: torch.Tensor float32 of shape [*, d, t]

    Outputs:
        x_segsum: torch.Tensor float32 of shape [*, d, t, t]
    """
    t_len = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=t_len)
    mask = torch.tril(torch.ones(t_len, t_len, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t_len, t_len, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(x, a, b, c, block_len, initial_states=None):
    """
    Compute a reference SSD output using the discrete formulation.

    Main feature:
        Implements the chunked SSD reference with streaming softmax.

    Inputs:
        x: torch.Tensor float32 of shape [batch, length, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, length, num_heads]
        b: torch.Tensor float32 of shape [batch, length, num_groups, state_dim]
        c: torch.Tensor float32 of shape [batch, length, num_groups, state_dim]
        block_len: int32 scalar
        initial_states: optional torch.Tensor float32 of shape [batch, 1, num_heads, head_dim, state_dim]

    Outputs:
        y: torch.Tensor float32 of shape [batch, length, num_heads, head_dim]
        final_state: torch.Tensor float32 of shape [batch, num_heads, head_dim, state_dim]
    """
    if initial_states is None:
        initial_states = None

    if x.dtype != a.dtype or x.dtype != b.dtype or x.dtype != c.dtype:
        raise ValueError("x, a, b, c must share the same dtype")
    if x.shape[1] % block_len != 0:
        raise ValueError("Sequence length must be divisible by block_len")

    x, a, b, c = [rearrange(tensor, "b (c l) ... -> b c l ...", l=block_len) for tensor in (x, a, b, c)]

    a = rearrange(a, "b c l h -> b h c l")
    a_cumsum = torch.cumsum(a, dim=-1)

    l_vals = torch.exp(segsum(a))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, l_vals, x)

    decay_states = torch.exp((a_cumsum[:, :, :, -1:] - a_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(functional.pad(a_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(a_cumsum)
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, state_decay_out)

    y = rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
    return y, final_state


def ssd_mamba2_torch(x, a, b, c, chunk_size=64):
    """
    Compute the reference SSD output using PyTorch.

    Main feature:
        Uses the minimal discrete reference implementation.

    Inputs:
        x: torch.Tensor float32 of shape [batch, seq_len, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, seq_len, num_heads]
        b: torch.Tensor float32 of shape [batch, seq_len, num_groups, state_dim]
        c: torch.Tensor float32 of shape [batch, seq_len, num_groups, state_dim]
        chunk_size: int32 scalar

    Outputs:
        y: torch.Tensor float32 of shape [batch, seq_len, num_heads, head_dim]
        final_states: torch.Tensor float32 of shape [batch, num_heads, head_dim, state_dim]
    """
    y, final_states = ssd_minimal_discrete(x, a, b, c, chunk_size)
    return y, final_states


def benchmark_torch_ssd(x, a, b, c, chunk_size, warmup, rep):
    """
    Benchmark the PyTorch SSD reference implementation.

    Main feature:
        Measures reference latency using Triton's benchmarking utilities.

    Inputs:
        x: torch.Tensor float32 of shape [batch, seq_len, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, seq_len, num_heads]
        b: torch.Tensor float32 of shape [batch, seq_len, num_groups, state_dim]
        c: torch.Tensor float32 of shape [batch, seq_len, num_groups, state_dim]
        chunk_size: int32 scalar
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute the PyTorch SSD reference for benchmarking.

        Main feature:
            Computes SSD output and final states.

        Inputs:
            None

        Outputs:
            None
        """
        ssd_mamba2_torch(x, a, b, c, chunk_size)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_ssd(x, a, b, c, out_y, chunk_size, warmup, rep):
    """
    Benchmark the Triton SSD kernels.

    Main feature:
        Measures Triton pipeline latency.

    Inputs:
        x: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        a: torch.Tensor float32 of shape [batch, seq_len, num_heads]
        b: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        c: torch.Tensor bf16 of shape [batch, seq_len, num_groups, state_dim]
        out_y: torch.Tensor bf16 of shape [batch, seq_len, num_heads, head_dim]
        chunk_size: int32 scalar
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute the Triton SSD pipeline for benchmarking.

        Main feature:
            Dispatches the intra- and inter-chunk kernels.

        Inputs:
            None

        Outputs:
            None
        """
        launch_ssd_mamba2(x, a, b, c, out_y, chunk_size)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref, approx):
    """
    Compute the relative error between two tensors.

    Main feature:
        Returns ||ref - approx|| / ||ref|| in float32.

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
    denom = torch.linalg.norm(ref.to(torch.float32))
    if denom == 0:
        return float(torch.linalg.norm(ref.to(torch.float32) - approx.to(torch.float32)))
    return float(torch.linalg.norm(ref.to(torch.float32) - approx.to(torch.float32)) / denom)


def run_tests():
    """
    Compare Triton SSD outputs with the PyTorch reference.

    Main feature:
        Runs correctness and performance checks for SSD.

    Inputs:
        None

    Outputs:
        None
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
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    a = torch.log(torch.rand(batch_size, seq_len, num_heads, device=device, dtype=torch.float32))
    b = torch.randn(batch_size, seq_len, num_groups, state_dim, device=device, dtype=dtype)
    c = torch.randn(batch_size, seq_len, num_groups, state_dim, device=device, dtype=dtype)
    y_triton = torch.empty_like(x, dtype=dtype)

    y_torch, final_states_torch = ssd_mamba2_torch(
        x.to(torch.float32), a, b.to(torch.float32), c.to(torch.float32), chunk_size=chunk_size
    )
    y_torch = y_torch.to(dtype)
    torch_ms = benchmark_torch_ssd(
        x.to(torch.float32),
        a,
        b.to(torch.float32),
        c.to(torch.float32),
        chunk_size=chunk_size,
        warmup=warmup_iters,
        rep=repetition_iters,
    )
    print(f"Torch time: {torch_ms :.3f} ms")

    final_states_triton = launch_ssd_mamba2(x, a, b, c, y_triton, chunk_size=chunk_size)
    error_y = relative_error(y_torch, y_triton)
    print(f"Y relative error: {error_y:.3e}")
    error_final_state = relative_error(final_states_torch, final_states_triton)
    print(f"Final state relative error: {error_final_state:.3e}")

    triton_ms = benchmark_triton_ssd(
        x,
        a,
        b,
        c,
        y_triton,
        chunk_size=chunk_size,
        warmup=warmup_iters,
        rep=repetition_iters,
    )

    num_chunks = seq_len // chunk_size
    triton_flops = (
        2 * batch_size * num_groups * num_chunks * (chunk_size * chunk_size * state_dim)
        + 2 * batch_size * num_heads * num_chunks * (chunk_size * chunk_size * head_dim)
        + 2 * batch_size * num_heads * num_chunks * (chunk_size * state_dim * head_dim)
        + 2 * batch_size * num_heads * ((num_chunks + 1) * state_dim * head_dim)
        + 2 * batch_size * num_heads * num_chunks * (chunk_size * state_dim * head_dim)
    )

    sizeof_bf16 = 2
    sizeof_float32 = 4
    triton_bytes_transferred = (
        batch_size * seq_len * num_heads * head_dim * sizeof_bf16
        + batch_size * seq_len * num_heads * sizeof_float32
        + batch_size * seq_len * num_groups * state_dim * sizeof_bf16
        + batch_size * seq_len * num_groups * state_dim * sizeof_bf16
        + batch_size * seq_len * num_heads * head_dim * sizeof_float32
        + batch_size * num_heads * (1 + num_chunks) * sizeof_float32
        + batch_size * num_heads * seq_len * sizeof_float32
        + batch_size * (1 + num_chunks) * num_heads * head_dim * state_dim * sizeof_float32
        + batch_size * (1 + num_chunks) * num_heads * head_dim * state_dim * sizeof_float32
        + batch_size * seq_len * num_heads * head_dim * sizeof_float32
        + batch_size * num_heads * head_dim * state_dim * sizeof_float32
        + batch_size * seq_len * num_heads * head_dim * sizeof_bf16
    )

    triton_bw = triton_bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    triton_tflops = triton_flops / (triton_ms * 1e-3) / 1e12
    print(f"Triton time: {triton_ms :.3f} ms | bandwidth: {triton_bw:.3f} TB/s | TFLOPS: {triton_tflops:.3f}")


if __name__ == "__main__":
    run_tests()
