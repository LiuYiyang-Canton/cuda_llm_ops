# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for Flash GQA decoding Triton kernel.
# ==============================================================================
"""Tests for the Flash GQA decoding Triton kernel."""

import math
import os
import sys

import torch
import triton.testing

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from decode_flash_gqa_triton_kernel import (  # noqa: E402
    decode_flash_gqa_bf16_kernel,
    launch_decode_flash_gqa_bf16_kernel,
)


def flash_gqa_reference(q, k, v, scale=None):
    """
    Compute a reference grouped attention result.

    Main feature:
        Uses fp32 accumulation and tiling for numerical stability.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        scale: optional float scalar

    Outputs:
        out: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    batch, q_heads, seqlen_q, head_dim = q.shape
    _, num_kv_heads, seqlen_kv, _ = k.shape
    group_size = q_heads // num_kv_heads
    tile_n = 128

    outputs = []
    for b_idx in range(batch):
        q_b = q[b_idx].to(torch.float32)
        k_b = k[b_idx].to(torch.float32)
        v_b = v[b_idx].to(torch.float32)
        out_heads = []
        for kv_idx in range(num_kv_heads):
            q_slice = q_b[kv_idx * group_size : (kv_idx + 1) * group_size]
            m_i = torch.full((group_size, seqlen_q), -float("inf"), device=q_slice.device, dtype=torch.float32)
            l_i = torch.zeros((group_size, seqlen_q), device=q_slice.device, dtype=torch.float32)
            acc = torch.zeros((group_size, seqlen_q, head_dim), device=q_slice.device, dtype=torch.float32)
            for start in range(0, seqlen_kv, tile_n):
                end = min(start + tile_n, seqlen_kv)
                k_tile = k_b[kv_idx, start:end]
                v_tile = v_b[kv_idx, start:end]
                qk_scores = torch.matmul(q_slice, k_tile.transpose(-1, -2)) * scale

                m_i_new = torch.maximum(m_i, qk_scores.max(dim=-1).values)
                p = torch.exp(qk_scores - m_i_new.unsqueeze(-1))
                alpha = torch.exp(m_i - m_i_new)

                acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_tile)
                l_i = l_i * alpha + p.sum(dim=-1)
                m_i = m_i_new

            out_slice = acc / l_i.unsqueeze(-1)
            out_heads.append(out_slice.to(q.dtype))
        out_b = torch.cat(out_heads, dim=0).to(q.dtype)
        outputs.append(out_b)
    return torch.stack(outputs, dim=0)


def benchmark_torch_flash_gqa(q, k, v, out, warmup, rep):
    """
    Benchmark the PyTorch reference grouped attention.

    Main feature:
        Measures reference latency using Triton's benchmarking utilities.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        out: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Run the reference kernel for benchmarking.

        Main feature:
            Computes reference output and copies into out.

        Inputs:
            None

        Outputs:
            None
        """
        out.copy_(flash_gqa_reference(q, k, v))

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_flash_gqa(q, k, v, out, warmup, rep):
    """
    Benchmark the Triton Flash GQA decoding kernel.

    Main feature:
        Measures Triton kernel latency using autotuned configurations.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        out: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Run the Triton kernel for benchmarking.

        Main feature:
        Launches the Triton Flash GQA decoding kernel.

        Inputs:
            None

        Outputs:
            None
        """
        launch_decode_flash_gqa_bf16_kernel(q, k, v, out)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref, approx):
    """
    Compute ||ref - approx|| / ||ref||.

    Main feature:
        Returns a relative L2 error for two tensors.

    Inputs:
        ref: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        approx: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]

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


def bytes_accessed(q, k, v, o):
    """
    Estimate bytes moved by the attention kernel.

    Main feature:
        Approximates total bytes read and written.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim]
        o: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim]

    Outputs:
        total_bytes: int64 scalar
    """
    return (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()


def run_tests(run_kernel=True, warmup=100, rep=100):
    """
    Validate and benchmark Flash GQA decoding on a single batch.

    Main feature:
        Runs reference and Triton paths and reports bandwidth.

    Inputs:
        run_kernel: bool scalar, whether to benchmark Triton kernel
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        None
    """
    batch_size = 60
    seqlen_q = 1
    seqlen_kv = 4096
    num_query_heads = 128
    num_kv_heads = 8
    head_dim = 128

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q = torch.rand((batch_size, num_query_heads, seqlen_q, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.rand((batch_size, num_kv_heads, seqlen_kv, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.rand((batch_size, num_kv_heads, seqlen_kv, head_dim), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    ref = flash_gqa_reference(q, k, v)
    if not run_kernel:
        print("Flash GQA decoding Triton kernel ready; set run_kernel=True to run benchmarks.")
        return

    torch_ms = benchmark_torch_flash_gqa(q, k, v, out, warmup=warmup, rep=rep)

    triton_ms = None
    if device.type == "cuda":
        triton_ms = benchmark_triton_flash_gqa(q, k, v, out, warmup=warmup, rep=rep)
        _ = launch_decode_flash_gqa_bf16_kernel(q, k, v, out)
    else:
        print("Triton benchmark skipped (CPU fallback).")

    err = relative_error(ref, out)
    moved_bytes = bytes_accessed(q, k, v, out)
    print(f"decode_flash_gqa_bf16_kernel relative error: {err:.3e}")
    torch_bw = moved_bytes / (torch_ms * 1e-3) / (1024 ** 4)
    print(f"PyTorch reference: {torch_ms:.5f} ms | approx. bandwidth: {torch_bw:.6f} TB/s")
    if triton_ms is not None:
        triton_bw = moved_bytes / (triton_ms * 1e-3) / (1024 ** 4)
        print(f"Triton Flash GQA decoding: {triton_ms:.5f} ms | approx. bandwidth: {triton_bw:.6f} TB/s")
        best_cfg = getattr(decode_flash_gqa_bf16_kernel, "best_config", None)
        if best_cfg is not None:
            print(f"Triton autotune best config: {best_cfg}")


if __name__ == "__main__":
    run_tests()
