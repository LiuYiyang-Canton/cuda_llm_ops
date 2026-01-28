# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for reduce-sum Triton kernel.
# ==============================================================================
"""Tests for the reduce-sum Triton kernel."""

import os
import sys

import torch
import triton.testing

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from reducesum_triton_kernel import (  # noqa: E402
    launch_reducesum_fp32_kernel,
)


def benchmark_torch_reducesum(matrix, rowsum, warmup, rep):
    """
    Benchmark torch.sum over rows using Triton's timing helper.

    Main feature:
        Measures PyTorch row-sum latency.

    Inputs:
        matrix: torch.Tensor float32 of shape [n_rows, n_cols]
        rowsum: torch.Tensor float32 of shape [n_rows]
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute torch.sum for benchmarking.

        Main feature:
            Computes row-wise sums into rowsum.

        Inputs:
            None

        Outputs:
            None
        """
        torch.sum(matrix, dim=1, out=rowsum)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_reducesum(matrix, rowsum, warmup, rep):
    """
    Benchmark the Triton reduce-sum kernel.

    Main feature:
        Measures Triton kernel latency.

    Inputs:
        matrix: torch.Tensor float32 of shape [n_rows, n_cols]
        rowsum: torch.Tensor float32 of shape [n_rows]
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute the Triton reduce-sum kernel for benchmarking.

        Main feature:
            Dispatches the Triton kernel with validated inputs.

        Inputs:
            None

        Outputs:
            None
        """
        launch_reducesum_fp32_kernel(matrix, rowsum)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref, approx):
    """
    Compute ||ref - approx|| / ||ref||.

    Main feature:
        Returns a relative L2 error for two tensors.

    Inputs:
        ref: torch.Tensor float32 of shape [n_rows]
        approx: torch.Tensor float32 of shape [n_rows]

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
    Compare PyTorch and Triton reduce-sum outputs and bandwidth.

    Main feature:
        Runs correctness and performance checks on GPU.

    Inputs:
        None

    Outputs:
        None
    """
    n_rows = 4096
    n_cols = 4096
    warmup_iters = 10000
    device = "cuda:0"

    torch.manual_seed(0)
    matrix = torch.rand((n_rows, n_cols), dtype=torch.float32, device=device)
    rowsum_ref = torch.empty((n_rows,), dtype=torch.float32, device=device)
    rowsum_triton = torch.empty_like(rowsum_ref)

    torch_ms = benchmark_torch_reducesum(matrix, rowsum_ref, warmup_iters, rep=100)
    triton_ms = benchmark_triton_reducesum(matrix, rowsum_triton, warmup_iters, rep=100)
    launch_reducesum_fp32_kernel(matrix, rowsum_triton)

    bytes_transferred = matrix.element_size() * matrix.numel() + rowsum_ref.element_size() * rowsum_ref.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4)
    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    print(f"PyTorch time: {torch_ms * 1000:.3f} us | bandwidth: {torch_bw:.3f} TB/s")
    print(f"Triton time: {triton_ms * 1000:.3f} us | bandwidth: {triton_bw:.3f} TB/s")

    error = relative_error(rowsum_ref, rowsum_triton)
    print(f"reducesum_fp32_kernel relative error: {error:.3e}")


if __name__ == "__main__":
    run_tests()
