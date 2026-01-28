# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for elementwise add Triton kernel.
# ==============================================================================
"""Tests for elementwise add Triton kernel."""

import os
import sys

import torch
import triton.testing

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from elementwiseadd_triton_kernel import (  # noqa: E402
    launch_elementwiseadd_fp32_kernel,
)


def benchmark_torch_add(a, b, out, warmup, rep):
    """
    Benchmark torch.add using Triton's timing helper.

    Main feature:
        Measures torch.add latency using Triton's benchmarking utilities.

    Inputs:
        a: torch.Tensor float32 of shape [m, n]
        b: torch.Tensor float32 of shape [m, n]
        out: torch.Tensor float32 of shape [m, n]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Invoke torch.add for benchmarking.

        Main feature:
            Performs elementwise addition in-place into out.

        Inputs:
            None

        Outputs:
            None
        """
        torch.add(a, b, out=out)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_add(a, b, out, warmup, rep):
    """
    Benchmark the Triton elementwise add kernel.

    Main feature:
        Measures kernel latency with autotuned block sizes.

    Inputs:
        a: torch.Tensor float32 of shape [m, n]
        b: torch.Tensor float32 of shape [m, n]
        out: torch.Tensor float32 of shape [m, n]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Invoke the Triton kernel for benchmarking.

        Main feature:
            Dispatches the elementwise add kernel.

        Inputs:
            None

        Outputs:
            None
        """
        launch_elementwiseadd_fp32_kernel(a, b, out)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref, approx):
    """
    Compute ||ref - approx|| / ||ref||.

    Main feature:
        Returns a relative L2 error for two tensors.

    Inputs:
        ref: torch.Tensor float32 of shape [m, n]
        approx: torch.Tensor float32 of shape [m, n]

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
    Compare PyTorch and Triton elementwise add and report bandwidth.

    Main feature:
        Runs accuracy and performance checks for the Triton kernel.

    Inputs:
        None

    Outputs:
        None
    """
    m = 4096
    n = 4096
    warmup_iters = 100
    device = "cuda:0"

    torch.manual_seed(0)
    a = torch.rand((m, n), dtype=torch.float32, device=device)
    b = torch.rand((m, n), dtype=torch.float32, device=device)
    c_ref = torch.empty_like(a)
    c_triton = torch.empty_like(a)

    torch_ms = benchmark_torch_add(a, b, c_ref, warmup_iters, rep=100)
    triton_ms = benchmark_triton_add(a, b, c_triton, warmup_iters, rep=100)
    launch_elementwiseadd_fp32_kernel(a, b, c_triton)

    bytes_transferred = 3 * a.element_size() * a.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4)
    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    print(f"PyTorch time: {torch_ms * 1000:.3f} us | bandwidth: {torch_bw:.3f} TB/s")
    print(f"Triton time: {triton_ms * 1000:.3f} us | bandwidth: {triton_bw:.3f} TB/s")

    error = relative_error(c_ref, c_triton)
    print(f"elementwiseadd_fp32_kernel relative error: {error:.3e}")


if __name__ == "__main__":
    run_tests()
