# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Accuracy and performance tests for FP8 GEMM Triton kernel.
# ==============================================================================
"""Tests for the FP8 GEMM Triton kernel."""

import os
import sys

import torch
import triton.testing

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from gemm_fp8_triton_kernel import (  # noqa: E402
    fp8_dtype,
    launch_gemm_fp8_kernel,
)


def benchmark_torch_matmul(a, b, out, warmup, rep):
    """
    Benchmark torch.matmul on fp32 copies of fp8 inputs.

    Main feature:
        Measures the PyTorch baseline latency for GEMM.

    Inputs:
        a: torch.Tensor float32 of shape [m, k]
        b: torch.Tensor float32 of shape [n, k]
        out: torch.Tensor float32 of shape [m, n]
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute torch.matmul for benchmarking.

        Main feature:
            Computes out = a @ b^T.

        Inputs:
            None

        Outputs:
            None
        """
        torch.matmul(a, b.t(), out=out)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_matmul(a, b, out, warmup, rep):
    """
    Benchmark the Triton FP8 GEMM kernel.

    Main feature:
        Measures Triton kernel latency.

    Inputs:
        a: torch.Tensor fp8 of shape [m, k]
        b: torch.Tensor fp8 of shape [n, k]
        out: torch.Tensor float32 of shape [m, n]
        warmup: int32 scalar
        rep: int32 scalar

    Outputs:
        elapsed_ms: float scalar
    """
    def op():
        """
        Execute the Triton GEMM kernel for benchmarking.

        Main feature:
            Dispatches the Triton kernel with validated inputs.

        Inputs:
            None

        Outputs:
            None
        """
        launch_gemm_fp8_kernel(a, b, out)

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
    Compare PyTorch and Triton GEMM outputs and throughput.

    Main feature:
        Runs correctness and performance checks on GPU.

    Inputs:
        None

    Outputs:
        None
    """
    m = 4096
    n = 4096
    k = 4096
    warmup_iters = 10
    device = "cuda:0"

    torch.manual_seed(0)
    a = torch.rand((m, k), dtype=torch.float32, device=device).to(fp8_dtype)
    b = torch.rand((n, k), dtype=torch.float32, device=device).to(fp8_dtype)
    a_fp32 = a.float()
    b_fp32 = b.float()

    c_ref = torch.empty((m, n), dtype=torch.float32, device=device)
    c_triton = torch.empty_like(c_ref)

    torch_ms = benchmark_torch_matmul(a_fp32, b_fp32, c_ref, warmup_iters, rep=10)
    torch.matmul(a_fp32, b_fp32.t(), out=c_ref)

    flops = 2 * m * n * k
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12
    print(f"PyTorch time: {torch_ms * 1000:.3f} us | throughput: {torch_tflops:.3f} TFLOPS")

    triton_ms = benchmark_triton_matmul(a, b, c_triton, warmup_iters, rep=100)
    launch_gemm_fp8_kernel(a, b, c_triton)

    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    print(f"Triton time: {triton_ms * 1000:.3f} us | throughput: {triton_tflops:.3f} TFLOPS")

    error = relative_error(c_ref, c_triton)
    print(f"gemm_fp8_kernel relative error: {error:.3e}")


if __name__ == "__main__":
    run_tests()
