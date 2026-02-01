# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-31
# Purpose: Accuracy tests for Flash GQA Triton kernel (Training / Inference Prefill).
# ==============================================================================

import math
import os
import sys

import torch
import triton.testing

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from flash_gqa_triton_kernel import (  # noqa: E402
    launch_flash_gqa_forward_bf16_kernel,
)
from flash_gqa_backward_triton_kernel import (  # noqa: E402
    launch_flash_gqa_backward_bf16_kernel,
)


def flash_gqa_forward_reference(q, k, v, scale=None, out_dtype=None, return_lse=False):
    """
    Compute a reference Flash GQA forward output with stable softmax.

    Main feature:
        Recomputes attention probabilities with row-max stabilization and causal masking.

    Inputs:
        q: torch.Tensor bf16/float32 of shape [q_heads, seq_len_q, head_dim_qk]
        k: torch.Tensor bf16/float32 of shape [kv_heads, seq_len_kv, head_dim_qk]
        v: torch.Tensor bf16/float32 of shape [kv_heads, seq_len_kv, head_dim_v]
        scale: optional float scalar
        out_dtype: optional torch.dtype for the output
        return_lse: bool scalar, return log-sum-exp tensor when True

    Outputs:
        out: torch.Tensor out_dtype of shape [q_heads, seq_len_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seq_len_q] (if return_lse is True)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if out_dtype is None:
        out_dtype = q.dtype

    num_query_heads, seq_len_q, head_dim_qk = q.shape
    num_kv_heads, seq_len_kv, k_head_dim = k.shape
    _, _, head_dim_v = v.shape
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    group_size = num_query_heads // num_kv_heads

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)
    out_fp32 = torch.empty(
        (num_query_heads, seq_len_q, head_dim_v),
        device=q.device,
        dtype=torch.float32,
    )
    lse_fp32 = torch.empty((num_query_heads, seq_len_q), device=q.device, dtype=torch.float32)
    causal_mask = torch.triu(
        torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool),
        diagonal=1,
    )

    for kv_idx in range(num_kv_heads):
        head_start = kv_idx * group_size
        head_end = (kv_idx + 1) * group_size
        q_slice = q_fp32[head_start:head_end, :, :]
        k_slice = k_fp32[kv_idx, :, :]
        v_slice = v_fp32[kv_idx, :, :]

        scores = torch.matmul(q_slice, k_slice.transpose(0, 1)) * scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -float("inf"))
        row_max = torch.amax(scores, dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - row_max)
        probs = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
        lse_slice = torch.log(torch.sum(exp_scores, dim=-1)) + row_max.squeeze(-1)

        out_slice = torch.matmul(probs, v_slice)
        out_fp32[head_start:head_end, :, :] = out_slice
        lse_fp32[head_start:head_end, :] = lse_slice

    if return_lse:
        return out_fp32.to(out_dtype), lse_fp32
    return out_fp32.to(out_dtype)


def flash_gqa_backward_reference(q, k, v, o, grad_o, scale=None):
    """
    Compute a reference Flash GQA backward pass with stable softmax.

    Main feature:
        Recomputes attention probabilities for gradients with row-max stabilization and causal masking.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [kv_heads, seq_len_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [kv_heads, seq_len_kv, head_dim_v]
        o: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_v]
        grad_o: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_v]
        scale: optional float scalar

    Outputs:
        grad_q: torch.Tensor same dtype as q of shape [q_heads, seq_len_q, head_dim_qk]
        grad_k: torch.Tensor same dtype as k of shape [kv_heads, seq_len_kv, head_dim_qk]
        grad_v: torch.Tensor same dtype as v of shape [kv_heads, seq_len_kv, head_dim_v]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    num_query_heads, seq_len_q, head_dim_qk = q.shape
    num_kv_heads, seq_len_kv, k_head_dim = k.shape
    _, _, head_dim_v = v.shape
    _, _, o_head_dim = o.shape
    _, _, grad_o_head_dim = grad_o.shape
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    if o_head_dim != head_dim_v or grad_o_head_dim != head_dim_v:
        raise ValueError("O and grad_o must share head_dim_v with V")
    group_size = num_query_heads // num_kv_heads

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)
    grad_o_fp32 = grad_o.to(torch.float32)

    grad_q = torch.zeros_like(q_fp32)
    grad_k = torch.zeros_like(k_fp32)
    grad_v = torch.zeros_like(v_fp32)
    causal_mask = torch.triu(
        torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool),
        diagonal=1,
    )

    for kv_idx in range(num_kv_heads):
        head_start = kv_idx * group_size
        head_end = (kv_idx + 1) * group_size
        q_slice = q_fp32[head_start:head_end, :, :]
        grad_o_slice = grad_o_fp32[head_start:head_end, :, :]
        k_slice = k_fp32[kv_idx, :, :]
        v_slice = v_fp32[kv_idx, :, :]
        o_slice = o[head_start:head_end, :, :].to(torch.float32)

        scores = torch.matmul(q_slice, k_slice.transpose(0, 1)) * scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -float("inf"))
        row_max = torch.amax(scores, dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - row_max)
        probs = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

        grad_p = torch.matmul(grad_o_slice, v_slice.transpose(0, 1))
        grad_p_dot = torch.sum(grad_o_slice * o_slice, dim=-1, keepdim=True)
        grad_scores = (grad_p - grad_p_dot) * probs

        grad_q_slice = torch.matmul(grad_scores, k_slice) * scale
        grad_k_heads = torch.matmul(grad_scores.transpose(-1, -2), q_slice) * scale
        grad_v_heads = torch.matmul(probs.transpose(-1, -2), grad_o_slice)

        grad_q[head_start:head_end, :, :] = grad_q_slice
        grad_k[kv_idx, :, :] = grad_k_heads.sum(dim=0)
        grad_v[kv_idx, :, :] = grad_v_heads.sum(dim=0)

    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


def benchmark_torch_flash_gqa_backward(q, k, v, o, grad_o, warmup, rep):
    """
    Benchmark the reference Flash GQA backward implementation.

    Main feature:
        Measures backward latency with Triton's benchmarking utilities.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [kv_heads, seq_len_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [kv_heads, seq_len_kv, head_dim_v]
        o: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_v]
        grad_o: torch.Tensor bf16 of shape [q_heads, seq_len_q, head_dim_v]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Invoke the reference backward for benchmarking.

        Main feature:
            Computes Flash GQA backward gradients.

        Inputs:
            None

        Outputs:
            None
        """
        _ = flash_gqa_backward_reference(q, k, v, o, grad_o)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_flash_gqa_forward(q, k, v, warmup, rep):
    """
    Benchmark the Triton Flash GQA forward kernel.

    Main feature:
        Measures kernel latency using Triton's benchmarking utilities.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        warmup: int32 scalar, warmup iterations
        rep: int32 scalar, measurement repetitions

    Outputs:
        elapsed_ms: float scalar, mean latency in milliseconds
    """
    def op():
        """
        Invoke the Triton Flash GQA forward kernel.

        Main feature:
            Dispatches the Triton Flash GQA forward kernel.

        Inputs:
            None

        Outputs:
            None
        """
        _ = launch_flash_gqa_forward_bf16_kernel(q, k, v)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def bytes_accessed_forward(q, k, v, o, lse):
    """
    Estimate bytes moved by Flash GQA forward.

    Main feature:
        Approximates total bytes read and written.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
           [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
           [batch, q_heads, seqlen_q]

    Outputs:
        total_bytes: int64 scalar
    """
    return (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + o.numel() * o.element_size()
        + lse.numel() * lse.element_size()
    )


def flops_flash_gqa_forward(batch, q_heads, seqlen_q, seqlen_kv, head_dim_qk, head_dim_v):
    """
    Estimate FLOPs for Flash GQA forward (QK + PV).

    Main feature:
        Approximates multiply-add operations for attention.

    Inputs:
        batch: int32 scalar
        q_heads: int32 scalar
        seqlen_q: int32 scalar
        seqlen_kv: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output

    Outputs:
        total_flops: int64 scalar
    """
    flops_per_head = 2 * seqlen_q * seqlen_kv * (head_dim_qk + head_dim_v) // 2
    return int(batch * q_heads * flops_per_head)


def bytes_accessed_backward(q, k, v, lse, grad_o, grad_q, grad_k, grad_v):
    """
    Estimate bytes moved by Flash GQA backward.

    Main feature:
        Approximates total bytes read and written by the backward kernel.

    Inputs:
        q: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
           [batch, q_heads, seqlen_q]
        grad_o: torch.Tensor bf16 of shape [q_heads, seqlen_q, head_dim_v] or
           [batch, q_heads, seqlen_q, head_dim_v]
        grad_q: torch.Tensor float32 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k: torch.Tensor float32 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v: torch.Tensor float32 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]

    Outputs:
        total_bytes: int64 scalar
    """
    return (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + lse.numel() * lse.element_size()
        + grad_o.numel() * grad_o.element_size()
        + grad_q.numel() * grad_q.element_size()
        + grad_k.numel() * grad_k.element_size()
        + grad_v.numel() * grad_v.element_size()
    )


def flops_flash_gqa_backward(batch, q_heads, seqlen_q, seqlen_kv, head_dim_qk, head_dim_v):
    """
    Estimate FLOPs for Flash GQA backward (QK + dP + dQ + dK + dV).

    Main feature:
        Approximates multiply-add operations for backward matmuls.

    Inputs:
        batch: int32 scalar
        q_heads: int32 scalar
        seqlen_q: int32 scalar
        seqlen_kv: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output

    Outputs:
        total_flops: int64 scalar
    """
    flops_per_head = 2 * seqlen_q * seqlen_kv * (3 * head_dim_qk + 2 * head_dim_v) // 2
    return int(batch * q_heads * flops_per_head)


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


def generate_inputs(seq_len, head_dim_qk, head_dim_v, num_query_heads, num_kv_heads, device):
    """
    Generate Flash GQA backward inputs for testing.

    Main feature:
        Creates bf16 tensors and a matching forward output.

    Inputs:
        seq_len: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output
        num_query_heads: int32 scalar
        num_kv_heads: int32 scalar
        device: torch.device scalar

    Outputs:
        q: torch.Tensor bf16 of shape [num_query_heads, seq_len, head_dim_qk]
        k: torch.Tensor bf16 of shape [num_kv_heads, seq_len, head_dim_qk]
        v: torch.Tensor bf16 of shape [num_kv_heads, seq_len, head_dim_v]
        o: torch.Tensor bf16 of shape [num_query_heads, seq_len, head_dim_v]
        grad_o: torch.Tensor bf16 of shape [num_query_heads, seq_len, head_dim_v]
    """
    torch.manual_seed(0)
    q = torch.rand((num_query_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    k = torch.rand((num_kv_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    v = torch.rand((num_kv_heads, seq_len, head_dim_v), device=device, dtype=torch.bfloat16)
    o = flash_gqa_forward_reference(q, k, v, out_dtype=q.dtype)
    grad_o = torch.rand_like(o)
    return q, k, v, o, grad_o


def run_tests():
    """
    Validate Flash GQA forward and backward kernels.

    Main feature:
        Compares reference gradients with Triton backward and benchmarks Triton kernels.

    Inputs:
        None

    Outputs:
        None
    """
    seq_len = 4096
    head_dim_qk = 128
    head_dim_v = 128
    num_kv_heads = 1
    num_query_heads = 16
    warmup_iters = 20
    rep_iters = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q, k, v, o, grad_o = generate_inputs(
        seq_len,
        head_dim_qk,
        head_dim_v,
        num_query_heads,
        num_kv_heads,
        device,
    )
    grad_q_ref, grad_k_ref, grad_v_ref = flash_gqa_backward_reference(q, k, v, o, grad_o)

    out_triton, lse_triton = launch_flash_gqa_forward_bf16_kernel(q, k, v)
    out_ref, lse_ref = flash_gqa_forward_reference(q, k, v, out_dtype=q.dtype, return_lse=True)
    err_out = relative_error(out_ref.to(torch.float32), out_triton.to(torch.float32))
    err_lse = relative_error(lse_ref, lse_triton.to(torch.float32))
    print(f"forward output relative error: {err_out:.3e}")
    print(f"forward lse relative error: {err_lse:.3e}")

    grad_q_triton, grad_k_triton, grad_v_triton = launch_flash_gqa_backward_bf16_kernel(
        q,
        k,
        v,
        o,
        lse_ref,
        grad_o,
    )
    err_grad_q = relative_error(grad_q_ref.to(torch.float32), grad_q_triton.to(torch.float32))
    err_grad_k = relative_error(grad_k_ref.to(torch.float32), grad_k_triton.to(torch.float32))
    err_grad_v = relative_error(grad_v_ref.to(torch.float32), grad_v_triton.to(torch.float32))
    print(f"backward grad_q relative error: {err_grad_q:.3e}")
    print(f"backward grad_k relative error: {err_grad_k:.3e}")
    print(f"backward grad_v relative error: {err_grad_v:.3e}")

    forward_ms = benchmark_triton_flash_gqa_forward(
        q,
        k,
        v,
        warmup=warmup_iters,
        rep=rep_iters,
    )
    backward_ms = triton.testing.do_bench(
        lambda: launch_flash_gqa_backward_bf16_kernel(q, k, v, o, lse_ref, grad_o),
        warmup=warmup_iters,
        rep=rep_iters,
    )
    moved_bytes = bytes_accessed_forward(q, k, v, out_triton, lse_triton)
    forward_bw = moved_bytes / (forward_ms * 1e-3) / (1024 ** 4)
    total_flops = flops_flash_gqa_forward(
        batch=1,
        q_heads=q.shape[0],
        seqlen_q=q.shape[1],
        seqlen_kv=k.shape[1],
        head_dim_qk=q.shape[2],
        head_dim_v=v.shape[2],
    )
    forward_tflops = total_flops / (forward_ms * 1e-3) / 1e12
    backward_moved_bytes = bytes_accessed_backward(
        q,
        k,
        v,
        lse_ref,
        grad_o,
        grad_q_triton,
        grad_k_triton,
        grad_v_triton,
    )
    backward_bw = backward_moved_bytes / (backward_ms * 1e-3) / (1024 ** 4)
    backward_flops = flops_flash_gqa_backward(
        batch=1,
        q_heads=q.shape[0],
        seqlen_q=q.shape[1],
        seqlen_kv=k.shape[1],
        head_dim_qk=q.shape[2],
        head_dim_v=v.shape[2],
    )
    backward_tflops = backward_flops / (backward_ms * 1e-3) / 1e12
    print(
        "flash_gqa_forward_bf16_kernel: "
        f"{forward_ms:.5f} ms | "
        f"approx. bandwidth: {forward_bw:.6f} TB/s | "
        f"approx. TFLOPS: {forward_tflops:.6f}"
    )
    print(
        "flash_gqa_backward_bf16_kernel: "
        f"{backward_ms:.5f} ms | "
        f"approx. bandwidth: {backward_bw:.6f} TB/s | "
        f"approx. TFLOPS: {backward_tflops:.6f}"
    )


if __name__ == "__main__":
    run_tests()
