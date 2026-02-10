# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-06
# Purpose: Tests for Flash SGQA Triton kernels.
# ==============================================================================

import math
import os
import sys

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_src_dir = os.path.join(repo_root, "src", "python")
if python_src_dir not in sys.path:
    sys.path.append(python_src_dir)

from SparseGQA import (  # noqa: E402
    launch_flash_sgqa_backward_bf16_kernel,
    launch_flash_sgqa_forward_bf16_kernel,
)

NUM_SPARSE_KV = 512
PROFILE_WARMUP_ITERS = 20
PROFILE_TIMED_ITERS = 20
REFERENCE_BYPASS_MIN_SEQ_LEN = 16384
DEFAULT_SEQ_LEN = 32768


def build_sparse_kv_indices(batch, num_kv_heads, seqlen_q, seqlen_kv, device):
    """
    Build sparse KV index tensors with guaranteed recent-window coverage.

    Main feature:
        Generates int32 sparse index lists shared per KV head across that KV-head's query-head group,
        guarantees inclusion of the recent 16 causal tokens per query row (with edge clipping),
        samples any additional valid causal indices without replacement to avoid non-negative duplicates,
        and pads remaining slots strictly with -1.

    Inputs:
        batch: int32 scalar
        num_kv_heads: int32 scalar
        seqlen_q: int32 scalar
        seqlen_kv: int32 scalar
        device: torch.device scalar

    Outputs:
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]
    """
    if seqlen_kv < 1:
        raise ValueError("seqlen_kv must be >= 1")

    recent_window = 16
    if NUM_SPARSE_KV < recent_window:
        raise ValueError(f"NUM_SPARSE_KV must be >= {recent_window}")

    sparse_kv_indices = torch.full(
        (batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )
    if seqlen_q == 0:
        return sparse_kv_indices.contiguous()

    for q_pos in range(seqlen_q):
        upper = min(q_pos, seqlen_kv - 1)
        valid_unique_count = min(NUM_SPARSE_KV, upper + 1)
        recent_count = min(recent_window, valid_unique_count)
        recent_start = upper - recent_count + 1

        if recent_count > 0:
            recent_indices = torch.arange(
                recent_start,
                upper + 1,
                device=device,
                dtype=torch.int32,
            )
            sparse_kv_indices[:, :, q_pos, :recent_count] = recent_indices.view(1, 1, recent_count)

        remaining_valid_count = valid_unique_count - recent_count
        if remaining_valid_count > 0:
            candidate_count = recent_start
            random_scores = torch.rand(
                (batch, num_kv_heads, candidate_count),
                device=device,
                dtype=torch.float32,
            )
            random_order = torch.argsort(random_scores, dim=-1)
            sampled_positions = random_order[:, :, :remaining_valid_count]
            sampled_indices = sampled_positions.to(torch.int32)
            sparse_kv_indices[:, :, q_pos, recent_count : recent_count + remaining_valid_count] = sampled_indices

    return sparse_kv_indices.contiguous()


def flash_sgqa_forward_reference(
    q,
    k,
    v,
    sparse_kv_indices,
    scale=None,
    out_dtype=None,
    return_lse=False,
    causal=True,
):
    """
    Compute a reference Flash SGQA forward output with stable softmax.

    Main feature:
        Applies sparse KV selection per KV head and query token, then computes grouped-query
        attention using math-exact softmax with vectorized per-(batch, KV head) processing.

    Inputs:
        q: torch.Tensor bf16/float32 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16/float32 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16/float32 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]
        scale: optional float scalar, attention scaling factor
        out_dtype: optional torch.dtype for output
        return_lse: bool scalar, return log-sum-exp tensor when True
        causal: bool scalar, apply causal filtering on sparse indices when True

    Outputs:
        out: torch.Tensor out_dtype of shape [q_heads, seqlen_q, head_dim_v] or
             [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [q_heads, seqlen_q] or
             [batch, q_heads, seqlen_q] (if return_lse is True)
    """
    if q.dim() not in (3, 4):
        raise ValueError("Q must be 3D or 4D")
    if k.dim() != q.dim() or v.dim() != q.dim():
        raise ValueError("Q, K, and V must have matching ranks")
    if sparse_kv_indices.dtype != torch.int32:
        raise ValueError("sparse_kv_indices must be int32")
    if sparse_kv_indices.dim() != 4:
        raise ValueError("sparse_kv_indices must be 4D [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]")

    squeeze_output = q.dim() == 3
    if squeeze_output:
        q_batched = q.unsqueeze(0)
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)
    else:
        q_batched = q
        k_batched = k
        v_batched = v

    batch, q_heads, seqlen_q, head_dim_qk = q_batched.shape
    _, num_kv_heads, seqlen_kv, k_head_dim = k_batched.shape
    _, v_heads, v_seqlen, head_dim_v = v_batched.shape

    if sparse_kv_indices.shape != (batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV):
        raise ValueError(
            "sparse_kv_indices must have shape "
            f"[batch, num_kv_heads, seqlen_q, {NUM_SPARSE_KV}]"
        )
    if q_heads % num_kv_heads != 0:
        raise ValueError("q_heads must be divisible by num_kv_heads")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    if v_heads != num_kv_heads or v_seqlen != seqlen_kv:
        raise ValueError("K and V must share [num_kv_heads, seqlen_kv]")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)
    if out_dtype is None:
        out_dtype = q.dtype

    group_size = q_heads // num_kv_heads
    q_fp32 = q_batched.to(torch.float32)
    k_fp32 = k_batched.to(torch.float32)
    v_fp32 = v_batched.to(torch.float32)

    out_fp32 = torch.zeros((batch, q_heads, seqlen_q, head_dim_v), device=q.device, dtype=torch.float32)
    lse_fp32 = torch.full((batch, q_heads, seqlen_q), -float("inf"), device=q.device, dtype=torch.float32)
    q_positions = torch.arange(seqlen_q, device=q.device, dtype=torch.int64).view(seqlen_q, 1)
    max_kv_index = max(seqlen_kv - 1, 0)

    for batch_idx in range(batch):
        for kv_head_idx in range(num_kv_heads):
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            q_slice = q_fp32[batch_idx, head_start:head_end, :, :]
            k_slice = k_fp32[batch_idx, kv_head_idx, :, :]
            v_slice = v_fp32[batch_idx, kv_head_idx, :, :]
            sparse_idx = sparse_kv_indices[batch_idx, kv_head_idx, :, :].to(torch.int64)

            valid_mask = (sparse_idx >= 0) & (sparse_idx < seqlen_kv)
            if causal:
                valid_mask = valid_mask & (sparse_idx <= q_positions)

            safe_idx = sparse_idx.clamp(min=0, max=max_kv_index)
            k_sparse = k_slice.index_select(0, safe_idx.reshape(-1)).reshape(seqlen_q, NUM_SPARSE_KV, head_dim_qk)
            v_sparse = v_slice.index_select(0, safe_idx.reshape(-1)).reshape(seqlen_q, NUM_SPARSE_KV, head_dim_v)

            scores = torch.einsum("gqd,qsd->gqs", q_slice, k_sparse) * scale
            scores = torch.where(valid_mask.unsqueeze(0), scores, torch.full_like(scores, -float("inf")))

            row_max = torch.amax(scores, dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - row_max)
            exp_scores = torch.where(valid_mask.unsqueeze(0), exp_scores, torch.zeros_like(exp_scores))
            denom = torch.sum(exp_scores, dim=-1, keepdim=True)
            probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))

            out_fp32[batch_idx, head_start:head_end, :, :] = torch.einsum("gqs,qsv->gqv", probs, v_sparse)
            lse_fp32[batch_idx, head_start:head_end, :] = torch.where(
                denom.squeeze(-1) > 0,
                torch.log(denom.squeeze(-1)) + row_max.squeeze(-1),
                torch.full_like(row_max.squeeze(-1), -float("inf")),
            )

    out_cast = out_fp32.to(out_dtype)
    if squeeze_output:
        out_cast = out_cast.squeeze(0)
        lse_fp32 = lse_fp32.squeeze(0)
    if return_lse:
        return out_cast, lse_fp32
    return out_cast


def flash_sgqa_backward_reference(q, k, v, o, grad_o, sparse_kv_indices, scale=None, causal=True):
    """
    Compute a reference Flash SGQA backward pass.

    Main feature:
        Recomputes math-exact sparse attention probabilities and accumulates sparse gradients into
        K/V rows with vectorized per-(batch, KV head) processing.

    Inputs:
        q: torch.Tensor bf16/float32 of shape [q_heads, seqlen_q, head_dim_qk] or
           [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16/float32 of shape [num_kv_heads, seqlen_kv, head_dim_qk] or
           [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16/float32 of shape [num_kv_heads, seqlen_kv, head_dim_v] or
           [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o: torch.Tensor bf16/float32 of shape [q_heads, seqlen_q, head_dim_v] or
           [batch, q_heads, seqlen_q, head_dim_v]
        grad_o: torch.Tensor bf16/float32 of shape [q_heads, seqlen_q, head_dim_v] or
                [batch, q_heads, seqlen_q, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]
        scale: optional float scalar, attention scaling factor
        causal: bool scalar, apply causal filtering on sparse indices when True

    Outputs:
        grad_q: torch.Tensor same dtype as q with shape [q_heads, seqlen_q, head_dim_qk] or
                [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k: torch.Tensor same dtype as k with shape [num_kv_heads, seqlen_kv, head_dim_qk] or
                [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v: torch.Tensor same dtype as v with shape [num_kv_heads, seqlen_kv, head_dim_v] or
                [batch, num_kv_heads, seqlen_kv, head_dim_v]
    """
    if q.dim() not in (3, 4):
        raise ValueError("Q must be 3D or 4D")
    if k.dim() != q.dim() or v.dim() != q.dim() or o.dim() != q.dim() or grad_o.dim() != q.dim():
        raise ValueError("Q, K, V, O, and grad_o must have matching ranks")
    if sparse_kv_indices.dtype != torch.int32:
        raise ValueError("sparse_kv_indices must be int32")
    if sparse_kv_indices.dim() != 4:
        raise ValueError("sparse_kv_indices must be 4D [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]")

    squeeze_output = q.dim() == 3
    if squeeze_output:
        q_batched = q.unsqueeze(0)
        k_batched = k.unsqueeze(0)
        v_batched = v.unsqueeze(0)
        o_batched = o.unsqueeze(0)
        grad_o_batched = grad_o.unsqueeze(0)
    else:
        q_batched = q
        k_batched = k
        v_batched = v
        o_batched = o
        grad_o_batched = grad_o

    batch, q_heads, seqlen_q, head_dim_qk = q_batched.shape
    _, num_kv_heads, seqlen_kv, k_head_dim = k_batched.shape
    _, v_heads, v_seqlen, head_dim_v = v_batched.shape

    if sparse_kv_indices.shape != (batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV):
        raise ValueError(
            "sparse_kv_indices must have shape "
            f"[batch, num_kv_heads, seqlen_q, {NUM_SPARSE_KV}]"
        )
    if q_heads % num_kv_heads != 0:
        raise ValueError("q_heads must be divisible by num_kv_heads")
    if k_head_dim != head_dim_qk:
        raise ValueError("Q and K must share head_dim_qk")
    if v_heads != num_kv_heads or v_seqlen != seqlen_kv:
        raise ValueError("K and V must share [num_kv_heads, seqlen_kv]")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim_qk)

    group_size = q_heads // num_kv_heads
    q_fp32 = q_batched.to(torch.float32)
    k_fp32 = k_batched.to(torch.float32)
    v_fp32 = v_batched.to(torch.float32)
    o_fp32 = o_batched.to(torch.float32)
    grad_o_fp32 = grad_o_batched.to(torch.float32)

    grad_q_fp32 = torch.zeros_like(q_fp32)
    grad_k_fp32 = torch.zeros_like(k_fp32)
    grad_v_fp32 = torch.zeros_like(v_fp32)
    q_positions = torch.arange(seqlen_q, device=q.device, dtype=torch.int64).view(seqlen_q, 1)
    max_kv_index = max(seqlen_kv - 1, 0)

    for batch_idx in range(batch):
        for kv_head_idx in range(num_kv_heads):
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            q_slice = q_fp32[batch_idx, head_start:head_end, :, :]
            o_slice = o_fp32[batch_idx, head_start:head_end, :, :]
            grad_o_slice = grad_o_fp32[batch_idx, head_start:head_end, :, :]
            k_slice = k_fp32[batch_idx, kv_head_idx, :, :]
            v_slice = v_fp32[batch_idx, kv_head_idx, :, :]
            sparse_idx = sparse_kv_indices[batch_idx, kv_head_idx, :, :].to(torch.int64)

            valid_mask = (sparse_idx >= 0) & (sparse_idx < seqlen_kv)
            if causal:
                valid_mask = valid_mask & (sparse_idx <= q_positions)

            safe_idx = sparse_idx.clamp(min=0, max=max_kv_index)
            k_sparse = k_slice.index_select(0, safe_idx.reshape(-1)).reshape(seqlen_q, NUM_SPARSE_KV, head_dim_qk)
            v_sparse = v_slice.index_select(0, safe_idx.reshape(-1)).reshape(seqlen_q, NUM_SPARSE_KV, head_dim_v)

            scores = torch.einsum("gqd,qsd->gqs", q_slice, k_sparse) * scale
            scores = torch.where(valid_mask.unsqueeze(0), scores, torch.full_like(scores, -float("inf")))

            row_max = torch.amax(scores, dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - row_max)
            exp_scores = torch.where(valid_mask.unsqueeze(0), exp_scores, torch.zeros_like(exp_scores))
            denom = torch.sum(exp_scores, dim=-1, keepdim=True)
            probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))

            grad_p = torch.einsum("gqv,qsv->gqs", grad_o_slice, v_sparse)
            grad_dot = torch.sum(grad_o_slice * o_slice, dim=-1, keepdim=True)
            grad_scores = (grad_p - grad_dot) * probs * scale

            grad_q_fp32[batch_idx, head_start:head_end, :, :] = torch.einsum("gqs,qsd->gqd", grad_scores, k_sparse)

            grad_k_update = torch.einsum("gqs,gqd->qsd", grad_scores, q_slice)
            grad_v_update = torch.einsum("gqs,gqv->qsv", probs, grad_o_slice)

            safe_idx_flat = safe_idx.reshape(-1)
            valid_flat = valid_mask.reshape(-1)
            grad_k_flat = grad_k_update.reshape(-1, head_dim_qk)
            grad_v_flat = grad_v_update.reshape(-1, head_dim_v)
            grad_k_fp32[batch_idx, kv_head_idx, :, :].index_add_(0, safe_idx_flat[valid_flat], grad_k_flat[valid_flat])
            grad_v_fp32[batch_idx, kv_head_idx, :, :].index_add_(0, safe_idx_flat[valid_flat], grad_v_flat[valid_flat])

    grad_q = grad_q_fp32.to(q.dtype)
    grad_k = grad_k_fp32.to(k.dtype)
    grad_v = grad_v_fp32.to(v.dtype)
    if squeeze_output:
        return grad_q.squeeze(0), grad_k.squeeze(0), grad_v.squeeze(0)
    return grad_q, grad_k, grad_v


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


def profile_cuda_event(op, warmup_iters, timed_iters):
    """
    Profile a CUDA operation using CUDA events.

    Main feature:
        Runs warmup iterations and timed iterations, then returns mean/min/max latency in milliseconds.

    Inputs:
        op: callable with no inputs and None output
        warmup_iters: int32 scalar, number of warmup iterations
        timed_iters: int32 scalar, number of timed iterations

    Outputs:
        mean_ms: float scalar, mean latency in milliseconds
        min_ms: float scalar, minimum latency in milliseconds
        max_ms: float scalar, maximum latency in milliseconds
    """
    for _ in range(warmup_iters):
        op()
    torch.cuda.synchronize()

    elapsed_ms = []
    for _ in range(timed_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        op()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms.append(start_event.elapsed_time(end_event))

    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    min_ms = min(elapsed_ms)
    max_ms = max(elapsed_ms)
    return mean_ms, min_ms, max_ms


def bytes_accessed_sgqa_forward(q, k, v, sparse_kv_indices, o, lse):
    """
    Estimate bytes moved by Flash SGQA forward.

    Main feature:
        Uses tensor-I/O approximation including sparse index bytes.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]
        o: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [batch, q_heads, seqlen_q]

    Outputs:
        total_bytes: int64 scalar
    """
    return (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + sparse_kv_indices.numel() * sparse_kv_indices.element_size()
        + o.numel() * o.element_size()
        + lse.numel() * lse.element_size()
    )


def bytes_accessed_sgqa_backward(q, k, v, o, lse, grad_o, sparse_kv_indices, grad_q, grad_k, grad_v):
    """
    Estimate bytes moved by Flash SGQA backward.

    Main feature:
        Uses tensor-I/O approximation including sparse index bytes.

    Inputs:
        q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_qk]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_v]
        o: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]
        lse: torch.Tensor float32 of shape [batch, q_heads, seqlen_q]
        grad_o: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seqlen_q, NUM_SPARSE_KV]
        grad_q: torch.Tensor bf16 of shape [batch, q_heads, seqlen_q, head_dim_qk]
        grad_k: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_qk]
        grad_v: torch.Tensor bf16 of shape [batch, num_kv_heads, seqlen_kv, head_dim_v]

    Outputs:
        total_bytes: int64 scalar
    """
    return (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + o.numel() * o.element_size()
        + lse.numel() * lse.element_size()
        + grad_o.numel() * grad_o.element_size()
        + sparse_kv_indices.numel() * sparse_kv_indices.element_size()
        + grad_q.numel() * grad_q.element_size()
        + grad_k.numel() * grad_k.element_size()
        + grad_v.numel() * grad_v.element_size()
    )


def flops_flash_sgqa_forward_fixed_sparse(batch, q_heads, seqlen_q, head_dim_qk, head_dim_v, num_sparse_kv):
    """
    Estimate FLOPs for Flash SGQA forward with fixed sparse length.

    Main feature:
        Uses fixed sparse token count per query as the FLOPs basis.

    Inputs:
        batch: int32 scalar
        q_heads: int32 scalar
        seqlen_q: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output
        num_sparse_kv: int32 scalar, fixed sparse KV count per query

    Outputs:
        total_flops: int64 scalar
    """
    return int(2 * batch * q_heads * seqlen_q * num_sparse_kv * (head_dim_qk + head_dim_v))


def flops_flash_sgqa_backward_fixed_sparse(batch, q_heads, seqlen_q, head_dim_qk, head_dim_v, num_sparse_kv):
    """
    Estimate FLOPs for Flash SGQA backward with fixed sparse length.

    Main feature:
        Uses fixed sparse token count per query and standard attention backward matmul terms.

    Inputs:
        batch: int32 scalar
        q_heads: int32 scalar
        seqlen_q: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output
        num_sparse_kv: int32 scalar, fixed sparse KV count per query

    Outputs:
        total_flops: int64 scalar
    """
    return int(2 * batch * q_heads * seqlen_q * num_sparse_kv * (3 * head_dim_qk + 2 * head_dim_v))


def generate_inputs(batch, seq_len, head_dim_qk, head_dim_v, num_query_heads, num_kv_heads, device):
    """
    Generate SGQA test inputs and sparse indices.

    Main feature:
        Produces bf16 Q/K/V, grad_o, and sparse index tensors and asserts generated sparse
        indices satisfy causality, required recent-window coverage, and non-negative uniqueness.

    Inputs:
        batch: int32 scalar
        seq_len: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output
        num_query_heads: int32 scalar
        num_kv_heads: int32 scalar
        device: torch.device scalar

    Outputs:
        q: torch.Tensor bf16 of shape [batch, num_query_heads, seq_len, head_dim_qk]
        k: torch.Tensor bf16 of shape [batch, num_kv_heads, seq_len, head_dim_qk]
        v: torch.Tensor bf16 of shape [batch, num_kv_heads, seq_len, head_dim_v]
        grad_o: torch.Tensor bf16 of shape [batch, num_query_heads, seq_len, head_dim_v]
        sparse_kv_indices: torch.Tensor int32 of shape [batch, num_kv_heads, seq_len, NUM_SPARSE_KV]
    """
    torch.manual_seed(0)
    q = torch.rand((batch, num_query_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    k = torch.rand((batch, num_kv_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    v = torch.rand((batch, num_kv_heads, seq_len, head_dim_v), device=device, dtype=torch.bfloat16)
    sparse_kv_indices = build_sparse_kv_indices(batch, num_kv_heads, seq_len, seq_len, device)

    q_positions = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, 1, seq_len, 1)
    in_range_mask = (sparse_kv_indices >= 0) & (sparse_kv_indices < seq_len)
    causal_violation_mask = in_range_mask & (sparse_kv_indices > q_positions)
    if torch.any(causal_violation_mask):
        raise AssertionError("build_sparse_kv_indices produced non-causal valid sparse indices")

    sorted_indices, _ = torch.sort(sparse_kv_indices, dim=-1)
    duplicate_non_negative = (sorted_indices[..., 1:] == sorted_indices[..., :-1]) & (sorted_indices[..., 1:] >= 0)
    if torch.any(duplicate_non_negative):
        raise AssertionError("build_sparse_kv_indices produced duplicated non-negative sparse indices")

    recent_window = 16
    q_upper = torch.minimum(q_positions, torch.full_like(q_positions, seq_len - 1))
    recent_offsets = torch.arange(recent_window, device=device, dtype=torch.int32).view(1, 1, 1, recent_window)
    recent_count = torch.minimum(torch.full_like(q_upper, recent_window), q_upper + 1)
    recent_start = q_upper - (recent_count - 1)
    required_recent = recent_start + recent_offsets
    required_recent_valid = recent_offsets < recent_count
    required_present = (sparse_kv_indices.unsqueeze(-1) == required_recent.unsqueeze(-2)).any(dim=-2)
    required_present = torch.where(required_recent_valid, required_present, torch.ones_like(required_present))
    if not torch.all(required_present):
        raise AssertionError("build_sparse_kv_indices did not include all required recent-window tokens")

    grad_o = torch.rand((batch, num_query_heads, seq_len, head_dim_v), device=device, dtype=torch.bfloat16)
    return q, k, v, grad_o, sparse_kv_indices


def validate_forward_output_buffer_contract(device):
    """
    Validate forward launcher behavior for provided output buffers.

    Main feature:
        Checks that non-contiguous provided `out` or `lse` tensors are rejected and that contiguous
        provided buffers are returned in-place without allocation replacement.

    Inputs:
        device: torch.device scalar

    Outputs:
        None
    """
    batch = 1
    seq_len = 64
    head_dim_qk = 128
    head_dim_v = 128
    num_query_heads = 16
    num_kv_heads = 1

    torch.manual_seed(0)
    q = torch.rand((batch, num_query_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    k = torch.rand((batch, num_kv_heads, seq_len, head_dim_qk), device=device, dtype=torch.bfloat16)
    v = torch.rand((batch, num_kv_heads, seq_len, head_dim_v), device=device, dtype=torch.bfloat16)
    sparse_kv_indices = build_sparse_kv_indices(batch, num_kv_heads, seq_len, seq_len, device)

    out_non_contiguous = torch.empty(
        (batch, num_query_heads, head_dim_v, seq_len),
        device=device,
        dtype=torch.bfloat16,
    ).transpose(-1, -2)
    if out_non_contiguous.is_contiguous():
        raise AssertionError("expected out_non_contiguous to be non-contiguous")
    try:
        _ = launch_flash_sgqa_forward_bf16_kernel(
            q,
            k,
            v,
            sparse_kv_indices,
            num_sparse_kv=NUM_SPARSE_KV,
            out=out_non_contiguous,
            causal=True,
        )
        raise AssertionError("expected non-contiguous out buffer to raise ValueError")
    except ValueError as error:
        if "Output must be contiguous when provided" not in str(error):
            raise

    out_contiguous = torch.empty((batch, num_query_heads, seq_len, head_dim_v), device=device, dtype=torch.bfloat16)
    lse_non_contiguous = torch.empty((batch, num_query_heads, seq_len, 2), device=device, dtype=torch.float32)[..., 0]
    if lse_non_contiguous.is_contiguous():
        raise AssertionError("expected lse_non_contiguous to be non-contiguous")
    try:
        _ = launch_flash_sgqa_forward_bf16_kernel(
            q,
            k,
            v,
            sparse_kv_indices,
            num_sparse_kv=NUM_SPARSE_KV,
            out=out_contiguous,
            lse=lse_non_contiguous,
            causal=True,
        )
        raise AssertionError("expected non-contiguous lse buffer to raise ValueError")
    except ValueError as error:
        if "LSE must be contiguous when provided" not in str(error):
            raise

    lse_contiguous = torch.empty((batch, num_query_heads, seq_len), device=device, dtype=torch.float32)
    out_returned, lse_returned = launch_flash_sgqa_forward_bf16_kernel(
        q,
        k,
        v,
        sparse_kv_indices,
        num_sparse_kv=NUM_SPARSE_KV,
        out=out_contiguous,
        lse=lse_contiguous,
        causal=True,
    )
    if out_returned.data_ptr() != out_contiguous.data_ptr():
        raise AssertionError("forward launcher did not return the provided contiguous out buffer")
    if lse_returned.data_ptr() != lse_contiguous.data_ptr():
        raise AssertionError("forward launcher did not return the provided contiguous lse buffer")


def validate_forward_zero_query_guard(device):
    """
    Validate SGQA forward launcher behavior when query sequence length is zero.

    Main feature:
        Verifies the launcher skips Triton dispatch for zero-query inputs and returns
        correctly shaped empty output tensors without raising runtime errors.

    Inputs:
        device: torch.device scalar

    Outputs:
        None
    """
    batch = 1
    seqlen_q = 0
    seqlen_kv = 32
    head_dim_qk = 128
    head_dim_v = 128
    num_query_heads = 16
    num_kv_heads = 1

    torch.manual_seed(0)
    q = torch.empty((batch, num_query_heads, seqlen_q, head_dim_qk), device=device, dtype=torch.bfloat16)
    k = torch.rand((batch, num_kv_heads, seqlen_kv, head_dim_qk), device=device, dtype=torch.bfloat16)
    v = torch.rand((batch, num_kv_heads, seqlen_kv, head_dim_v), device=device, dtype=torch.bfloat16)
    sparse_kv_indices = build_sparse_kv_indices(batch, num_kv_heads, seqlen_q, seqlen_kv, device)

    out, lse = launch_flash_sgqa_forward_bf16_kernel(
        q,
        k,
        v,
        sparse_kv_indices,
        num_sparse_kv=NUM_SPARSE_KV,
        causal=True,
    )

    expected_out_shape = (batch, num_query_heads, seqlen_q, head_dim_v)
    expected_lse_shape = (batch, num_query_heads, seqlen_q)
    if out.shape != expected_out_shape:
        raise AssertionError(f"unexpected zero-query output shape: {out.shape} vs {expected_out_shape}")
    if lse.shape != expected_lse_shape:
        raise AssertionError(f"unexpected zero-query lse shape: {lse.shape} vs {expected_lse_shape}")
    if out.numel() != 0:
        raise AssertionError(f"expected zero-query output to be empty, got numel={out.numel()}")
    if lse.numel() != 0:
        raise AssertionError(f"expected zero-query lse to be empty, got numel={lse.numel()}")


def run_single_case(
    case_name,
    batch,
    seq_len,
    head_dim_qk,
    head_dim_v,
    num_query_heads,
    num_kv_heads,
    device,
):
    """
    Run one SGQA correctness case and assert relative-error thresholds.

    Main feature:
        Runs Triton SGQA forward/backward and compares against Torch references only when
        sequence length is below the configured reference-bypass threshold.

    Inputs:
        case_name: string scalar
        batch: int32 scalar
        seq_len: int32 scalar
        head_dim_qk: int32 scalar, head dimension for Q/K
        head_dim_v: int32 scalar, head dimension for V/output
        num_query_heads: int32 scalar
        num_kv_heads: int32 scalar
        device: torch.device scalar
    Outputs:
        None
    """
    q, k, v, grad_o, sparse_kv_indices = generate_inputs(
        batch,
        seq_len,
        head_dim_qk,
        head_dim_v,
        num_query_heads,
        num_kv_heads,
        device,
    )

    q_input = q
    k_input = k
    v_input = v
    grad_o_input = grad_o
    skip_reference = seq_len >= REFERENCE_BYPASS_MIN_SEQ_LEN

    out_triton, lse_triton = launch_flash_sgqa_forward_bf16_kernel(
        q_input,
        k_input,
        v_input,
        sparse_kv_indices,
        num_sparse_kv=NUM_SPARSE_KV,
        causal=True,
    )
    o_input = out_triton
    lse_for_backward = lse_triton

    if skip_reference:
        print(
            f"[{case_name}] reference bypass enabled: seq_len={seq_len} >= "
            f"{REFERENCE_BYPASS_MIN_SEQ_LEN}; using Triton-only forward/backward path"
        )
    else:
        out_ref, lse_ref = flash_sgqa_forward_reference(
            q_input,
            k_input,
            v_input,
            sparse_kv_indices,
            out_dtype=q_input.dtype,
            return_lse=True,
            causal=True,
        )
        o_input = out_ref
        lse_for_backward = lse_ref

        grad_q_ref, grad_k_ref, grad_v_ref = flash_sgqa_backward_reference(
            q_input,
            k_input,
            v_input,
            o_input,
            grad_o_input,
            sparse_kv_indices,
            causal=True,
        )

    grad_q_triton, grad_k_triton, grad_v_triton = launch_flash_sgqa_backward_bf16_kernel(
        q_input,
        k_input,
        v_input,
        o_input,
        lse_for_backward,
        grad_o_input,
        sparse_kv_indices,
        num_sparse_kv=NUM_SPARSE_KV,
        causal=True,
    )

    if not skip_reference:
        err_out = relative_error(out_ref.to(torch.float32), out_triton.to(torch.float32))
        lse_ref_fp32 = lse_ref.to(torch.float32)
        lse_triton_fp32 = lse_triton.to(torch.float32)
        lse_finite_mask = torch.isfinite(lse_ref_fp32) & torch.isfinite(lse_triton_fp32)
        if not torch.equal(torch.isfinite(lse_ref_fp32), torch.isfinite(lse_triton_fp32)):
            raise AssertionError(f"{case_name}: finite mask mismatch in LSE outputs")
        if torch.any(lse_finite_mask):
            err_lse = relative_error(lse_ref_fp32[lse_finite_mask], lse_triton_fp32[lse_finite_mask])
        else:
            err_lse = 0.0
        err_grad_q = relative_error(grad_q_ref.to(torch.float32), grad_q_triton.to(torch.float32))
        err_grad_k = relative_error(grad_k_ref.to(torch.float32), grad_k_triton.to(torch.float32))
        err_grad_v = relative_error(grad_v_ref.to(torch.float32), grad_v_triton.to(torch.float32))

        print(f"[{case_name}] forward output relative error: {err_out:.3e}")
        print(f"[{case_name}] forward lse relative error: {err_lse:.3e}")
        print(f"[{case_name}] backward grad_q relative error: {err_grad_q:.3e}")
        print(f"[{case_name}] backward grad_k relative error: {err_grad_k:.3e}")
        print(f"[{case_name}] backward grad_v relative error: {err_grad_v:.3e}")

        if err_out > 3.0e-2:
            raise AssertionError(f"{case_name}: forward output relative error too high: {err_out:.3e}")
        if err_lse > 3.0e-2:
            raise AssertionError(f"{case_name}: forward lse relative error too high: {err_lse:.3e}")
        if err_grad_q > 6.0e-2:
            raise AssertionError(f"{case_name}: backward grad_q relative error too high: {err_grad_q:.3e}")
        if err_grad_k > 6.0e-2:
            raise AssertionError(f"{case_name}: backward grad_k relative error too high: {err_grad_k:.3e}")
        if err_grad_v > 6.0e-2:
            raise AssertionError(f"{case_name}: backward grad_v relative error too high: {err_grad_v:.3e}")

    def forward_op():
        """
        Launch SGQA forward kernel for CUDA-event profiling.

        Main feature:
            Executes one forward-kernel invocation with batched tensors.

        Inputs:
            None

        Outputs:
            None
        """
        _ = launch_flash_sgqa_forward_bf16_kernel(
            q_input,
            k_input,
            v_input,
            sparse_kv_indices,
            num_sparse_kv=NUM_SPARSE_KV,
            causal=True,
        )

    def backward_op():
        """
        Launch SGQA backward kernel for CUDA-event profiling.

        Main feature:
            Executes one backward-kernel invocation with batched tensors.

        Inputs:
            None

        Outputs:
            None
        """
        _ = launch_flash_sgqa_backward_bf16_kernel(
            q_input,
            k_input,
            v_input,
            o_input,
            lse_for_backward,
            grad_o_input,
            sparse_kv_indices,
            num_sparse_kv=NUM_SPARSE_KV,
            causal=True,
        )

    forward_ms, forward_min_ms, forward_max_ms = profile_cuda_event(
        forward_op,
        warmup_iters=PROFILE_WARMUP_ITERS,
        timed_iters=PROFILE_TIMED_ITERS,
    )
    backward_ms, backward_min_ms, backward_max_ms = profile_cuda_event(
        backward_op,
        warmup_iters=PROFILE_WARMUP_ITERS,
        timed_iters=PROFILE_TIMED_ITERS,
    )

    forward_bytes = bytes_accessed_sgqa_forward(
        q_input,
        k_input,
        v_input,
        sparse_kv_indices,
        out_triton,
        lse_triton,
    )
    backward_bytes = bytes_accessed_sgqa_backward(
        q_input,
        k_input,
        v_input,
        o_input,
        lse_for_backward,
        grad_o_input,
        sparse_kv_indices,
        grad_q_triton,
        grad_k_triton,
        grad_v_triton,
    )
    forward_flops = flops_flash_sgqa_forward_fixed_sparse(
        batch=q_input.shape[0],
        q_heads=q_input.shape[1],
        seqlen_q=q_input.shape[2],
        head_dim_qk=q_input.shape[3],
        head_dim_v=v_input.shape[3],
        num_sparse_kv=NUM_SPARSE_KV,
    )
    backward_flops = flops_flash_sgqa_backward_fixed_sparse(
        batch=q_input.shape[0],
        q_heads=q_input.shape[1],
        seqlen_q=q_input.shape[2],
        head_dim_qk=q_input.shape[3],
        head_dim_v=v_input.shape[3],
        num_sparse_kv=NUM_SPARSE_KV,
    )

    forward_bw_tb_s = forward_bytes / (forward_ms * 1e-3) / (1024 ** 4)
    backward_bw_tb_s = backward_bytes / (backward_ms * 1e-3) / (1024 ** 4)
    forward_tflops = forward_flops / (forward_ms * 1e-3) / 1e12
    backward_tflops = backward_flops / (backward_ms * 1e-3) / 1e12

    print(
        f"[{case_name}] forward latency (ms): mean={forward_ms:.4f}, "
        f"min={forward_min_ms:.4f}, max={forward_max_ms:.4f}"
    )
    print(
        f"[{case_name}] forward approx bandwidth: {forward_bw_tb_s:.6f} TB/s | "
        f"approx TFLOPS: {forward_tflops:.6f}"
    )
    print(
        f"[{case_name}] backward latency (ms): mean={backward_ms:.4f}, "
        f"min={backward_min_ms:.4f}, max={backward_max_ms:.4f}"
    )
    print(
        f"[{case_name}] backward approx bandwidth: {backward_bw_tb_s:.6f} TB/s | "
        f"approx TFLOPS: {backward_tflops:.6f}"
    )


def run_tests():
    """
    Validate Flash SGQA forward and backward kernels.

    Main feature:
        Validates buffer contracts, then runs the default SGQA case and automatically bypasses
        Torch references for long-sequence runs.

    Inputs:
        None

    Outputs:
        None
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Flash SGQA Triton tests")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    validate_forward_output_buffer_contract(device)
    validate_forward_zero_query_guard(device)

    run_single_case(
        case_name="batched_32k",
        batch=1,
        seq_len=DEFAULT_SEQ_LEN,
        head_dim_qk=128,
        head_dim_v=128,
        num_query_heads=16,
        num_kv_heads=1,
        device=device,
    )


if __name__ == "__main__":
    run_tests()
