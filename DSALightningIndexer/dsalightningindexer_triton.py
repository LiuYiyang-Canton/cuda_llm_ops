import torch
import triton
import triton.testing
import triton.language as tl

# Tiling meta-parameters (tweak as needed).
BLOCK_SQ = 1  # query tile size along sequence dimension
BLOCK_SK = 4096  # key tile size along sequence dimension

@triton.jit
def lightningindexer(
    q: torch.Tensor, k: torch.Tensor, weight: torch.Tensor, out_idx_ptr, topk: tl.constexpr, batch_size,
    num_heads, head_dim, s_q, s_k, BLOCK_SQ: tl.constexpr, BLOCK_SK: tl.constexpr,
    BLOCK_HEAD: tl.constexpr, BLOCK_D: tl.constexpr
) -> torch.Tensor:
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = batch_size
    n = num_heads
    d = head_dim

    sq_offsets = pid_q * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sk_offsets = pid_k * BLOCK_SK + tl.arange(0, BLOCK_SK)
    dim_offsets = tl.arange(0, BLOCK_D)

    sq_mask = sq_offsets < s_q
    sk_mask = sk_offsets < s_k

    weighted_scores = tl.zeros((BLOCK_SQ, BLOCK_SK), dtype = tl.float32)
    for h in range(BLOCK_HEAD):
        q_ptr = q + sq_offsets * b * n * d + pid_b * n * d + h * d + dim_offsets
        q_tile = tl.load(q_ptr, sq_mask, other=0.0)
        k_ptr = k + sk_offsets[:, None] * b * n * d + pid_b * n * d + h * d + dim_offsets
        k_tile = tl.load(k_ptr, sk_mask[:, None], other=0.0)

        # scores_h = tl.dot(q_tile, tl.trans(k_tile))
        scores_h = tl.sum(q_tile[None, :].to(tl.float32) * k_tile.to(tl.float32), axis=1) # (t,)
        scores_h = tl.maximum(scores_h, 0.0) # (t,)

        weight_ptr = weight + sq_offsets[:, None] * b * n + pid_b * n + h
        weight_tile = tl.load(weight_ptr, sq_mask, other=0.0) # (1,)

        weighted_scores += scores_h * weight_tile

    # Local top-k selection within this key tile.
    weighted_scores = tl.where(sk_mask[None, :], weighted_scores, -float("inf"))
    topk_idx = tl.full((BLOCK_SQ, topk), -1, dtype=tl.int32)
    col_range = tl.arange(0, topk)[None, :]

    for i in range(topk):
        v = tl.max(weighted_scores, axis=1)
        is_max = weighted_scores == v[:, None]
        j = tl.argmax(is_max, axis=1)
        topk_idx = tl.where(col_range == i, pid_k * BLOCK_SK + j[:, None], topk_idx)
        weighted_scores = tl.where(is_max, -float("inf"), weighted_scores)


    # Write local topk indices to global memory.
    out_ptr = (
        out_idx_ptr
        + sq_offsets[:, None] * (b * topk)
        + pid_b * topk
        + tl.arange(0, topk)[None, :]
    )
    tl.store(out_ptr, topk_idx, mask=sq_mask[:, None])


def torch_lightningindexer(
    q: torch.Tensor, k: torch.Tensor, weight: torch.Tensor, topk: int
) -> torch.Tensor:
    """Reference lightning indexer: S = ReLU(Q K^T), P = weight @ S (reduce heads), then top-k over seq."""
    expected_dtype = torch.float8_e4m3fn
    if q.dtype != expected_dtype or k.dtype != expected_dtype or weight.dtype != expected_dtype:
        raise ValueError(f"Expected fp8 tensors ({expected_dtype}) for q, k, and weight")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q should be (s, b, n, d) and k should be (t, b, n, d)")

    s_q, b, n, d = q.shape
    s_k, _, _, _ = k.shape

    # S = Q @ K^T (reduce d) -> (s, t, b, n), then ReLU.
    q_f = q.float()
    k_f = k.float()
    s_scores = torch.einsum("sbnd,tbnd->bnst", q_f, k_f)
    s_scores = torch.relu(s_scores)

    # P = weight @ S (reduce n) -> (s, t, b)
    p_scores = torch.einsum("sbn,bnst->bst", weight.float(), s_scores)

    # Top-k along the sequence length dimension (t).
    topk_idx = torch.topk(p_scores, k=topk, dim=2).indices
    return topk_idx


def launch_dsalightningindexer(
    q: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    topk: int,
    out_idx: torch.Tensor,
    *,
    block_sq: int = BLOCK_SQ,
    block_sk: int = BLOCK_SK,
) -> None:
    """Launch the Triton kernel assuming contiguous tensors."""
    if not (q.is_contiguous() and k.is_contiguous() and weight.is_contiguous()):
        raise ValueError("lightning indexer kernel expects contiguous inputs")
    if not out_idx.is_contiguous():
        raise ValueError("Output buffers must be contiguous")
    if out_idx.dtype != torch.int64:
        raise ValueError("Output indices must be int64")

    grid = (triton.cdiv(q.shape[0], block_sq), triton.cdiv(k.shape[0], block_sk), q.shape[1])
    lightningindexer[grid](q, k, weight, out_idx, topk, q.shape[1], q.shape[2], q.shape[3], q.shape[0], k.shape[0], 
                           BLOCK_SQ, BLOCK_SK, q.shape[2], q.shape[-1])



def benchmark_torch_lightningindexer(
    q: torch.Tensor, k: torch.Tensor, weight: torch.Tensor, topk: int, warmup: int, rep: int
) -> float:
    """Benchmark the torch reference using Triton's timing helper."""
    def op():
        torch_lightningindexer(q, k, weight, topk)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_lightningindexer(
    q: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    topk: int,
    out_idx: torch.Tensor,
    warmup: int,
    rep: int,
    *,
    block_sq: int = BLOCK_SQ,
    block_sk: int = BLOCK_SK,
) -> float:
    """Benchmark the Triton kernel with autotuned tile sizes."""
    def op():
        launch_dsalightningindexer(q, k, weight, topk, out_idx, block_sq=block_sq, block_sk=block_sk)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """Run the lightning indexer reference path and (optionally) compare against the Triton kernel."""
    assert(BLOCK_SQ == 1)
    s = BLOCK_SK
    b = 1
    n = 8
    d = 128
    topk = 2048
    warmup_iters = 100
    rep_iters = 50
    device = "cuda:0"
    fp8 = torch.float8_e4m3fn

    torch.manual_seed(0)
    q = torch.randn((s, b, n, d), dtype=torch.float32, device=device).to(fp8)
    k = torch.randn((s, b, n, d), dtype=torch.float32, device=device).to(fp8)
    weight = torch.randn((s, b, n), dtype=torch.float32, device=device).to(fp8)

    torch_idx = torch_lightningindexer(q, k, weight, topk)
    torch_ms = benchmark_torch_lightningindexer(q, k, weight, topk, warmup_iters, rep=rep_iters)
    print(f"PyTorch reference time: {torch_ms * 1000:.3f} us | output shape (idx): {torch_idx.shape}")

    triton_idx = torch.empty_like(torch_idx, dtype=torch.int64)
    triton_ms = benchmark_triton_lightningindexer(
        q, k, weight, topk, triton_idx, warmup=warmup_iters, rep=rep_iters
    )
    launch_dsalightningindexer(q, k, weight, topk, triton_idx)

    torch_sorted, _ = torch.sort(torch_idx, dim=-1)
    triton_sorted, _ = torch.sort(triton_idx, dim=-1)
    torch_flat = torch_sorted.reshape(-1, topk)
    triton_flat = triton_sorted.reshape(-1, topk)
    torch_only_counts = []
    triton_only_counts = []
    for t_row, tri_row in zip(torch_flat, triton_flat):
        torch_only_counts.append((~torch.isin(t_row, tri_row)).sum())
        triton_only_counts.append((~torch.isin(tri_row, t_row)).sum())
    torch_only = torch.stack(torch_only_counts).reshape(q.shape[1], q.shape[0])
    triton_only = torch.stack(triton_only_counts).reshape(q.shape[1], q.shape[0])
    idx_match = bool((torch_only == 0).all().item() and (triton_only == 0).all().item())
    torch_only_total = int(torch_only.sum().item())
    triton_only_total = int(triton_only.sum().item())
    print(
        f"Triton kernel time: {triton_ms * 1000:.3f} us | indices match (set): {idx_match} | "
        f"torch_only total: {torch_only_total} | triton_only total: {triton_only_total} | "
        f"hit rate: {1 - (torch_only_total + triton_only_total) / (b * s * topk)}"
    )


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


if __name__ == "__main__":
    test()
