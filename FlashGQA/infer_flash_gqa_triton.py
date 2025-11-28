"""Flash GQA Triton implementation plus launcher, reference path, and benchmarks.

Provides a Triton kernel for bf16 grouped query attention (decoding-focused),
its Python launcher, a PyTorch reference for validation, and benchmarking helpers.
Default shapes target Llama 3.1 70B decoding (batch 60, seqlen_q 1, seqlen_kv 4096,
128 query heads, 8 KV heads with group size 16, head_dim 128).
"""

import math
import torch
import triton
import triton.language as tl
import triton.testing

BATCH_SIZE = 60
SEQLEN_Q = 1
SEQLEN_KV = 4096
NUM_QUERY_HEADS = 128
NUM_KV_HEADS = 8
GROUP_SIZE = NUM_QUERY_HEADS // NUM_KV_HEADS
HEAD_DIM = 128


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_G": 1, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_G": 1, "BLOCK_KV": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_G": 2, "BLOCK_KV": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_G": 2, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_G": 4, "BLOCK_KV": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_G": 4, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
    ],
    key=["q_heads"],
)
@triton.jit
def infer_flash_gqa_bf16_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_heads,
    seqlen_q,
    seqlen_kv,
    scale,
    BLOCK_G: tl.constexpr,  # number of groups of query heads per block
    BLOCK_KV: tl.constexpr,  # kv tokens per flash attention iteration
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Compute grouped query attention on bf16 inputs with fp32 accumulation.

    Args:
        q_ptr (pointer): Q pointer; shape (B, Qh, Sq, D), dtype bf16.
        k_ptr (pointer): K pointer; shape (B, KVh, Skv, D), dtype bf16.
        v_ptr (pointer): V pointer; shape (B, KVh, Skv, D), dtype bf16.
        o_ptr (pointer): Output pointer; shape (B, Qh, Sq, D), dtype bf16.
        q_heads (int): Number of query heads (Qh).
        seqlen_q (int): Query sequence length (Sq).
        seqlen_kv (int): KV sequence length (Skv).
        scale (float): Attention scaling factor (1/sqrt(D)).
        BLOCK_G (constexpr int): Query-head groups processed per block.
        BLOCK_KV (constexpr int): KV tokens processed per iteration.
        HEAD_DIM (constexpr int): Head dimension (D).
        GROUP_SIZE (constexpr int): Query heads per KV head (G).
        NUM_KV_HEADS (constexpr int): Number of KV heads (KVh).

    Returns:
        None. Writes bf16 outputs to o_ptr.
    """
    pid_g = tl.program_id(axis=0)  # group of query heads
    pid_b = tl.program_id(axis=1)  # batch

    for g in range(0, BLOCK_G):
        # Determine query head indices for this group and masks for tail fragments.
        head_offsets = pid_g * BLOCK_G * GROUP_SIZE + g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        head_mask = head_offsets < q_heads
        kv_offsets = tl.arange(0, BLOCK_KV)
        d_offsets = tl.arange(0, HEAD_DIM)

        # Load Q tile for this head group (seqlen_q expected to be small, often 1).
        q_base = (pid_b * q_heads + head_offsets) * seqlen_q * HEAD_DIM
        q_ptrs = q_ptr + q_base[:, None] + d_offsets[None, :]
        q = tl.load(q_ptrs, mask=head_mask[:, None], other=0.0) # shape: (GROUP_SIZE, HEAD_DIM)

        # Initialize streaming softmax statistics and accumulation buffers.
        rowmax = tl.full((GROUP_SIZE,), -float("inf"), dtype=tl.float32) # shape: (GROUP_SIZE,)
        rowsum = tl.zeros((GROUP_SIZE,), dtype=tl.float32) # shape: (GROUP_SIZE,)
        acc = tl.zeros((GROUP_SIZE, HEAD_DIM), dtype=tl.float32) # shape: (GROUP_SIZE, HEAD_DIM)

        # Identify KV head for this group and iterate over KV tiles.
        kv_head = pid_g * BLOCK_G + g  # each group of query heads maps to one kv head
        k_base = (pid_b * NUM_KV_HEADS + kv_head) * seqlen_kv * HEAD_DIM
        for start_kv in range(0, seqlen_kv, BLOCK_KV):
            # Compute offsets for current KV tile and load K/V.
            kv_idx = start_kv + kv_offsets
            kv_mask = kv_idx < seqlen_kv

            kv_offsets_2d = kv_idx * HEAD_DIM  # shape: (BLOCK_KV,)
            k_ptrs = k_ptr + (k_base + kv_offsets_2d[:, None] + d_offsets[None, :])
            v_ptrs = v_ptr + (k_base + kv_offsets_2d[:, None] + d_offsets[None, :])

            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0) # shape: (BLOCK_KV, HEAD_DIM)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0) # shape: (BLOCK_KV, HEAD_DIM)

            # Compute scaled QK, mask padding, and update streaming softmax.
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale  # (GROUP_SIZE, BLOCK_KV)
            qk = tl.where(head_mask[:, None] & kv_mask[None, :], qk, -float("inf"))

            rowmax_new = tl.maximum(rowmax, tl.max(qk, axis=1))
            p = tl.exp(qk - rowmax_new[:, None]) # shape: (GROUP_SIZE, BLOCK_KV)
            alpha = tl.exp(rowmax - rowmax_new)

            # Accumulate weighted values and denominator in fp32.
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            rowsum = rowsum * alpha + tl.sum(p, axis=1)
            rowmax = rowmax_new

        # Normalize accumulated values and store bf16 outputs.
        rowsum_safe = tl.where(rowsum > 0, rowsum, 1.0)
        o = acc / rowsum_safe[:, None]

        o_base = (pid_b * q_heads + head_offsets) * seqlen_q * HEAD_DIM
        o_ptrs = o_ptr + o_base[:, None] + d_offsets[None, :]
        tl.store(o_ptrs, o.to(tl.bfloat16), mask=head_mask[:, None])


def launch_flash_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Launch the Triton Flash GQA kernel on bf16 tensors.

    Args:
        q (Tensor): (B, Qh, Sq, D) bf16 query tensor.
        k (Tensor): (B, KVh, Skv, D) bf16 key tensor.
        v (Tensor): (B, KVh, Skv, D) bf16 value tensor.
        out (Tensor, optional): Preallocated output (B, Qh, Sq, D) bf16 tensor.

    Returns:
        Tensor: Output attention result (B, Qh, Sq, D) bf16.
    """
    # Validate shapes/dtypes and ensure contiguity.
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("Flash GQA expects bf16 inputs")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Flash GQA kernel currently expects contiguous Q/K/V")

    batch, q_heads, seqlen_q, head_dim = q.shape
    k_batch, kv_heads, seqlen_kv, k_head_dim = k.shape
    v_batch, v_heads, v_seqlen, v_head_dim = v.shape

    if batch != k_batch or batch != v_batch:
        raise ValueError("Q, K, and V must share batch dimension")
    if (seqlen_kv != v_seqlen) or (k_head_dim != v_head_dim):
        raise ValueError("K and V must share sequence/head_dim sizes")
    if q_heads != kv_heads * GROUP_SIZE or v_heads != kv_heads:
        raise ValueError(f"Query heads must equal KV heads * GROUP_SIZE ({GROUP_SIZE}); got q={q_heads}, kv={kv_heads}")
    if head_dim != HEAD_DIM:
        raise ValueError(f"Expected head_dim {HEAD_DIM}, got {head_dim}")

    if out is None:
        out = torch.empty_like(q)
    if not out.is_contiguous():
        raise ValueError("Output tensor must be contiguous")

    # Compute scale and grid; grid depends on autotuned meta-parameters.
    scale = 1.0 / math.sqrt(head_dim)

    def grid(meta):
        return (
            triton.cdiv(NUM_KV_HEADS, meta["BLOCK_G"]),  # query head tiles
            batch,  # batch tiles
        )

    infer_flash_gqa_bf16_kernel[grid](
        q,
        k,
        v,
        out,
        q_heads,
        seqlen_q,
        seqlen_kv,
        scale,
        HEAD_DIM=HEAD_DIM,
        GROUP_SIZE=GROUP_SIZE,
        NUM_KV_HEADS=NUM_KV_HEADS,
    )
    return out


def flash_gqa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Reference grouped attention used for validation/benchmarks.

    Args:
        q (Tensor): (B, Qh, Sq, D) tensor.
        k (Tensor): (B, KVh, Skv, D) tensor.
        v (Tensor): (B, KVh, Skv, D) tensor.
        scale (float, optional): Scaling factor; defaults to 1/sqrt(D).

    Returns:
        Tensor: (B, Qh, Sq, D) attention output, dtype matches q.
    """
    # Operate in fp32 for stability and iterate over KV tiles to bound memory.
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    batch, _, seqlen_q, _ = q.shape
    _, kv_heads, seqlen_kv, _ = k.shape
    TILE_N = 128  # key/value sequence tile

    outputs = []
    for b in range(batch):
        # Slice per-batch inputs and upcast to fp32.
        q_b = q[b].to(torch.float32)  # (Qh, S_q, D)
        k_b = k[b].to(torch.float32)  # (KVh, S_kv, D)
        v_b = v[b].to(torch.float32)  # (KVh, S_kv, D)
        out_heads = []
        for kv_idx in range(kv_heads):
            # Process one KV head-group at a time with streaming softmax over tiles.
            q_slice = q_b[kv_idx * GROUP_SIZE : (kv_idx + 1) * GROUP_SIZE]  # (G, S_q, D)
            m_i = torch.full((GROUP_SIZE, seqlen_q), -float("inf"), device=q_slice.device, dtype=torch.float32)
            l_i = torch.zeros((GROUP_SIZE, seqlen_q), device=q_slice.device, dtype=torch.float32)
            acc = torch.zeros((GROUP_SIZE, seqlen_q, HEAD_DIM), device=q_slice.device, dtype=torch.float32)
            for start in range(0, seqlen_kv, TILE_N):
                end = min(start + TILE_N, seqlen_kv)
                k_tile = k_b[kv_idx, start:end]  # (T, D)
                v_tile = v_b[kv_idx, start:end]  # (T, D)
                qk = torch.matmul(q_slice, k_tile.transpose(-1, -2)) * scale  # (G, S_q, T)

                m_i_new = torch.maximum(m_i, qk.max(dim=-1).values)  # (G, S_q)
                p = torch.exp(qk - m_i_new.unsqueeze(-1))
                alpha = torch.exp(m_i - m_i_new)

                acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_tile)  # (G, S_q, D)
                l_i = l_i * alpha + p.sum(dim=-1)
                m_i = m_i_new

            out_slice = acc / l_i.unsqueeze(-1)
            out_heads.append(out_slice.to(q.dtype))
        out_b = torch.cat(out_heads, dim=0).to(q.dtype)
        outputs.append(out_b)
    return torch.stack(outputs, dim=0)


def benchmark_torch_flash_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark the PyTorch reference grouped attention (returns ms).

    Args:
        q, k, v (Tensor): Same shapes/dtypes as Flash GQA inputs (bf16).
        out (Tensor): Output buffer copied from reference result.
        warmup (int): Warmup iterations.
        rep (int): Timed iterations.
    """
    def op():
        out.copy_(flash_gqa_reference(q, k, v))
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_flash_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark the Triton Flash GQA kernel (returns ms).

    Args mirror benchmark_torch_flash_gqa (bf16 inputs/outputs with matching shapes).
    """
    def op():
        launch_flash_gqa(q, k, v, out)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def relative_error(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Return ||ref - approx|| / ||ref||.

    Args:
        ref (Tensor): Reference tensor (shape/dtype must match approx).
        approx (Tensor): Tensor under test.
    """
    if ref.shape != approx.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {approx.shape}")
    if ref.dtype != approx.dtype:
        raise ValueError(f"Data type mismatch: {ref.dtype} vs {approx.dtype}")
    denom = torch.linalg.norm(ref)
    if denom == 0:
        return float(torch.linalg.norm(ref - approx))
    return float(torch.linalg.norm(ref - approx) / denom)


def _bytes_accessed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor) -> int:
    """Estimate bytes moved: read Q/K/V once, write O once."""
    return (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()


def test(run_kernel: bool = True, warmup: int = 100, rep: int = 100):
    """Harness to validate and report perf/bandwidth (single batch).

    Args:
        run_kernel (bool): Whether to benchmark Triton kernel in addition to reference.
        warmup (int): Warmup iterations for benchmarking.
        rep (int): Timed iterations for benchmarking.
    """
    # Set seed and pick device; clear CUDA cache for consistent measurements.
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q = torch.rand((BATCH_SIZE, NUM_QUERY_HEADS, SEQLEN_Q, HEAD_DIM), device=device, dtype=torch.bfloat16)
    k = torch.rand((BATCH_SIZE, NUM_KV_HEADS, SEQLEN_KV, HEAD_DIM), device=device, dtype=torch.bfloat16)
    v = torch.rand((BATCH_SIZE, NUM_KV_HEADS, SEQLEN_KV, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    # Reference computation for correctness checking.
    ref = flash_gqa_reference(q, k, v)
    if not run_kernel:
        print("Flash GQA Triton kernel ready; set run_kernel=True to run benchmarks.")
        return

    # Warmup + timed runs
    torch_ms = benchmark_torch_flash_gqa(q, k, v, out, warmup=warmup, rep=rep)

    triton_ms = None
    if device.type == "cuda":
        # Benchmark Triton kernel and run once more to materialize output for error calc.
        triton_ms = benchmark_triton_flash_gqa(q, k, v, out, warmup=warmup, rep=rep)
        _ = launch_flash_gqa(q, k, v, out)
    else:
        print("Triton benchmark skipped (CPU fallback).")

    err = relative_error(ref, out)

    bytes_moved = _bytes_accessed(q, k, v, out)
    print(f"infer_flash_gqa_bf16_kernel relative error: {err:.3e}")
    torch_bw = bytes_moved / (torch_ms * 1e-3) / (1024 ** 4)
    print(f"PyTorch reference: {torch_ms:.5f} ms | approx. bandwidth: {torch_bw:.6f} TB/s")
    if triton_ms is not None:
        triton_bw = bytes_moved / (triton_ms * 1e-3) / (1024 ** 4)
        print(f"Triton Flash GQA: {triton_ms:.5f} ms | approx. bandwidth: {triton_bw:.6f} TB/s")
        best_cfg = getattr(infer_flash_gqa_bf16_kernel, "best_config", None)
        if best_cfg is not None:
            print(f"Triton autotune best config: {best_cfg}")


if __name__ == "__main__":
    test()
