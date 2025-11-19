"""Triton reduce-sum utilities with PyTorch comparison and benchmarking scaffolding."""

import torch
import triton
import triton.language as tl
import triton.testing


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048, "NUM_ROWS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "NUM_ROWS_PER_BLOCK": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "NUM_ROWS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "NUM_ROWS_PER_BLOCK": 8}, num_warps=8, num_stages=2),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def reducesum_fp32_kernel(matrix_ptr, rowsum_ptr, n_rows, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_ROWS_PER_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_ids = pid * NUM_ROWS_PER_BLOCK + tl.arange(0, NUM_ROWS_PER_BLOCK)
    acc = tl.zeros((NUM_ROWS_PER_BLOCK,), dtype=tl.float32)
    cols = tl.arange(0, BLOCK_SIZE)
    row_offsets = row_ids[:, None] * n_cols
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_ids = col_start + cols
        mask = (row_ids[:, None] < n_rows) & (col_ids[None, :] < n_cols)
        matrix_vals = tl.load(matrix_ptr + row_offsets + col_ids[None, :], mask=mask, other=0.0)
        acc += tl.sum(matrix_vals, axis=1)
    mask_out = row_ids < n_rows
    tl.store(rowsum_ptr + row_ids, acc, mask=mask_out)


def launch_reducesum(matrix: torch.Tensor, rowsum: torch.Tensor) -> None:
    """Launch the Triton reduce-sum kernel assuming contiguous tensors."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {matrix.shape}")
    if rowsum.ndim != 1 or rowsum.shape[0] != matrix.shape[0]:
        raise ValueError("rowsum must be a 1D tensor with length equal to matrix rows")
    if not (matrix.is_contiguous() and rowsum.is_contiguous()):
        raise ValueError("reducesum_fp32_kernel expects contiguous tensors")

    n_rows, n_cols = matrix.shape

    def grid(meta):
        return (triton.cdiv(n_rows, meta["NUM_ROWS_PER_BLOCK"]),)

    reducesum_fp32_kernel[grid](matrix, rowsum, n_rows, n_cols)


def benchmark_torch_reducesum(matrix: torch.Tensor, rowsum: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark torch.sum over rows using Triton's timing helper."""

    def op():
        torch.sum(matrix, dim=1, out=rowsum)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_reducesum(matrix: torch.Tensor, rowsum: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark the Triton reduce-sum kernel with autotuned tile sizes."""

    def op():
        launch_reducesum(matrix, rowsum)

    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """Compare PyTorch and Triton reduce-sum implementations and report bandwidth."""
    m = 4096
    n = 4096
    warmup_iters = 10000
    device = "cuda:0"

    torch.manual_seed(0)
    matrix = torch.rand((m, n), dtype=torch.float32, device=device)
    rowsum_ref = torch.empty((m,), dtype=torch.float32, device=device)
    rowsum_triton = torch.empty_like(rowsum_ref)

    torch_ms = benchmark_torch_reducesum(matrix, rowsum_ref, warmup_iters, rep=100)
    triton_ms = benchmark_triton_reducesum(matrix, rowsum_triton, warmup_iters, rep=100)
    launch_reducesum(matrix, rowsum_triton)

    bytes_transferred = matrix.element_size() * matrix.numel() + rowsum_ref.element_size() * rowsum_ref.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024**4)
    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024**4)
    print(f"PyTorch time: {torch_ms * 1000:.3f} us | bandwidth: {torch_bw:.3f} TB/s")
    print(f"Triton time: {triton_ms * 1000:.3f} us | bandwidth: {triton_bw:.3f} TB/s")

    error = relative_error(rowsum_ref, rowsum_triton)
    print(f"reducesum_fp32_kernel relative error: {error:.3e}")


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
