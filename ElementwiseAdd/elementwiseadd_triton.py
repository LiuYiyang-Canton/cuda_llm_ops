"""Triton elementwise add kernel with PyTorch comparison and benchmarking utilities."""

import torch
import triton
import triton.language as tl
import triton.testing


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def elementwiseadd_fp32_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise C = A + B across a flat contiguous buffer."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a_vals + b_vals, mask=mask)


def launch_elementwiseadd(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the Triton kernel assuming contiguous tensors."""
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("elementwise_add_fp32_kernel expects contiguous tensors")
    if a.shape != b.shape or a.shape != out.shape:
        raise ValueError("Input/output tensors must share the same shape")

    n_elements = a.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    elementwiseadd_fp32_kernel[grid](a, b, out, n_elements)


def benchmark_torch_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark torch.add using Triton's timing helper."""
    def op():
        torch.add(a, b, out=out)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, warmup: int, rep: int) -> float:
    """Benchmark the Triton kernel with autotuned tile sizes."""
    def op():
        launch_elementwiseadd(a, b, out)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """Compare PyTorch and Triton elementwise adds and report bandwidth."""
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
    launch_elementwiseadd(a, b, c_triton)

    bytes_transferred = 3 * a.element_size() * a.numel()
    torch_bw = bytes_transferred / (torch_ms * 1e-3) / (1024 ** 4)
    triton_bw = bytes_transferred / (triton_ms * 1e-3) / (1024 ** 4)
    print(f"PyTorch time: {torch_ms * 1000:.3f} us | bandwidth: {torch_bw:.3f} TB/s")
    print(f"Triton time: {triton_ms * 1000:.3f} us | bandwidth: {triton_bw:.3f} TB/s")

    error = relative_error(c_ref, c_triton)
    print(f"elementwiseadd_fp32_kernel relative error: {error:.3e}")


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
