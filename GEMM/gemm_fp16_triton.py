"""Triton GEMM with PyTorch comparison and benchmarking utilities."""

import torch
import triton.language as tl
import triton
import triton.testing

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_fp16_kernel(a_ptr, b_ptr, c_ptr, 
                     M: tl.constexpr, 
                     N: tl.constexpr, 
                     K: tl.constexpr, 
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offsets_k = tl.arange(0, BLOCK_K) + k
        mask_a = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        mask_b = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

        a_tile = tl.load(
            a_ptr + offsets_m[:, None] * K + offsets_k[None, :],
            mask=mask_a,
            other=0.0,
        )
        b_tile = tl.load(
            b_ptr + offsets_n[:, None] * K + offsets_k[None, :],
            mask=mask_b,
            other=0.0,
        )

        accumulator += tl.dot(a_tile, tl.trans(b_tile))
    
    mask_c = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(
        c_ptr + offsets_m[:, None] * N + offsets_n[None, :],
        accumulator,
        mask=mask_c,
    )


def launch_triton_gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the Triton fp16->fp32 GEMM kernel."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("launch_triton_gemm expects 2D matrices")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} x {b.shape}")
    if out.shape != (a.shape[0], b.shape[0]):
        raise ValueError(
            f"Output shape {out.shape} does not match {(a.shape[0], b.shape[1])}"
        )
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Input tensors must be float16 for this GEMM example")
    if out.dtype != torch.float32:
        raise ValueError("Output tensor must be float32 for fp32 accumulation")
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("launch_triton_gemm currently expects contiguous tensors")

    M, K = a.shape
    N, _ = b.shape

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    gemm_fp16_kernel[grid](a, b, out, M, N, K)


def benchmark_torch_matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, warmup: int, rep: int
) -> float:
    """Benchmark torch.matmul on fp32 copies of the fp16 inputs."""
    def op():
        torch.matmul(a, b.t(), out=out)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def benchmark_triton_matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, warmup: int, rep: int
) -> float:
    """Benchmark the Triton GEMM kernel (once implemented)."""
    def op():
        launch_triton_gemm(a, b, out)
    return triton.testing.do_bench(op, warmup=warmup, rep=rep)


def test():
    """Compare PyTorch and Triton GEMM paths and report throughput."""
    m = 4096
    n = 4096
    k = 4096
    warmup_iters = 10
    device = "cuda:0"

    torch.manual_seed(0)
    a = torch.rand((m, k), dtype=torch.float16, device=device)
    b = torch.rand((n, k), dtype=torch.float16, device=device)
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
    launch_triton_gemm(a, b, c_triton)

    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    print(f"Triton time: {triton_ms * 1000:.3f} us | throughput: {triton_tflops:.3f} TFLOPS")

    error = relative_error(c_ref, c_triton)
    print(f"gemm_triton relative error: {error:.3e}")


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
