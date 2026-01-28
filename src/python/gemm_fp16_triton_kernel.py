# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Triton FP16 GEMM kernel and launch wrapper.
# ==============================================================================
"""Triton FP16 GEMM kernel and launch wrapper."""

import functools

import torch
import triton
import triton.language as tl
from triton.runtime import _allocation


def ensure_tma_allocator(device):
    """
    Ensure Triton has an allocator for TMA descriptor workspace.

    Main feature:
        Installs a torch-backed allocator when Triton uses a null allocator.

    Inputs:
        device: torch.device scalar

    Outputs:
        ok: bool scalar
    """
    try:
        current = _allocation._allocator.get()
    except Exception:
        return False

    if isinstance(current, _allocation.NullAllocator):
        def torch_allocator(size, alignment, stream):
            """
            Allocate a device buffer for Triton TMA descriptors.

            Main feature:
                Returns a byte-addressable buffer on the target device.

            Inputs:
                size: int64 scalar
                alignment: int64 scalar
                stream: torch.cuda.Stream or None

            Outputs:
                buffer: torch.Tensor int8 of shape [size]
            """
            return torch.empty(size, device=device, dtype=torch.int8)

        triton.set_allocator(torch_allocator)
    return True


def should_use_tma(use_tma, device):
    """
    Decide whether to enable TMA for GEMM.

    Main feature:
        Gates TMA on user choice, CUDA capability, and allocator availability.

    Inputs:
        use_tma: bool scalar
        device: torch.device scalar

    Outputs:
        enable_tma: bool scalar
    """
    if not use_tma:
        return False
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    if major < 9:
        return False
    return ensure_tma_allocator(device)


@triton.autotune(
    configs=[
        triton.Config(
            {"block_m": 128, "block_n": 128, "block_k": 64},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def gemm_fp16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    use_tma: tl.constexpr = True,
):
    """
    Compute C = A @ B^T for FP16 inputs with FP32 accumulation.

    Main feature:
        Uses TMA when tiles are fully in-bounds, otherwise masked loads/stores.

    Inputs:
        a_ptr: pointer to float16 with shape [m, k]
        b_ptr: pointer to float16 with shape [n, k]
        c_ptr: pointer to float32 with shape [m, n]
        m: int32 scalar
        n: int32 scalar
        k: int32 scalar
        block_m: constexpr int32 scalar
        block_n: constexpr int32 scalar
        block_k: constexpr int32 scalar
        use_tma: constexpr bool scalar

    Outputs:
        None
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    tma_path = use_tma & ((pid_m + 1) * block_m <= m) & ((pid_n + 1) * block_n <= n)

    if tma_path:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[m, k],
            strides=[k, 1],
            block_shape=[block_m, block_k],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[n, k],
            strides=[k, 1],
            block_shape=[block_n, block_k],
        )
        for k_start in range(0, (k // block_k) * block_k, block_k):
            a_tile = a_desc.load([pid_m * block_m, k_start])
            b_tile = b_desc.load([pid_n * block_n, k_start])
            accumulator += tl.dot(a_tile, tl.trans(b_tile))
        for k_start in range((k // block_k) * block_k, k, block_k):
            offsets_k = tl.arange(0, block_k) + k_start
            mask_a = (offsets_m[:, None] < m) & (offsets_k[None, :] < k)
            mask_b = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)

            a_tile = tl.load(
                a_ptr + offsets_m[:, None] * k + offsets_k[None, :],
                mask=mask_a,
                other=0.0,
            )
            b_tile = tl.load(
                b_ptr + offsets_n[:, None] * k + offsets_k[None, :],
                mask=mask_b,
                other=0.0,
            )
            accumulator += tl.dot(a_tile, tl.trans(b_tile))
    else:
        for k_start in range(0, k, block_k):
            offsets_k = tl.arange(0, block_k) + k_start
            mask_a = (offsets_m[:, None] < m) & (offsets_k[None, :] < k)
            mask_b = (offsets_n[:, None] < n) & (offsets_k[None, :] < k)

            a_tile = tl.load(
                a_ptr + offsets_m[:, None] * k + offsets_k[None, :],
                mask=mask_a,
                other=0.0,
            )
            b_tile = tl.load(
                b_ptr + offsets_n[:, None] * k + offsets_k[None, :],
                mask=mask_b,
                other=0.0,
            )

            accumulator += tl.dot(a_tile, tl.trans(b_tile))

    if tma_path:
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[m, n],
            strides=[n, 1],
            block_shape=[block_m, block_n],
        )
        c_desc.store([pid_m * block_m, pid_n * block_n], accumulator)
    else:
        mask_c = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
        tl.store(
            c_ptr + offsets_m[:, None] * n + offsets_n[None, :],
            accumulator,
            mask=mask_c,
        )


def gemm_fp16_grid(meta, m, n):
    """
    Compute the Triton launch grid for GEMM.

    Main feature:
        Maps matrix tiles to a 2D grid.

    Inputs:
        meta: dict with keys "block_m" and "block_n" (int32)
        m: int32 scalar
        n: int32 scalar

    Outputs:
        grid: tuple[int32, int32] with shape [2]
    """
    return (triton.cdiv(m, meta["block_m"]), triton.cdiv(n, meta["block_n"]))


def launch_gemm_fp16_kernel(a, b, out, use_tma=True):
    """
    Launch the Triton FP16 GEMM kernel.

    Main feature:
        Validates inputs, selects TMA usage, and dispatches the kernel.

    Inputs:
        a: torch.Tensor float16 of shape [m, k]
        b: torch.Tensor float16 of shape [n, k]
        out: torch.Tensor float32 of shape [m, n]
        use_tma: bool scalar

    Outputs:
        None
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("launch_gemm_fp16_kernel expects 2D matrices")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} x {b.shape}")
    if out.shape != (a.shape[0], b.shape[0]):
        raise ValueError(
            f"Output shape {out.shape} does not match {(a.shape[0], b.shape[0])}"
        )
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Input tensors must be float16 for this GEMM")
    if out.dtype != torch.float32:
        raise ValueError("Output tensor must be float32 for fp32 accumulation")
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("launch_gemm_fp16_kernel expects contiguous tensors")
    if not (a.device == b.device == out.device):
        raise ValueError("Input and output tensors must be on the same device")

    m, k = a.shape
    n, _ = b.shape
    use_tma_flag = should_use_tma(use_tma, a.device)

    grid = functools.partial(gemm_fp16_grid, m=m, n=n)
    gemm_fp16_kernel[grid](a, b, out, m, n, k, use_tma=use_tma_flag)
