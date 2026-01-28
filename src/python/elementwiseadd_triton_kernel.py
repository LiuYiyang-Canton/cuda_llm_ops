# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Triton elementwise add kernel and launch wrapper.
# ==============================================================================
"""Triton elementwise add kernel and launch wrapper."""

import functools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_size": 1024}, num_warps=4),
        triton.Config({"block_size": 2048}, num_warps=4),
        triton.Config({"block_size": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def elementwiseadd_fp32_kernel(a_ptr, b_ptr, c_ptr, n_elements, block_size: tl.constexpr):
    """
    Compute elementwise c = a + b for a flat contiguous buffer.

    Main feature:
        Loads two float32 buffers and stores the elementwise sum.

    Inputs:
        a_ptr: pointer to float32 with shape [n_elements]
        b_ptr: pointer to float32 with shape [n_elements]
        c_ptr: pointer to float32 with shape [n_elements]
        n_elements: int32 scalar, total number of elements
        block_size: constexpr int32 scalar, block size per program

    Outputs:
        None
    """
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a_vals + b_vals, mask=mask)


def elementwiseadd_fp32_grid(meta, n_elements):
    """
    Compute the Triton grid for elementwise add.

    Main feature:
        Uses block size to compute the 1D launch grid.

    Inputs:
        meta: dict with key "block_size" (int32)
        n_elements: int32 scalar, total number of elements

    Outputs:
        grid: tuple[int32] with shape [1]
    """
    return (triton.cdiv(n_elements, meta["block_size"]),)


def launch_elementwiseadd_fp32_kernel(a, b, out):
    """
    Launch the Triton elementwise add kernel.

    Main feature:
        Validates inputs and dispatches the autotuned Triton kernel.

    Inputs:
        a: torch.Tensor float32 of shape [n_elements]
        b: torch.Tensor float32 of shape [n_elements]
        out: torch.Tensor float32 of shape [n_elements]

    Outputs:
        None
    """
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("elementwiseadd_fp32_kernel expects contiguous tensors")
    if a.shape != b.shape or a.shape != out.shape:
        raise ValueError("Input/output tensors must share the same shape")

    n_elements = a.numel()
    grid = functools.partial(elementwiseadd_fp32_grid, n_elements=n_elements)
    elementwiseadd_fp32_kernel[grid](a, b, out, n_elements)
