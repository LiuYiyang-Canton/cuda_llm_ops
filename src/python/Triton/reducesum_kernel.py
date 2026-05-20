# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-30
# Purpose: Triton reduce-sum kernel and launch wrapper.
# ==============================================================================
"""Triton reduce-sum kernel and launch wrapper."""

import functools

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_size": 2048, "num_rows_per_block": 4}, num_warps=8, num_stages=2),
        triton.Config({"block_size": 2048, "num_rows_per_block": 8}, num_warps=8, num_stages=2),
        triton.Config({"block_size": 4096, "num_rows_per_block": 4}, num_warps=8, num_stages=2),
        triton.Config({"block_size": 4096, "num_rows_per_block": 8}, num_warps=8, num_stages=2),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def reducesum_fp32_kernel(
    matrix_ptr,
    rowsum_ptr,
    n_rows,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
    num_rows_per_block: tl.constexpr,
):
    """
    Compute row-wise sums for a 2D FP32 matrix.

    Main feature:
        Sums rows using block-wise reductions in Triton.

    Inputs:
        matrix_ptr: pointer to float32 with shape [n_rows, n_cols]
        rowsum_ptr: pointer to float32 with shape [n_rows]
        n_rows: int32 scalar
        n_cols: constexpr int32 scalar
        block_size: constexpr int32 scalar
        num_rows_per_block: constexpr int32 scalar

    Outputs:
        None
    """
    pid = tl.program_id(0)
    row_ids = pid * num_rows_per_block + tl.arange(0, num_rows_per_block)
    acc = tl.zeros((num_rows_per_block,), dtype=tl.float32)
    cols = tl.arange(0, block_size)
    row_offsets = row_ids[:, None] * n_cols
    for col_start in range(0, n_cols, block_size):
        col_ids = col_start + cols
        mask = (row_ids[:, None] < n_rows) & (col_ids[None, :] < n_cols)
        matrix_vals = tl.load(matrix_ptr + row_offsets + col_ids[None, :], mask=mask, other=0.0)
        acc += tl.sum(matrix_vals, axis=1)
    mask_out = row_ids < n_rows
    tl.store(rowsum_ptr + row_ids, acc, mask=mask_out)


def reducesum_fp32_grid(meta, n_rows):
    """
    Compute the Triton launch grid for reduce-sum.

    Main feature:
        Maps row blocks to a 1D grid.

    Inputs:
        meta: dict with key "num_rows_per_block" (int32)
        n_rows: int32 scalar

    Outputs:
        grid: tuple[int32] with shape [1]
    """
    return (triton.cdiv(n_rows, meta["num_rows_per_block"]),)


def launch_reducesum_fp32_kernel(matrix, rowsum):
    """
    Launch the Triton reduce-sum kernel.

    Main feature:
        Validates inputs and dispatches the kernel.

    Inputs:
        matrix: torch.Tensor float32 of shape [n_rows, n_cols]
        rowsum: torch.Tensor float32 of shape [n_rows]

    Outputs:
        None
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {matrix.shape}")
    if rowsum.ndim != 1 or rowsum.shape[0] != matrix.shape[0]:
        raise ValueError("rowsum must be a 1D tensor with length equal to matrix rows")
    if not (matrix.is_contiguous() and rowsum.is_contiguous()):
        raise ValueError("reducesum_fp32_kernel expects contiguous tensors")

    n_rows, n_cols = matrix.shape
    grid = functools.partial(reducesum_fp32_grid, n_rows=n_rows)
    reducesum_fp32_kernel[grid](matrix, rowsum, n_rows, n_cols)
