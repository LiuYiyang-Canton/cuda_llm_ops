// ==============================================================================
// Author: Liu Yiyang
// Date:   2026-01-30
// Purpose: Common utilities for CUDA LLM ops.
// ==============================================================================
#pragma once

/**
 * @brief Computes the ceiling of integer division.
 * @param a Numerator value.
 * @param b Denominator value (must be > 0).
 * @return Smallest integer >= a / b.
 */
__host__ __device__ constexpr int CEIL_DIV(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * @brief Flattens a 3D index into a 1D index.
 * @param i1 First index.
 * @param i2 Second index.
 * @param i3 Third index.
 * @param D2 Size of the second dimension.
 * @param D3 Size of the third dimension.
 * @return Flattened 1D index.
 */
__host__ __device__ constexpr int GET_1D_INDEX_FROM_3D(int i1, int i2, int i3, int D2, int D3) {
    return i1 * D2 * D3 + i2 * D3 + i3;
}

/**
 * @brief Flattens a 4D index into a 1D index.
 * @param i1 First index.
 * @param i2 Second index.
 * @param i3 Third index.
 * @param i4 Fourth index.
 * @param D2 Size of the second dimension.
 * @param D3 Size of the third dimension.
 * @param D4 Size of the fourth dimension.
 * @return Flattened 1D index.
 */
__host__ __device__ constexpr int GET_1D_INDEX_FROM_4D(int i1, int i2, int i3, int i4, int D2, int D3, int D4) {
    return i1 * D2 * D3 * D4 + i2 * D3 * D4 + i3 * D4 + i4;
}
