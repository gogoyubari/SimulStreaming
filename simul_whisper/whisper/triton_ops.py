from functools import lru_cache

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; try `pip install --pre triton`")


@triton.jit
def dtw_kernel(
    cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    for k in range(1, N + M + 1):  # k = i + j
        tl.debug_barrier()

        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1

        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)

        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)

        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)

        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))

@triton.jit
def median_kernel_3(y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_stride
    x_ptr = x + row_idx * x_stride
    y_ptr = y + row_idx * y_stride

    row0 = tl.load(x_ptr + offsets + 0, mask=mask)
    row1 = tl.load(x_ptr + offsets + 1, mask=mask)
    row2 = tl.load(x_ptr + offsets + 2, mask=mask)

    smaller = tl.where(row0 < row1, row0, row1)
    larger = tl.where(row0 > row1, row0, row1)
    row0 = smaller
    row1 = larger

    smaller = tl.where(row1 < row2, row1, row2)
    larger = tl.where(row1 > row2, row1, row2)
    row1 = smaller
    row2 = larger

    smaller = tl.where(row0 < row1, row0, row1)
    larger = tl.where(row0 > row1, row0, row1)
    row0 = smaller
    row1 = larger

    tl.store(y_ptr + offsets, row1, mask=mask)


@triton.jit
def median_kernel_5(y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_stride
    x_ptr = x + row_idx * x_stride
    y_ptr = y + row_idx * y_stride

    row0 = tl.load(x_ptr + offsets + 0, mask=mask)
    row1 = tl.load(x_ptr + offsets + 1, mask=mask)
    row2 = tl.load(x_ptr + offsets + 2, mask=mask)
    row3 = tl.load(x_ptr + offsets + 3, mask=mask)
    row4 = tl.load(x_ptr + offsets + 4, mask=mask)

    # Bubble sort for 5 elements
    for _ in range(3):  # (5 // 2) + 1 = 3 passes
        smaller = tl.where(row0 < row1, row0, row1)
        larger = tl.where(row0 > row1, row0, row1)
        row0 = smaller
        row1 = larger

        smaller = tl.where(row1 < row2, row1, row2)
        larger = tl.where(row1 > row2, row1, row2)
        row1 = smaller
        row2 = larger

        smaller = tl.where(row2 < row3, row2, row3)
        larger = tl.where(row2 > row3, row2, row3)
        row2 = smaller
        row3 = larger

        smaller = tl.where(row3 < row4, row3, row4)
        larger = tl.where(row3 > row4, row3, row4)
        row3 = smaller
        row4 = larger

    tl.store(y_ptr + offsets, row2, mask=mask)


@triton.jit
def median_kernel_7(y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_stride
    x_ptr = x + row_idx * x_stride
    y_ptr = y + row_idx * y_stride

    row0 = tl.load(x_ptr + offsets + 0, mask=mask)
    row1 = tl.load(x_ptr + offsets + 1, mask=mask)
    row2 = tl.load(x_ptr + offsets + 2, mask=mask)
    row3 = tl.load(x_ptr + offsets + 3, mask=mask)
    row4 = tl.load(x_ptr + offsets + 4, mask=mask)
    row5 = tl.load(x_ptr + offsets + 5, mask=mask)
    row6 = tl.load(x_ptr + offsets + 6, mask=mask)

    # Bubble sort for 7 elements
    for _ in range(4):  # (7 // 2) + 1 = 4 passes
        smaller = tl.where(row0 < row1, row0, row1)
        larger = tl.where(row0 > row1, row0, row1)
        row0 = smaller
        row1 = larger

        smaller = tl.where(row1 < row2, row1, row2)
        larger = tl.where(row1 > row2, row1, row2)
        row1 = smaller
        row2 = larger

        smaller = tl.where(row2 < row3, row2, row3)
        larger = tl.where(row2 > row3, row2, row3)
        row2 = smaller
        row3 = larger

        smaller = tl.where(row3 < row4, row3, row4)
        larger = tl.where(row3 > row4, row3, row4)
        row3 = smaller
        row4 = larger

        smaller = tl.where(row4 < row5, row4, row5)
        larger = tl.where(row4 > row5, row4, row5)
        row4 = smaller
        row5 = larger

        smaller = tl.where(row5 < row6, row5, row6)
        larger = tl.where(row5 > row6, row5, row6)
        row5 = smaller
        row6 = larger

    tl.store(y_ptr + offsets, row3, mask=mask)


@lru_cache(maxsize=None)
def median_kernel(filter_width: int):
    if filter_width == 3:
        return median_kernel_3
    elif filter_width == 5:
        return median_kernel_5
    elif filter_width == 7:
        return median_kernel_7
    else:
        raise NotImplementedError(
            f"median_kernel for filter_width={filter_width} is not implemented. "
            f"Only 3, 5, and 7 are currently supported."
        )


def median_filter_cuda(x: torch.Tensor, filter_width: int):
    """Apply a median filter of given width along the last dimension of x"""
    slices = x.contiguous().unfold(-1, filter_width, 1)
    grid = np.prod(slices.shape[:-2])

    kernel = median_kernel(filter_width)
    y = torch.empty_like(slices[..., 0])

    BLOCK_SIZE = 1 << (y.stride(-2) - 1).bit_length()
    kernel[(grid,)](y, x, x.stride(-2), y.stride(-2), BLOCK_SIZE=BLOCK_SIZE)

    return y
