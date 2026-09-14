# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP32 top8 route weighting/reduction, matching Torch's four accumulators."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def kernel(R, W, INV, OUT, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i // H
    h = i % H
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(2):
        v0 = mul_rn(
            tl.load(
                R + tl.load(INV + t * 8 + (j * 4), i < N, other=0) * H + h,
                i < N,
                other=0,
            ),
            tl.load(W + t * 8 + j * 4, i < N, other=0).to(tl.float32),
        )
        v1 = mul_rn(
            tl.load(
                R + tl.load(INV + t * 8 + (j * 4 + 1), i < N, other=0) * H + h,
                i < N,
                other=0,
            ),
            tl.load(W + t * 8 + j * 4 + 1, i < N, other=0).to(tl.float32),
        )
        v2 = mul_rn(
            tl.load(
                R + tl.load(INV + t * 8 + (j * 4 + 2), i < N, other=0) * H + h,
                i < N,
                other=0,
            ),
            tl.load(W + t * 8 + j * 4 + 2, i < N, other=0).to(tl.float32),
        )
        v3 = mul_rn(
            tl.load(
                R + tl.load(INV + t * 8 + (j * 4 + 3), i < N, other=0) * H + h,
                i < N,
                other=0,
            ),
            tl.load(W + t * 8 + j * 4 + 3, i < N, other=0).to(tl.float32),
        )
        a0 = a0 + v0
        a1 = a1 + v1
        a2 = a2 + v2
        a3 = a3 + v3
    result = ((a0 + a1) + a2) + a3
    tl.store(OUT + i, result, i < N)


def sum_routes(routed, weights, inverse, dtype=torch.bfloat16):
    t, topk = weights.shape
    assert topk == 8 and routed.dtype == torch.float32
    assert routed.is_contiguous() and weights.is_contiguous()
    h = routed.numel() // (t * 8)
    out = torch.empty((t, h), device=routed.device, dtype=dtype)
    kernel[(triton.cdiv(t * h, 256),)](
        routed, weights, inverse, out, t * h, h, 256, enable_fp_fusion=False
    )
    return out


@triton.jit(do_not_specialize=["C"])
def inverse_kernel(INV, COLD, SORTED, S: tl.constexpr, C, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Cold routes occupy their expert-major sorted positions.
    dst = tl.load(SORTED + i, i < C, other=0)
    tl.store(INV + dst, i, i < C)
    # Cold slots are ascending from nonzero. Binary search counts cold routes
    # strictly before each original slot. Noncold destinations remain stable.
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), C, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        v = tl.load(COLD + mid, (lo < hi) & (mid < C), other=S)
        take = (lo < hi) & (v < i)
        hi = tl.where((lo < hi) & ~take, mid, hi)
        lo = tl.where(take, mid + 1, lo)
    v = tl.load(COLD + lo, lo < C, other=S)
    tl.store(INV + i, C + i - lo, (i < S) & (v != i))


def make_inverse(inverse, cold_slots, sorted_slots, slots):
    inverse_kernel[(triton.cdiv(slots, 256),)](
        inverse, cold_slots, sorted_slots, slots, cold_slots.numel(), 256
    )


@triton.jit(do_not_specialize=["N"])
def scatter_kernel(R, SLOTS, VALUES, INV, N, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = tl.load(SLOTS + i // H, i < N, other=0)
    destination = tl.load(INV + original, i < N, other=0)
    value = tl.load(VALUES + i, i < N, other=0)
    tl.store(R + destination * H + i % H, value, i < N)


def scatter(routed, route_slots, values, inverse):
    if inverse is None:
        routed[route_slots] = values
        return
    assert values.is_contiguous() and values.dtype == routed.dtype
    scatter_kernel[(triton.cdiv(values.numel(), 256),)](
        routed, route_slots, values, inverse, values.numel(), routed.shape[1], 256
    )
