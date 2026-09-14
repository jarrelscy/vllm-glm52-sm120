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
def kernel(R, W, output, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i // H
    h = i % H
    base = t * 8 * H + h
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(2):
        v0 = mul_rn(
            tl.load(R + base + (j * 4) * H, i < N, other=0),
            tl.load(W + t * 8 + j * 4, i < N, other=0).to(tl.float32),
        )
        v1 = mul_rn(
            tl.load(R + base + (j * 4 + 1) * H, i < N, other=0),
            tl.load(W + t * 8 + j * 4 + 1, i < N, other=0).to(tl.float32),
        )
        v2 = mul_rn(
            tl.load(R + base + (j * 4 + 2) * H, i < N, other=0),
            tl.load(W + t * 8 + j * 4 + 2, i < N, other=0).to(tl.float32),
        )
        v3 = mul_rn(
            tl.load(R + base + (j * 4 + 3) * H, i < N, other=0),
            tl.load(W + t * 8 + j * 4 + 3, i < N, other=0).to(tl.float32),
        )
        a0 = a0 + v0
        a1 = a1 + v1
        a2 = a2 + v2
        a3 = a3 + v3
    result = ((a0 + a1) + a2) + a3
    tl.store(output + i, result, i < N)


def run(routed, weights, dtype=torch.bfloat16):
    t, topk = weights.shape
    assert topk == 8 and routed.dtype == torch.float32
    assert routed.is_contiguous() and weights.is_contiguous()
    h = routed.numel() // (t * 8)
    out = torch.empty((t, h), device=routed.device, dtype=dtype)
    kernel[(triton.cdiv(t * h, 256),)](
        routed, weights, out, t * h, h, 256, enable_fp_fusion=False
    )
    return out
