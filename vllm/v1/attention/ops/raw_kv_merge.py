# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact active LSE sequence plus BF16-rounded sequential RS tree, scratch only."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def corrected(P, L, idx, offsets, glse, peer: tl.constexpr, plane: tl.constexpr):
    raw = tl.load(L + peer * plane + idx)
    delta = raw - glse
    delta = tl.where((delta != delta) | (delta == float("inf")), -float("inf"), delta)
    factor = tl.exp2(delta)
    value = tl.load(P + offsets) * factor
    value = tl.where(factor == 0.0, 0.0, value)
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def merge_kernel(
    A,
    B,
    C,
    D,
    L,
    O,  # noqa: E741 - preserve the validated Triton kernel signature.
    PLANE: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    peers = tl.arange(0, 4)
    lse = tl.load(L + peers * PLANE + idx)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0, maximum)
    lse -= maximum
    ex = tl.exp2(lse)
    total = tl.sum(ex, 0)
    glse = tl.log2(total)
    glse += maximum
    offsets = idx * 512 + tl.arange(0, 512)
    v0 = corrected(A, L, idx, offsets, glse, P0, PLANE)
    v1 = corrected(B, L, idx, offsets, glse, P1, PLANE)
    v2 = corrected(C, L, idx, offsets, glse, P2, PLANE)
    v3 = corrected(D, L, idx, offsets, glse, P3, PLANE)
    acc = (v0 + v1).to(tl.bfloat16).to(tl.float32)
    acc = (acc + v2).to(tl.bfloat16).to(tl.float32)
    out_offsets = (
        (idx % 16) * (PLANE // 16) * 512 + (idx // 16) * 512 + tl.arange(0, 512)
    )
    tl.store(O + out_offsets, acc + v3)


def merge(parts, lses, order):
    out = torch.empty(
        (16, parts[0].shape[0], 512), device=parts[0].device, dtype=parts[0].dtype
    ).movedim(0, 1)
    assert all(p.is_contiguous() for p in parts)
    assert lses.is_contiguous()
    merge_kernel[(out.shape[0] * 16,)](
        *(parts[p] for p in order), lses, out, out.shape[0] * 16, *order
    )
    return out
