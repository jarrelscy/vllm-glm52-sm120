# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact FP32 score bits plus 16-bit local or24-bit global candidate ID."""

from vllm.triton_utils import tl, triton


@triton.jit
def pack(
    logits,
    indices,
    starts,
    output,
    LS0: tl.constexpr,
    LS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    COLS: tl.constexpr,
    RANK: tl.constexpr,
    BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    local = tl.load(indices + row * IS0 + j * IS1)
    start = tl.load(starts + row)
    column = tl.minimum(tl.maximum(local, 0) + start, COLS - 1)
    score = tl.load(
        logits + row * LS0 + column * LS1, mask=local >= 0, other=-float("inf")
    )
    words = output.to(tl.pointer_type(tl.uint32))
    tl.store(words + row * (BYTES * 2048 // 4) + j, score.to(tl.uint32, bitcast=True))
    if BYTES == 6:
        token = tl.where(local >= 0, local, 0xFFFF).to(tl.uint32)
    else:
        token = tl.where(local >= 0, local * 4 + RANK, 0xFFFFFF).to(tl.uint32)
    for plane in tl.static_range(BYTES - 4):
        tl.store(
            output + row * BYTES * 2048 + 8192 + plane * 2048 + j,
            ((token >> (8 * plane)) & 255).to(tl.uint8),
        )
