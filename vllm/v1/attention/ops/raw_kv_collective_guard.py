# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the active collective reduction order before raw-KV execution."""

import torch

from vllm.triton_utils import tl, triton

TREES = ((1, 2, 3, 0), (2, 3, 0, 1), (0, 3, 1, 2), (0, 1, 2, 3))
_RESULTS: dict[tuple[int, int], bool] = {}


@triton.jit
def value(index, rank: tl.constexpr, seed: tl.constexpr):
    h = index.to(tl.uint32) ^ (rank * 7919 + seed)
    h = (h ^ (h >> 16)) * 0x7FEB352D
    h = (h ^ (h >> 15)) * 0x846CA68B
    h = h ^ (h >> 16)
    mant = ((h & 255).to(tl.float32) + 1.0) / 128.0
    exponent = ((h >> 8) % 13).to(tl.int32) - 6
    sign = tl.where((h & 0x80000000) != 0, -1.0, 1.0)
    return (
        (mant * sign * tl.exp2(exponent.to(tl.float32))).to(tl.bfloat16).to(tl.float32)
    )


@triton.jit
def fill(
    P, N: tl.constexpr, RANK: tl.constexpr, SEED: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(P + index, value(index, RANK, SEED), index < N)


@triton.jit
def verify(
    P,
    OK,
    N: tl.constexpr,
    RANK: tl.constexpr,
    SEED: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    global_index = RANK * N + index
    v0 = value(global_index, P0, SEED)
    v1 = value(global_index, P1, SEED)
    v2 = value(global_index, P2, SEED)
    v3 = value(global_index, P3, SEED)
    acc = (v0 + v1).to(tl.bfloat16).to(tl.float32)
    acc = (acc + v2).to(tl.bfloat16).to(tl.float32)
    expected = (acc + v3).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    got = tl.load(P + index, index < N, other=0).to(tl.uint16, bitcast=True)
    mismatch = tl.sum(((index < N) & (expected != got)).to(tl.int32), 0)
    if mismatch > 0:
        tl.atomic_min(OK, 0)


def qualify(group, tokens):
    import torch.distributed as dist

    comm = group.device_communicator.pynccl_comm
    key = (id(comm), tokens)
    if key in _RESULTS:
        return _RESULTS[key]
    if comm is None or comm.disabled:
        return False
    rank = group.rank_in_group
    # At T4096: 256MiB input +64MiB output, no rank-input all-gather.
    x = torch.empty((64, tokens, 512), device=comm.device, dtype=torch.bfloat16)
    y = torch.empty((16, tokens, 512), device=comm.device, dtype=x.dtype)
    ok = torch.ones((), device=comm.device, dtype=torch.int32)
    for seed in (5197, 81719):
        fill[(triton.cdiv(x.numel(), 1024),)](x, x.numel(), rank, seed, 1024)
        comm.reduce_scatter(y, x)
        verify[(triton.cdiv(y.numel(), 1024),)](
            y, ok, y.numel(), rank, seed, *TREES[rank], 1024
        )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group.device_group)
    success = bool(ok.item())
    del x, y, ok
    _RESULTS[key] = success
    return success
