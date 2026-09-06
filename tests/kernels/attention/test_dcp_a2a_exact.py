# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU bit-exactness tests for the VLLM_DCP_A2A_EXACT combine kernel.

Simulates the 4-rank DCP epilogue locally (no NCCL):

  baseline reference = for each rank r, run the *actual*
  _correct_attn_cp_out_kernel (the ag_rs path's correction) on rank r's
  partial output with the gathered LSEs, then sum the corrected bf16 tensors
  in a given reduction-tree order with a bf16 rounding per node -- exactly
  what NCCL ReduceScatter's per-hop wire rounding does.

  test path = pack each rank's (out, lse) with _dcp_a2a_pack_send_kernel into
  a fake recv buffer (what all_to_all_single would deliver) and run
  _dcp_a2a_unpack_combine_exact with the same tree.

Every reduction tree candidate is exercised, plus adversarial LSEs
(+inf/-inf/NaN, all -inf) and wide-exponent outputs. The real-NCCL probe +
self-validation lives in kbench/probe_a2a_exact.py and in the runtime gate
itself (dcp_a2a_lse_reduce_exact refuses to enable on any bitwise mismatch).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.common import correct_attn_out
from vllm.v1.attention.ops.dcp_alltoall import (
    _apply_tree_fp,
    _dcp_a2a_lse_pack_dim,
    _dcp_a2a_pack_send,
    _dcp_a2a_unpack_combine_exact,
    _enumerate_exact_trees,
)

WORLD = 4
H_TOTAL = 64
D = 512

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)


def _wide(shape, seed, device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mant = torch.rand(shape, generator=gen, dtype=torch.float32) * 1.9 + 0.05
    sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
    expo = torch.randint(-6, 7, shape, generator=gen).to(torch.float32)
    return (mant * sign * (2.0**expo)).to(dtype).to(device)


def _simulated_recv_buffer(outs, lses, my_rank, dtype, device):
    """What rank ``my_rank`` receives from the a2a: every rank's packed
    payload chunk for my head shard, produced by the real pack kernel."""
    B = outs[0].shape[0]
    h_per_rank = H_TOTAL // WORLD
    lse_pack_dim = _dcp_a2a_lse_pack_dim(dtype)
    recv = torch.empty(
        (WORLD, B, h_per_rank, D + lse_pack_dim), dtype=dtype, device=device
    )
    for src in range(WORLD):
        send = torch.empty_like(recv)
        _dcp_a2a_pack_send(
            outs[src], lses[src].contiguous(), send, WORLD, h_per_rank, D,
            lse_pack_dim,
        )
        # all_to_all_single delivers send[my_rank] of rank ``src`` into
        # recv[src] of rank ``my_rank``.
        recv[src] = send[my_rank]
    return recv


def _reference(outs, lses, my_rank, tree, base_e):
    """ag_rs baseline: real correction kernel per rank + tree-ordered
    per-node-rounded summation of the head shard."""
    lses_all = torch.stack([l.contiguous() for l in lses])  # [N, B, H]
    corrected = []
    for r in range(WORLD):
        o = outs[r].clone()
        correct_attn_out(o, lses_all.clone(), r, None, is_lse_base_on_e=base_e)
        corrected.append(o)
    h_lo = my_rank * (H_TOTAL // WORLD)
    h_hi = (my_rank + 1) * (H_TOTAL // WORLD)
    shard = [c[:, h_lo:h_hi] for c in corrected]
    return _apply_tree_fp(shard, tree)


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("base_e", [True, False])
def test_exact_combine_all_trees(B, base_e):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    outs, lses = [], []
    for r in range(WORLD):
        outs.append(_wide((B, H_TOTAL, D), 100 + r + 31 * B, device, dtype))
        gen = torch.Generator(device="cpu").manual_seed(200 + r + 31 * B)
        lses.append((torch.randn((B, H_TOTAL), generator=gen) * 4.0).to(device))
    # adversarial LSEs
    lses[0].view(-1)[0] = float("inf")
    lses[1].view(-1)[0] = float("-inf")
    lses[2].view(-1)[0] = float("nan")

    for my_rank in range(WORLD):
        recv = _simulated_recv_buffer(outs, lses, my_rank, dtype, device)
        for tree in _enumerate_exact_trees():
            got = _dcp_a2a_unpack_combine_exact(
                recv, D, _dcp_a2a_lse_pack_dim(dtype), base_e, tree
            )
            ref = _reference(outs, lses, my_rank, tree, base_e)
            assert torch.equal(got, ref), (
                f"B={B} rank={my_rank} tree={tree} base_e={base_e}: "
                f"{(got != ref).sum().item()} mismatches"
            )


def test_exact_combine_all_neg_inf_lse():
    """All-ranks -inf LSE exercises the lse_max clamp + factor==0 zero-fill."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B = 4
    outs = [_wide((B, H_TOTAL, D), 300 + r, device, dtype) for r in range(WORLD)]
    lses = [
        torch.full((B, H_TOTAL), float("-inf"), device=device) for _ in range(WORLD)
    ]
    tree = (0, (1, 2, 3, 0))
    for my_rank in range(WORLD):
        recv = _simulated_recv_buffer(outs, lses, my_rank, dtype, device)
        got = _dcp_a2a_unpack_combine_exact(
            recv, D, _dcp_a2a_lse_pack_dim(dtype), True, tree
        )
        ref = _reference(outs, lses, my_rank, tree, True)
        assert torch.equal(got, ref)
