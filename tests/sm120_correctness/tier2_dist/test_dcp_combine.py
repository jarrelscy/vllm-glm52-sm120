# SPDX-License-Identifier: Apache-2.0
"""TIER 2 — DCP output-combine equivalence: ag_rs vs a2a vs CPU reference.

Guards idea 8 (DCP backend a2a vs ag_rs) and any future rewrite of the
LSE-weighted combine.  Both backends must implement the SAME
softmax(LSE)-weighted combination of per-rank partial attention outputs;
they may only differ by floating-point summation ORDER.

Single-GPU part (runs under the kernel-tier GPU budget, no comms):
  * the a2a triton pack kernel packs outputs + fp32 LSE (bit-split into
    two bf16 slots) losslessly;
  * the a2a triton unpack+combine kernel matches the pure-torch CPU
    reference `_lse_weighted_combine` within fp reorder tolerance;
  * degenerate rows: a rank with an empty KV shard (LSE = -inf / NaN /
    +inf) contributes zero weight, all-(-inf) rows produce zeros, not
    NaN (the masked-fill path).

Multi-GPU part (torchrun, 2-4 GPUs — run via launch_dist_tests.sh):
  * dcp_a2a_lse_reduce == cp_lse_ag_out_rs on identical inputs, per
    rank, within fp reorder tolerance.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")

# Real serving shape: DCP4, 4 verify tokens, 48/4=12 heads/rank... use
# several including the production-like one.
SHAPES = [
    # (world_size N, tokens B, total heads H, head dim D)
    (4, 4, 32, 128),
    (4, 1, 32, 128),
    (2, 8, 16, 64),
]


@pytest.fixture(scope="module")
def gpu():
    from common.kernels import pick_gpu
    idx = pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


def _synth(n, b, h, d, device, seed=0, degenerate=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    outs = torch.randn(n, b, h // n, d, generator=g).to(torch.bfloat16)
    lses = (torch.randn(n, b, h // n, generator=g) * 3).float()
    if degenerate == "one_rank_empty":
        lses[1] = float("-inf")            # empty KV shard on rank 1
    elif degenerate == "nan_inf":
        lses[0, 0, 0] = float("nan")
        lses[1, 0, 0] = float("inf")
    elif degenerate == "all_empty_row":
        lses[:, 0, 0] = float("-inf")      # nobody attends this (b,h)
    return outs.to(device), lses.to(device)


def _pack_unpack_combine(outs, lses, device):
    """Drive the a2a pack + unpack/combine kernels WITHOUT comms.

    Packing all ranks' shards locally and treating the packed buffer as
    the recv buffer is numerically identical to what each rank sees
    after all_to_all_single.
    """
    from vllm.v1.attention.ops.dcp_alltoall import (
        _dcp_a2a_lse_pack_dim,
        _dcp_a2a_pack_send,
        _dcp_a2a_unpack_combine,
    )
    n, b, hpr, d = outs.shape
    lse_pack_dim = _dcp_a2a_lse_pack_dim(outs.dtype)
    send = torch.empty((n, b, hpr, d + lse_pack_dim), device=device,
                       dtype=outs.dtype)
    # emulate rank r's local view: full [B, H] tensors
    full_out = outs.permute(1, 0, 2, 3).reshape(b, n * hpr, d).contiguous()
    full_lse = lses.permute(1, 0, 2).reshape(b, n * hpr).contiguous()
    _dcp_a2a_pack_send(full_out, full_lse, send, n, hpr, d, lse_pack_dim)
    return _dcp_a2a_unpack_combine(send, d, lse_pack_dim,
                                   return_lse=True, is_lse_base_on_e=True)


def test_pack_roundtrip_lossless(gpu):
    """fp32 LSE bit-split into two bf16 slots must round-trip exactly."""
    from vllm.v1.attention.ops.dcp_alltoall import (
        _dcp_a2a_lse_pack_dim,
        _dcp_a2a_pack_send,
    )
    n, b, h, d = 4, 4, 32, 128
    outs, lses = _synth(n, b, h, d, gpu, seed=1)
    hpr = h // n
    lse_pack_dim = _dcp_a2a_lse_pack_dim(outs.dtype)
    assert lse_pack_dim == 2  # bf16 payload -> split fp32 into 2x16 bits
    send = torch.empty((n, b, hpr, d + lse_pack_dim), device=gpu,
                       dtype=outs.dtype)
    full_out = outs.permute(1, 0, 2, 3).reshape(b, h, d).contiguous()
    full_lse = lses.permute(1, 0, 2).reshape(b, h).contiguous()
    _dcp_a2a_pack_send(full_out, full_lse, send, n, hpr, d, lse_pack_dim)
    # output payload verbatim
    got_out = send[..., :d]
    assert torch.equal(got_out.reshape(n, b, hpr, d), outs)
    # LSE bits verbatim
    lo = send[..., d].view(torch.uint16).to(torch.int64)
    hi = send[..., d + 1].view(torch.uint16).to(torch.int64)
    got_lse = ((hi << 16) | lo).to(torch.int32).view(torch.float32)
    assert torch.equal(got_lse.reshape(n, b, hpr), lses)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"N{s[0]}B{s[1]}H{s[2]}D{s[3]}")
@pytest.mark.parametrize("degenerate", [None, "one_rank_empty", "nan_inf",
                                        "all_empty_row"])
def test_a2a_combine_matches_reference(shape, degenerate, gpu):
    from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine
    n, b, h, d = shape
    outs, lses = _synth(n, b, h, d, gpu, seed=2, degenerate=degenerate)
    got, got_lse = _pack_unpack_combine(outs, lses, gpu)

    ref, ref_lse = _lse_weighted_combine(outs.float().cpu(), lses.cpu(),
                                         return_lse=True)
    got_f = got.float().cpu()

    assert torch.isfinite(got_f).all(), \
        f"combine produced non-finite values (degenerate={degenerate})"
    # tolerance: fp reorder + bf16 storage of the combined output
    atol = 2 * 2.0**-8 * ref.abs().max().item() + 1e-6
    torch.testing.assert_close(got_f, ref.reshape(got_f.shape), atol=atol,
                               rtol=2.0**-7)
    # global LSE compared in fp32 (a2a returns it in fp32)
    finite = torch.isfinite(ref_lse.reshape(-1))
    torch.testing.assert_close(
        got_lse.float().cpu().reshape(-1)[finite],
        ref_lse.reshape(-1)[finite], atol=1e-3, rtol=1e-3)


def test_all_empty_row_zero_not_nan(gpu):
    """All ranks empty for a row -> combined output must be finite
    (masked fill), never NaN — NaN here would silently poison the layer."""
    n, b, h, d = 4, 2, 16, 64
    outs, lses = _synth(n, b, h, d, gpu, seed=3, degenerate="all_empty_row")
    got, _ = _pack_unpack_combine(outs, lses, gpu)
    assert torch.isfinite(got.float()).all()


def test_ag_rs_correction_matches_reference(gpu):
    """Emulate the full ag_rs combine WITHOUT comms and compare it to the
    same CPU reference the a2a test uses (transitively: ag_rs == a2a).

    In ag_rs each rank r corrects its own partial output with the
    all-gathered LSEs (weight = exp(lse_r - lse_global)) and the
    reduce_scatter then SUMS the corrected outputs across ranks:
        combined = sum_r correct_attn_out(out_r, lses, r)
    """
    from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out
    from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

    n, b, h, d = 4, 4, 32, 128
    # per-(b,h) shard layout: each rank holds ALL H heads of its partial
    # attention over its KV shard -> outs_full[r]: [B, H, D]
    g = torch.Generator(device="cpu").manual_seed(4)
    outs_full = torch.randn(n, b, h, d, generator=g).float().to(gpu)
    lses_full = (torch.randn(n, b, h, generator=g) * 3).float().to(gpu)

    combined = torch.zeros(b, h, d, device=gpu)
    for r in range(n):
        ctx = CPTritonContext()
        corrected, _ = correct_attn_out(
            outs_full[r].clone(), lses_full.contiguous(), r, ctx)
        combined += corrected

    ref = _lse_weighted_combine(outs_full.cpu(), lses_full.cpu())
    atol = 2 * 2.0**-8 * ref.abs().max().item() + 1e-6
    torch.testing.assert_close(combined.cpu(), ref, atol=atol, rtol=2.0**-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
