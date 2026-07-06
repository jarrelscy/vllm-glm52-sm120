# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AQLM fused-MoE kernels (nvfp4_aqlm_hybrid)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
    _dequant_reference,
    _get_ext,
    _silu_and_mul,
)

E, ENTRIES, G = 8, 65536, 8


def _rand_aqlm(e: int, books: int, m: int, k: int, device: str):
    codes = torch.randint(
        -(2**15), 2**15, (e, books, m, k // G), dtype=torch.int16, device=device
    )
    codebooks = torch.randn(
        books, ENTRIES, G, dtype=torch.float16, device=device
    ) * 0.05
    scales = (
        torch.rand(e, m, dtype=torch.float16, device=device) * 0.5 + 0.75
    )
    return codes, codebooks, scales


@pytest.mark.parametrize("books", [1, 2])
@pytest.mark.parametrize("m,k", [(512, 512), (4096, 6144), (6144, 2048)])
def test_dequant_matches_reference(books: int, m: int, k: int):
    torch.manual_seed(0)
    device = "cuda"
    codes, codebooks, scales = _rand_aqlm(E, books, m, k, device)
    ext = _get_ext()

    experts = torch.arange(E, dtype=torch.int32, device=device)
    got = ext.aqlm_moe_dequant(codes, codebooks, scales, experts)
    ref = _dequant_reference(codes, codebooks, scales)
    torch.testing.assert_close(got, ref.half(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("books", [1, 2])
@pytest.mark.parametrize("m,k", [(4096, 6144), (6144, 2048)])
@pytest.mark.parametrize("n", [1, 7, 64])
def test_gemv_matches_reference(books: int, m: int, k: int, n: int):
    torch.manual_seed(0)
    device = "cuda"
    codes, codebooks, scales = _rand_aqlm(E, books, m, k, device)
    ext = _get_ext()

    x = torch.randn(n, k, dtype=torch.float16, device=device) * 0.1
    expert_ids = torch.randint(0, E, (n,), dtype=torch.int32, device=device)

    got = ext.aqlm_moe_gemv(x, codes, codebooks, scales, expert_ids)

    ref_w = _dequant_reference(codes, codebooks, scales)  # [E, M, K]
    ref = torch.einsum("nk,nmk->nm", x.float(), ref_w[expert_ids.long()].float())
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)


def _rand_nvfp4(e: int, m: int, k: int, s2n: int, device: str):
    packed = torch.randint(0, 256, (e, m, k // 2), dtype=torch.uint8,
                           device=device)
    # fp8 e4m3 bits in a sane positive range (exponent ~2^-3..2^3)
    bscale = torch.randint(48, 72, (e, m, k // 16), dtype=torch.uint8,
                           device=device)
    scale2 = torch.rand(e, s2n, dtype=torch.float32, device=device) * 0.02
    return packed, bscale, scale2


def _dequant_nvfp4_reference(packed, bscale, scale2):
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=packed.device)
    e, m, k2 = packed.shape
    k = k2 * 2
    vals = torch.empty(e, m, k, dtype=torch.float32, device=packed.device)
    vals[..., 0::2] = lut[(packed & 0xF).long()]
    vals[..., 1::2] = lut[(packed >> 4).long()]
    bs = bscale.view(torch.float8_e4m3fn).to(torch.float32)
    vals = vals * bs.repeat_interleave(16, dim=-1)
    s2n = scale2.shape[1]
    rows = (torch.arange(m, device=packed.device) * s2n) // m
    g = scale2[:, rows]  # [E, M]
    return vals * g.unsqueeze(-1)


@pytest.mark.parametrize("m,k,s2n", [(4096, 6144, 2), (6144, 2048, 1),
                                     (512, 512, 2)])
def test_nvfp4_dequant_matches_reference(m: int, k: int, s2n: int):
    torch.manual_seed(0)
    device = "cuda"
    packed, bscale, scale2 = _rand_nvfp4(E, m, k, s2n, device)
    ext = _get_ext()
    experts = torch.arange(E, dtype=torch.int32, device=device)
    got = ext.nvfp4_moe_dequant(packed, bscale, scale2, experts)
    ref = _dequant_nvfp4_reference(packed, bscale, scale2)
    torch.testing.assert_close(got.float(), ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("m,k,s2n", [(4096, 6144, 2), (6144, 2048, 1)])
@pytest.mark.parametrize("n", [1, 7, 33])
def test_nvfp4_gemv_matches_reference(m: int, k: int, s2n: int, n: int):
    torch.manual_seed(0)
    device = "cuda"
    packed, bscale, scale2 = _rand_nvfp4(E, m, k, s2n, device)
    ext = _get_ext()
    x = torch.randn(n, k, dtype=torch.float16, device=device) * 0.1
    expert_ids = torch.randint(0, E, (n,), dtype=torch.int32, device=device)
    expert_ids[0] = -1  # masked slot must produce a zero row

    got = ext.nvfp4_moe_gemv(x, packed, bscale, scale2, expert_ids)
    ref_w = _dequant_nvfp4_reference(packed, bscale, scale2)
    ref = torch.einsum("nk,nmk->nm", x.float(),
                       ref_w[expert_ids.long().clamp(min=0)])
    ref[0] = 0
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)


def test_aqlm_gemv_negative_expert_zeroes_row():
    torch.manual_seed(0)
    device = "cuda"
    codes, codebooks, scales = _rand_aqlm(E, 1, 512, 512, device)
    ext = _get_ext()
    x = torch.randn(3, 512, dtype=torch.float16, device=device)
    ids = torch.tensor([2, -1, 5], dtype=torch.int32, device=device)
    out = ext.aqlm_moe_gemv(x, codes, codebooks, scales, ids)
    assert out[1].abs().max().item() == 0
    assert out[0].abs().max().item() > 0


@pytest.mark.parametrize("n_nvfp4,n_cold", [(0, 0), (0, 2), (3, 2), (3, E - 3)])
def test_end_to_end_moe_paths_agree(n_nvfp4: int, n_cold: int):
    """Decode gemv path and prefill grouped path must produce the same MoE
    output for identical inputs, across per-expert tier mixes."""
    from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
        HybridExpertsMoEMethod,
    )

    torch.manual_seed(0)
    device = "cuda"
    h, i, top_k = 512, 256, 4
    nb = E - n_nvfp4
    nm = nb - n_cold

    # kinds: hot experts get the highest ids' predecessors spread around
    kind = torch.ones(E, dtype=torch.int8)
    hot_ids = [1, 4, 6][:n_nvfp4]
    if n_cold == E - n_nvfp4:  # two-tier: everything non-hot is cold
        cold_ids = [e_ for e_ in range(E) if e_ not in hot_ids]
    else:
        cold_ids = [3, 7][:n_cold]
    for e_ in hot_ids:
        kind[e_] = 0
    for e_ in cold_ids:
        kind[e_] = 2

    class FakeLayer:
        activation = "silu"

    layer = FakeLayer()
    layer.hyb_kind = kind.to(device)
    (layer.w13_codes, layer.w13_codebooks, layer.w13_scales) = _rand_aqlm(
        nb, 1, 2 * i, h, device)
    (layer.w2m_codes, layer.w2m_codebooks, layer.w2m_scales) = _rand_aqlm(
        nm, 2, h, i, device)
    (layer.w2c_codes, layer.w2c_codebooks, layer.w2c_scales) = _rand_aqlm(
        max(n_cold, 0), 1, h, i, device)
    if n_nvfp4 > 0:
        (layer.nvfp4_w13_packed, layer.nvfp4_w13_bscale,
         layer.nvfp4_w13_scale2) = _rand_nvfp4(n_nvfp4, 2 * i, h, 2, device)
        (layer.nvfp4_w2_packed, layer.nvfp4_w2_bscale,
         layer.nvfp4_w2_scale2) = _rand_nvfp4(n_nvfp4, h, i, 1, device)

    class FakeMoe:
        num_experts = E

        class moe_parallel_config:
            tp_size = 1
            ep_size = 1

    m = HybridExpertsMoEMethod.__new__(HybridExpertsMoEMethod)
    m.n_nvfp4, m.n_base, m.n_cold = n_nvfp4, nm, n_cold
    m.moe = FakeMoe()
    m.PREFILL_GROUP = 4
    HybridExpertsMoEMethod.process_weights_after_loading(m, layer)

    n = 16
    x = torch.randn(n, h, dtype=torch.bfloat16, device=device) * 0.1
    logits = torch.randn(n, E, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), top_k)
    topk_ids = topk_ids.to(torch.int32)

    dec = m._apply_gemv(layer, x, topk_weights, topk_ids)
    pre = m._apply_grouped(layer, x, topk_weights, topk_ids)
    torch.testing.assert_close(dec, pre, rtol=3e-2, atol=3e-2)
