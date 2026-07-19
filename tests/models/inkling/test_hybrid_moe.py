# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the Inkling NVFP4+AQLM hybrid MoE (task #87/#96):
TP-shard + de-interleave loader math, NVFP4/AQLM dequant math (cross-checked
against the checkpoint author's reference semantics in ink_common.py, notably
AQLM's negative-int16-code wraparound), and the hot/cold lookup built by
``process_weights_after_loading``. No GPU / real checkpoint needed.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling.aqlm_hybrid import HybridLayerInfo
from vllm.models.inkling.nvidia import hybrid_moe as hm
from vllm.platforms import current_platform

# Same LUT as ink_common.py's FP4_LUT / hybrid_moe._dequant_nvfp4.
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def test_gateup_deinterleave_shard_loader_splits_and_deinterleaves() -> None:
    # 2*I = 8 on-disk rows, interleaved [g0,u0,g1,u1,g2,u2,g3,u3]; world=2 ->
    # per-rank ish = I/world = 2, so each rank's param has 2*ish = 4 rows.
    loaded = torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(8, 3).contiguous()
    for rank, expected_rows in ((0, [0, 2, 1, 3]), (1, [4, 6, 5, 7])):
        param = torch.nn.Parameter(torch.empty(4, 3))
        hm._gateup_deinterleave_shard_loader(rank, 2, 0)(param, loaded)
        torch.testing.assert_close(param.data, loaded[expected_rows])


def test_row_shard_loader_plain_shard_no_interleave() -> None:
    loaded = torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(8, 3).contiguous()
    for rank, expected_rows in ((0, [0, 1, 2, 3]), (1, [4, 5, 6, 7])):
        param = torch.nn.Parameter(torch.empty(4, 3))
        hm._row_shard_loader(rank, 2, 0)(param, loaded)
        torch.testing.assert_close(param.data, loaded[expected_rows])


def test_row_shard_loader_asserts_on_indivisible_world() -> None:
    loaded = torch.zeros(7, 3)
    param = torch.nn.Parameter(torch.empty(3, 3))
    try:
        hm._row_shard_loader(0, 2, 0)(param, loaded)
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError for 7 % 2 != 0")


def _reference_dequant_nvfp4_one(packed, scale, scale2) -> torch.Tensor:
    """Direct reimplementation of ink_common.py's dequant_nvfp4_raw, for one
    expert (no batch dim), used as the ground truth for hybrid_moe's batched
    _dequant_nvfp4."""
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    out, half = packed.shape
    vals = torch.empty(out, half * 2)
    vals[:, 0::2] = _FP4_LUT[lo]
    vals[:, 1::2] = _FP4_LUT[hi]
    blk = scale.to(torch.float32).repeat_interleave(16, dim=1)
    return vals * blk * scale2.float()


def test_dequant_nvfp4_matches_reference_per_expert() -> None:
    torch.manual_seed(0)
    n, out, in_ = 3, 4, 32
    packed = torch.randint(0, 256, (n, out, in_ // 2), dtype=torch.uint8)
    scale = (torch.rand(n, out, in_ // 16) * 4 + 0.1).to(torch.float8_e4m3fn)
    scale2 = torch.rand(n) + 0.1

    got = hm._dequant_nvfp4(packed, scale, scale2).float()

    for e in range(n):
        expected = _reference_dequant_nvfp4_one(packed[e], scale[e], scale2[e])
        torch.testing.assert_close(got[e], expected.to(torch.bfloat16).float())


def test_dequant_aqlm_matches_reference_direct_indexing() -> None:
    """Ground truth mirrors ink_common.py's aqlm_dequant_expert: plain
    `codebook[codes]` gather relying on PyTorch's negative-index wraparound
    for int16 codes (entries == 2**15, so -1 lands on the last row) -- this
    is the exact correctness question flagged during implementation."""
    torch.manual_seed(1)
    n, out, ng, num_books = 2, 3, 4, 2
    g = 8
    # Book 0: 16 entries, uint8 codes (never negative).
    codes0 = torch.randint(0, 16, (n, out, ng), dtype=torch.uint8)
    cb0 = torch.randn(16, g)
    # Book 1: 8 entries but int16 codes, including negative wraparound values
    # (e.g. -1 should address row 7, the last row of an 8-entry book).
    codes1 = torch.tensor(
        [[-1, 2, -8, 0] * (ng // 4 + 1)] * (n * out), dtype=torch.int16
    )[:, :ng].reshape(n, out, ng)
    cb1 = torch.randn(8, g)

    got = hm._dequant_aqlm([codes0, codes1], [cb0, cb1], torch.ones(n, out))

    for e in range(n):
        for r in range(out):
            acc = torch.zeros(ng, g)
            for c, cb in ((codes0, cb0), (codes1, cb1)):
                idx = c[e, r].long()
                acc += cb[idx]  # PyTorch negative-index gather, the reference semantics
            expected = acc.reshape(ng * g)
            torch.testing.assert_close(
                got[e, r].float(), expected.to(torch.bfloat16).float()
            )


def test_dequant_aqlm_bitmask_equals_negative_index_for_full_range_book() -> None:
    """Explicitly pins the equivalence used to justify hybrid_moe's masking
    approach: for an entries == 2**k book, `idx & (entries-1)` on a raw
    two's-complement code equals plain negative-index addressing."""
    entries = 65536
    cb = torch.randn(entries, 8)
    codes = torch.tensor([-1, -5, -32768, 32767, 0, 12345], dtype=torch.int16).long()
    masked = cb[codes & (entries - 1)]
    direct = cb[codes]  # relies on PyTorch's Python-style negative indexing
    torch.testing.assert_close(masked, direct)


def test_process_weights_after_loading_builds_hot_cold_lookup() -> None:
    layer = SimpleNamespace(
        global_num_experts=6,
        hot_ids=torch.tensor([1, 4], dtype=torch.int32),
        cold_ids=torch.tensor([0, 2, 3, 5], dtype=torch.int32),
    )
    info = HybridLayerInfo(n_hot=2, n_cold=4, packed=True, hot_format="nvfp4")
    method = hm.InklingHybridExpertsMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1)),
        layer_id=5,
        info=info,
        w13_book_entries=[65536],
        w2_book_entries=[65536, 256],
        w13_code_dtypes=["int16"],
        w2_code_dtypes=["int16", "uint8"],
        tp_size=1,
        tp_rank=0,
    )

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(
        layer._hybrid_hot_lookup, torch.tensor([-1, 0, -1, -1, 1, -1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        layer._hybrid_cold_lookup, torch.tensor([0, -1, 1, 2, -1, 3], dtype=torch.int32)
    )


def test_create_weights_shapes_nvfp4_hot() -> None:
    info = HybridLayerInfo(n_hot=2, n_cold=6, packed=True, hot_format="nvfp4")
    method = hm.InklingHybridExpertsMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1)),
        layer_id=3,
        info=info,
        w13_book_entries=[65536],
        w2_book_entries=[65536, 256],
        w13_code_dtypes=["int16"],
        w2_code_dtypes=["int16", "uint8"],
        tp_size=2,
        tp_rank=0,
    )
    layer = torch.nn.Module()
    h, ish = 16, 8  # already TP-divided per-partition intermediate size

    method.create_weights(layer, num_experts=8, hidden_size=h,
                           intermediate_size_per_partition=ish,
                           params_dtype=torch.bfloat16)

    assert layer.hot_ids.shape == (2,)
    assert layer.cold_ids.shape == (6,)
    assert layer.w13_hot_weight.shape == (2, 2 * ish, h // 2)
    assert layer.w13_hot_weight_scale.shape == (2, 2 * ish, h // 16)
    assert layer.w13_hot_weight_scale2.shape == (2,)
    assert layer.w2_hot_weight.shape == (2, h, ish // 2)
    assert not hasattr(layer, "w13_hot_bf16")
    assert layer.w13_cold_codes_0.shape == (6, 2 * ish, h // hm._G)
    assert layer.w13_cold_codebook_0.shape == (65536, hm._G)
    assert layer.w13_cold_scales.shape == (6, 2 * ish)
    assert layer.w2_cold_codes_0.shape == (6, h, ish // hm._G)
    assert layer.w2_cold_codes_1.shape == (6, h, ish // hm._G)
    assert layer.w2_cold_scales.shape == (6, h)


def test_create_weights_shapes_bf16_hot() -> None:
    info = HybridLayerInfo(n_hot=1, n_cold=7, packed=False, hot_format="bf16")
    method = hm.InklingHybridExpertsMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1)),
        layer_id=2,
        info=info,
        w13_book_entries=[65536],
        w2_book_entries=[65536, 256],
        w13_code_dtypes=["int16"],
        w2_code_dtypes=["int16", "uint8"],
        tp_size=1,
        tp_rank=0,
    )
    layer = torch.nn.Module()
    h, ish = 16, 8

    method.create_weights(layer, num_experts=8, hidden_size=h,
                           intermediate_size_per_partition=ish,
                           params_dtype=torch.bfloat16)

    assert layer.w13_hot_bf16.shape == (1, 2 * ish, h)
    assert layer.w2_hot_bf16.shape == (1, h, ish)
    assert not hasattr(layer, "w13_hot_weight")


def test_load_expert_weight_accepts_real_checkpoint_dotted_key_names() -> None:
    """The real jarrelscy/Inkling-512k-NVFP4-AQLM-hybrid checkpoint names
    per-tensor metadata and book indices with a DOT suffix
    (``w13_hot_weight.scale``, ``w13_cold_codes.0``), while the registered
    params use underscores throughout (``w13_hot_weight_scale``,
    ``w13_cold_codes_0``). ``InklingMoE.load_expert_weight`` receives the
    dotted checkpoint key (relative to the mlp module, e.g.
    ``experts.w13_hot_weight.scale``) verbatim and must map it to the
    underscore-named param -- a naive ``getattr(experts, key)`` raises
    AttributeError on the real checkpoint (confirmed against the downloaded
    checkpoint's safetensors index during the #98 smoke test)."""
    from vllm.models.inkling.nvidia.moe import InklingMoE

    info = HybridLayerInfo(n_hot=2, n_cold=6, packed=True, hot_format="nvfp4")
    method = hm.InklingHybridExpertsMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1)),
        layer_id=3,
        info=info,
        w13_book_entries=[65536],
        w2_book_entries=[65536, 256],
        w13_code_dtypes=["int16"],
        w2_code_dtypes=["int16", "uint8"],
        tp_size=1,
        tp_rank=0,
    )
    layer = torch.nn.Module()
    h, ish = 32, 32
    method.create_weights(layer, num_experts=8, hidden_size=h,
                           intermediate_size_per_partition=ish,
                           params_dtype=torch.bfloat16)

    fake_self = SimpleNamespace(
        is_hybrid_layer=True, experts=SimpleNamespace(routed_experts=layer)
    )

    real_key_to_param = {
        "experts.hot_ids": "hot_ids",
        "experts.w13_hot_weight.scale": "w13_hot_weight_scale",
        "experts.w13_hot_weight.scale2": "w13_hot_weight_scale2",
        "experts.w13_cold_codes.0": "w13_cold_codes_0",
        "experts.w13_cold_codebook.0": "w13_cold_codebook_0",
        "experts.w2_cold_codes.1": "w2_cold_codes_1",
    }
    for ckpt_key, param_name in real_key_to_param.items():
        param = getattr(layer, param_name)
        w = (
            torch.randint(0, 50, param.shape, dtype=param.dtype)
            if param.dtype in (torch.int16, torch.int32, torch.uint8)
            else torch.rand(param.shape).to(param.dtype)
        )
        rel = InklingMoE.load_expert_weight(fake_self, ckpt_key, w)
        assert rel == [f"experts.routed_experts.{param_name}"]


def _make_hybrid_method(tp_size: int, tp_rank: int) -> hm.InklingHybridExpertsMoEMethod:
    info = HybridLayerInfo(n_hot=2, n_cold=6, packed=True, hot_format="nvfp4")
    return hm.InklingHybridExpertsMoEMethod(
        SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=1)),
        layer_id=7,
        info=info,
        w13_book_entries=[65536],
        w2_book_entries=[65536, 256],
        w13_code_dtypes=["int16"],
        w2_code_dtypes=["int16", "uint8"],
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _build_and_load(method, num_experts, h, ish, disk, device):
    layer = torch.nn.Module().to(device)
    method.create_weights(
        layer, num_experts=num_experts, hidden_size=h,
        intermediate_size_per_partition=ish, params_dtype=torch.bfloat16,
    )
    layer = layer.to(device)
    for name, tensor in disk.items():
        param = getattr(layer, name)
        param.weight_loader(param, tensor.to(device))
    layer.global_num_experts = num_experts
    method.process_weights_after_loading(layer)
    return layer


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_apply_end_to_end_tp_partial_sums_match_tp1_reference() -> None:
    """Builds the full create_weights -> weight_loader -> process_weights ->
    apply() pipeline on real CUDA tensors for tp_size=1 and tp_size=2, from
    the SAME fabricated on-disk checkpoint tensors, and checks the tp2
    per-rank partial outputs sum to the tp1 reference (mirroring the runner's
    late all-reduce for TP-sharded MoE, per hybrid_moe.py's docstring)."""
    torch.manual_seed(42)
    device = "cuda"
    num_experts, n_hot, n_cold = 8, 2, 6
    h, i_full = 32, 32  # hidden, full (unsharded) intermediate size
    g = hm._G

    hot_ids = torch.tensor([1, 4], dtype=torch.int32)
    cold_ids = torch.tensor([0, 2, 3, 5, 6, 7], dtype=torch.int32)

    # Deliberately small weight magnitudes: large per-expert GEMM outputs
    # combined with near-cancelling top-k weights amplify ordinary bf16
    # rounding noise into a false-looking mismatch (verified separately by
    # hand -- expert outputs of +-100 nearly cancel in the weighted sum, so
    # the ~0.4 absolute bf16 rounding error on each large term swamps the
    # tiny residual). Keeping activations/weights near unit scale avoids
    # that catastrophic-cancellation artifact while still exercising the
    # exact same code path.
    disk = {
        "hot_ids": hot_ids,
        "cold_ids": cold_ids,
        "w13_hot_weight": torch.randint(0, 256, (n_hot, 2 * i_full, h // 2), dtype=torch.uint8),
        "w13_hot_weight_scale": (torch.rand(n_hot, 2 * i_full, h // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w13_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w2_hot_weight": torch.randint(0, 256, (n_hot, h, i_full // 2), dtype=torch.uint8),
        "w2_hot_weight_scale": (torch.rand(n_hot, h, i_full // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w2_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w13_cold_codes_0": torch.randint(-32768, 32767, (n_cold, 2 * i_full, h // g), dtype=torch.int16),
        "w13_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w13_cold_scales": torch.rand(n_cold, 2 * i_full).to(torch.float16) * 0.1 + 0.02,
        "w2_cold_codes_0": torch.randint(-32768, 32767, (n_cold, h, i_full // g), dtype=torch.int16),
        "w2_cold_codes_1": torch.randint(0, 256, (n_cold, h, i_full // g), dtype=torch.uint8),
        "w2_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w2_cold_codebook_1": torch.randn(256, g, dtype=torch.float16) * 0.05,
        "w2_cold_scales": torch.rand(n_cold, h).to(torch.float16) * 0.1 + 0.02,
    }

    num_tokens, top_k = 5, 2
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64, device=device)
    topk_weights = torch.rand(num_tokens, top_k, device=device)

    ref_method = _make_hybrid_method(tp_size=1, tp_rank=0)
    ref_layer = _build_and_load(ref_method, num_experts, h, i_full, disk, device)
    ref_out = ref_method.apply(ref_layer, x, topk_weights, topk_ids, None, None)

    partial_sum = torch.zeros(num_tokens, h, dtype=ref_out.dtype, device=device)
    for rank in range(2):
        method = _make_hybrid_method(tp_size=2, tp_rank=rank)
        layer = _build_and_load(method, num_experts, h, i_full // 2, disk, device)
        partial_sum += method.apply(layer, x, topk_weights, topk_ids, None, None)

    torch.testing.assert_close(partial_sum, ref_out, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_decode_graphsafe_path_matches_eager_reference() -> None:
    """The fixed-shape, host-sync-free decode path (_apply_decode_graphsafe,
    used when num_tokens < _RESERVE_MEASURE_MIN_TOKENS) must produce the same
    result as the eager per-expert loop it replaces. Runs both on identical
    inputs -- graph-safe by default, then eager by forcing the threshold to 0
    -- and asserts they agree. This is the correctness gate for the CUDA-graph
    decode rewrite (task #120)."""
    torch.manual_seed(7)
    device = "cuda"
    # n_hot/n_cold fixed by _make_hybrid_method (HybridLayerInfo n_hot=2/n_cold=6).
    num_experts, n_hot, n_cold = 8, 2, 6
    h, i_full = 32, 32
    g = hm._G

    hot_ids = torch.tensor([0, 6], dtype=torch.int32)
    cold_ids = torch.tensor([1, 2, 3, 4, 5, 7], dtype=torch.int32)
    disk = {
        "hot_ids": hot_ids,
        "cold_ids": cold_ids,
        "w13_hot_weight": torch.randint(0, 256, (n_hot, 2 * i_full, h // 2), dtype=torch.uint8),
        "w13_hot_weight_scale": (torch.rand(n_hot, 2 * i_full, h // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w13_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w2_hot_weight": torch.randint(0, 256, (n_hot, h, i_full // 2), dtype=torch.uint8),
        "w2_hot_weight_scale": (torch.rand(n_hot, h, i_full // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w2_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w13_cold_codes_0": torch.randint(-32768, 32767, (n_cold, 2 * i_full, h // g), dtype=torch.int16),
        "w13_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w13_cold_scales": torch.rand(n_cold, 2 * i_full).to(torch.float16) * 0.1 + 0.02,
        "w2_cold_codes_0": torch.randint(-32768, 32767, (n_cold, h, i_full // g), dtype=torch.int16),
        "w2_cold_codes_1": torch.randint(0, 256, (n_cold, h, i_full // g), dtype=torch.uint8),
        "w2_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w2_cold_codebook_1": torch.randn(256, g, dtype=torch.float16) * 0.05,
        "w2_cold_scales": torch.rand(n_cold, h).to(torch.float16) * 0.1 + 0.02,
    }

    method = _make_hybrid_method(tp_size=1, tp_rank=0)
    layer = _build_and_load(method, num_experts, h, i_full, disk, device)

    # Exercise several decode batch sizes (S = num_tokens*top_k spans one and
    # multiple _DECODE_SLOT_CHUNK chunks) and top_k values.
    for num_tokens, top_k in [(1, 2), (3, 2), (5, 4), (8, 6)]:
        x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64, device=device)
        topk_weights = torch.rand(num_tokens, top_k, device=device)

        out_gs = method.apply(layer, x, topk_weights, topk_ids, None, None)

        orig = hm._RESERVE_MEASURE_MIN_TOKENS
        try:
            hm._RESERVE_MEASURE_MIN_TOKENS = 0  # force the eager per-expert path
            out_eager = method.apply(layer, x, topk_weights, topk_ids, None, None)
        finally:
            hm._RESERVE_MEASURE_MIN_TOKENS = orig

        torch.testing.assert_close(
            out_gs, out_eager, rtol=0.02, atol=0.02,
            msg=f"graphsafe vs eager mismatch at num_tokens={num_tokens} top_k={top_k}",
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_decode_grouped_matches_gemv_path() -> None:
    """The graph-safe grouped decode path (_apply_decode_grouped, used for
    multi-token decode batches like spec-decode verify) must agree with the
    per-slot gemv path on identical inputs. Covers the MTP verify shape
    (9 tokens), expert runs longer than one m-block (concentrated routing
    -> multi-block runs), and all-hot / all-cold routing."""
    torch.manual_seed(11)
    device = "cuda"
    num_experts, n_hot, n_cold = 8, 2, 6
    h, i_full = 32, 32
    g = hm._G

    hot_ids = torch.tensor([0, 6], dtype=torch.int32)
    cold_ids = torch.tensor([1, 2, 3, 4, 5, 7], dtype=torch.int32)
    disk = {
        "hot_ids": hot_ids,
        "cold_ids": cold_ids,
        "w13_hot_weight": torch.randint(0, 256, (n_hot, 2 * i_full, h // 2), dtype=torch.uint8),
        "w13_hot_weight_scale": (torch.rand(n_hot, 2 * i_full, h // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w13_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w2_hot_weight": torch.randint(0, 256, (n_hot, h, i_full // 2), dtype=torch.uint8),
        "w2_hot_weight_scale": (torch.rand(n_hot, h, i_full // 16) * 0.1 + 0.02).to(torch.float8_e4m3fn),
        "w2_hot_weight_scale2": torch.rand(n_hot) * 0.1 + 0.02,
        "w13_cold_codes_0": torch.randint(-32768, 32767, (n_cold, 2 * i_full, h // g), dtype=torch.int16),
        "w13_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w13_cold_scales": torch.rand(n_cold, 2 * i_full).to(torch.float16) * 0.1 + 0.02,
        "w2_cold_codes_0": torch.randint(-32768, 32767, (n_cold, h, i_full // g), dtype=torch.int16),
        "w2_cold_codes_1": torch.randint(0, 256, (n_cold, h, i_full // g), dtype=torch.uint8),
        "w2_cold_codebook_0": torch.randn(65536, g, dtype=torch.float16) * 0.05,
        "w2_cold_codebook_1": torch.randn(256, g, dtype=torch.float16) * 0.05,
        "w2_cold_scales": torch.rand(n_cold, h).to(torch.float16) * 0.1 + 0.02,
    }

    method = _make_hybrid_method(tp_size=1, tp_rank=0)
    layer = _build_and_load(method, num_experts, h, i_full, disk, device)

    # (num_tokens, top_k, id_pool_size): the last case routes 180 slots onto
    # 2 experts so runs span many block_m=16 m-blocks.
    for num_tokens, top_k, id_hi in [(9, 6, 8), (2, 2, 8), (5, 3, 8), (30, 6, 2)]:
        x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(
            0, id_hi, (num_tokens, top_k), dtype=torch.int64, device=device
        )
        topk_weights = torch.rand(num_tokens, top_k, device=device)
        out_grouped = method._apply_decode_grouped(layer, x, topk_weights, topk_ids)
        out_gemv = method._apply_decode_fused(layer, x, topk_weights, topk_ids)
        torch.testing.assert_close(
            out_grouped, out_gemv, rtol=0.02, atol=0.02,
            msg=f"grouped vs gemv mismatch at num_tokens={num_tokens} "
                f"top_k={top_k} id_hi={id_hi}",
        )

    # All-hot and all-cold routing at the MTP verify shape: exercises both
    # uniform branches of the grouped kernel end to end.
    for pool_cpu in (hot_ids, cold_ids):
        pool = pool_cpu.long().to(device)
        topk_ids = pool[torch.randint(0, pool.numel(), (9, 6), device=device)]
        x = torch.randn(9, h, dtype=torch.bfloat16, device=device)
        topk_weights = torch.rand(9, 6, device=device)
        out_grouped = method._apply_decode_grouped(layer, x, topk_weights, topk_ids)
        out_gemv = method._apply_decode_fused(layer, x, topk_weights, topk_ids)
        torch.testing.assert_close(out_grouped, out_gemv, rtol=0.02, atol=0.02)


def test_ep_size_greater_than_one_not_implemented() -> None:
    info = HybridLayerInfo(n_hot=1, n_cold=1, packed=True, hot_format="nvfp4")
    try:
        hm.InklingHybridExpertsMoEMethod(
            SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=2)),
            layer_id=2,
            info=info,
            w13_book_entries=[65536],
            w2_book_entries=[65536, 256],
            w13_code_dtypes=["int16"],
            w2_code_dtypes=["int16", "uint8"],
            tp_size=1,
            tp_rank=0,
        )
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for ep_size > 1")
