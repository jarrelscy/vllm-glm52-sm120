# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v3 storage oracle, loader contracts, and real SM120 routed MLP parity."""

import json
import os

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as runtime
from vllm.model_executor.layers.quantization.arvq_reference import activation_planes
from vllm.model_executor.layers.quantization.nvfp4_arvq_prefill import dequantize_cold

FORMAT = "rvq256_256x8_expert"
LEVELS = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM120")


def fixture(n, k, experts=2, seed=17, extreme=False):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, 65536, (experts, n, k // 8), dtype=np.uint16)
    pairs[..., 0] = 0
    pairs[..., -1] = 65535
    books = rng.integers(0, 2**32, (experts, 512), dtype=np.uint32)
    scales = rng.integers(40, 57, (experts, n, k // 128), dtype=np.uint8)
    if extreme:
        scales[..., 0] = np.resize(np.array([0, 1, 7, 8, 126], np.uint8), (experts, n))
    # Natural rows -> existing v2 MMA fragments (j, lane), then pairs/u32.
    fragments = pairs.reshape(experts, n // 16, 2, 8, k // 64, 2, 4)
    fragments = (
        fragments.transpose(0, 1, 4, 5, 2, 3, 6)
        .copy()
        .reshape(experts, n // 16, k // 64, 128)
    )
    packed = fragments[..., ::2].astype(np.uint32) | (
        fragments[..., 1::2].astype(np.uint32) << 16
    )
    scale_storage = (
        scales.reshape(experts, n // 16, 16, k // 128).transpose(0, 1, 3, 2).copy()
    )
    return (
        tuple(torch.from_numpy(a) for a in (packed, books, scale_storage)),
        pairs,
        scales,
    )


def oracle(pairs, books, scales, alpha=1.0):
    """CPU oracle from natural indices, independent of runtime packed decoder."""
    words = books.numpy().astype(np.uint32)
    values = LEVELS[(words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15]
    e = np.arange(len(words))[:, None, None]
    weights = values[e, pairs & 255] + values[e, 256 + (pairs >> 8)]
    sc = scales.astype(np.int32)
    decoded = np.where(
        sc >> 3 == 0, (sc & 7) * 2.0**-9, (1 + (sc & 7) / 8) * np.exp2((sc >> 3) - 7)
    ).astype(np.float32)
    weights = weights.reshape(len(words), pairs.shape[1], -1)
    return torch.from_numpy(weights * np.repeat(decoded, 128, axis=-1) * alpha)


def method(rank=0, tp=1, expert=True):
    m = object.__new__(runtime.ArvqExpertsMoEMethod)
    m._tp, m._tpr = tp, rank
    m.n_cold, m.n_nvfp4, m._chunk_tokens = 2, 1, 128
    m.arvq_format = FORMAT if expert else "rvq256_256x8"
    # Deliberately not the global-ID order: slot 0 -> ID 2, slot 1 -> ID 0.
    m.cold_expert_ids = [2, 0]
    return m


def layer_from(
    nhidden, intermediate, f13, f2, rank=0, tp=1, expert=True, device="cuda"
):
    m = method(rank, tp, expert)
    with torch.device(device):
        layer = torch.nn.Module()
        m.create_weights(layer, 3, nhidden, intermediate // tp, torch.bfloat16)
    for name, param in layer.named_parameters():
        if name == "hyb_kind":
            param.data.copy_(torch.tensor([2, 0, 2], dtype=torch.int8, device=device))
        elif name.startswith("arvq_"):
            proj = f13 if "w13" in name else f2
            if name.endswith("packed"):
                source = proj[0]
            elif name.endswith("codebooks"):
                source = proj[1] if expert else proj[1][0]
            elif name.endswith("scales"):
                source = proj[2]
            else:
                source = torch.tensor([1 / 256], dtype=torch.float32)
            param.weight_loader(param, source.to(device))
        elif name.endswith("scale2"):
            param.data.fill_(1 / 256)
        elif "bscale" in name:
            param.data.fill_(56)
        else:
            param.data.fill_(0x22)  # nonzero hot expert
    m.process_weights_after_loading(layer)
    return layer


def report(name, actual, expected, tolerance=0.003):
    a, b = actual.float(), expected.float()
    diff = a - b
    rel = (diff.norm() / b.norm().clamp_min(1e-20)).item()
    row = dict(case=name, max_abs=diff.abs().max().item(), relative_l2=rel)
    path = os.environ.get("ARVQ_V3_PARITY_REPORT")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(row) + "\n")
    assert torch.isfinite(a).all(), row
    assert rel <= tolerance, row


def test_v3_metadata(monkeypatch):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    monkeypatch.setattr(ModelOptNvFp4Config, "from_config", lambda _: None)
    marker = dict(
        format=FORMAT,
        version=3,
        codebook_scope="expert",
        codebook_sizes=[256, 256],
        activation_planes=4,
        weight_scale_group=128,
    )
    cfg = dict(
        arvq=marker,
        nvfp4={},
        aqlm_layer_books={
            "3": dict(n_base=0, n_cold=2, n_nvfp4=1, cold_expert_ids=[2, 0])
        },
    )
    parsed = runtime.NvFp4ArvqHybridConfig.from_config(cfg)
    assert parsed.arvq_format == FORMAT
    assert parsed.aqlm_layer_books[3]["cold_expert_ids"] == [2, 0]
    for key, wrong in [
        ("version", 2),
        ("codebook_scope", "shared"),
        ("codebook_sizes", [256, 128]),
        ("activation_planes", 1),
    ]:
        bad = {**cfg, "arvq": {**marker, key: wrong}}
        with pytest.raises(ValueError, match="metadata"):
            runtime.NvFp4ArvqHybridConfig.from_config(bad)


@pytest.mark.parametrize("rank", range(4))
def test_v3_books_replicated_and_no_broadcast(rank):
    m = method(rank, 4)
    layer = torch.nn.Module()
    m.create_weights(layer, 3, 128, 128, torch.bfloat16)
    for proj in ("w13", "w2"):
        cb = getattr(layer, f"arvq_{proj}_codebooks")
        expected = torch.arange(1024).reshape(2, 512).to(torch.uint32)
        cb.weight_loader(cb, expected)
        assert torch.equal(cb, expected)
        for wrong in [
            expected[0],
            expected[:1],
            expected.T.contiguous(),
            expected.float(),
        ]:
            with pytest.raises(ValueError, match="shape/dtype"):
                cb.weight_loader(cb, wrong)


@GPU
@pytest.mark.parametrize("n,k", [(256, 128), (128, 256)])
def test_distinct_books_cpu_oracle(n, k):
    (packed, books, scales), pairs, natural_scales = fixture(n, k, extreme=True)
    # Force the same indices/scales in two experts; only books differ.
    packed[1].copy_(packed[0])
    pairs[1] = pairs[0]
    scales[1].copy_(scales[0])
    natural_scales[1] = natural_scales[0]
    books[0].fill_(0x22222222)
    books[1].fill_(0x99999999)
    ref = oracle(pairs, books, natural_scales)
    for e in range(2):
        args = [a[e].cuda() for a in (packed, books, scales)]
        out = dequantize_cold(*args, 1.0, n, k)
        assert torch.equal(out.cpu(), ref[e].half())
    assert (ref[0] >= 0).all() and (ref[1] <= 0).all()
    assert not torch.equal(ref[0], ref[1])


@GPU
@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("tokens", [1, 4, 33])
def test_actual_shapes_reference_and_duplicate_parity(tp, tokens, monkeypatch):
    torch.manual_seed(77)
    h, intermediate = 6144, 2048
    f13, p13, s13 = fixture(2 * intermediate, h)
    f2, p2, s2 = fixture(h, intermediate, seed=19)
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.25
    ids = torch.tensor([[2, 0], [0, 2]], device="cuda").repeat((tokens + 1) // 2, 1)[
        :tokens
    ]
    weights = torch.tensor([[0.3, 0.7]], device="cuda").repeat(tokens, 1)
    refs = [oracle(p13, f13[1], s13, 1 / 256), oracle(p2, f2[1], s2, 1 / 256)]
    totals, expected_totals = [], []
    for rank in range(tp):
        layer = layer_from(h, intermediate, f13, f2, rank, tp)
        mapped = layer._arvq_lookups[0, ids.flatten()].long()
        assert mapped[0].item() == 0
        cold = mapped.int()
        hot = torch.full_like(cold, -1)
        xr = x.half().repeat_interleave(2, 0)
        ish = intermediate // tp
        n13 = 2 * ish
        gateup = runtime._projection(
            xr,
            cold,
            hot,
            layer._arvq_tensors[:6],
            1 / 256,
            n13,
            16 if len(xr) <= 32 else 8,
            2,
        )
        w13 = torch.cat(
            (
                refs[0][:, rank * ish : (rank + 1) * ish],
                refs[0][:, intermediate + rank * ish : intermediate + (rank + 1) * ish],
            ),
            1,
        ).cuda()
        w2 = refs[1][:, :, rank * ish : (rank + 1) * ish].cuda()
        px = activation_planes(xr)
        expected13 = torch.stack(
            [px[i] @ w13[e].T for i, e in enumerate(mapped.tolist())]
        )
        label = f"tp{tp}-rank{rank}-t{tokens}"
        report(label + "-gateup", gateup, expected13, 2e-5)
        gate, up = gateup.half().chunk(2, -1)
        act = (torch.nn.functional.silu(gate) * up).half()
        eg, eu = expected13.half().chunk(2, -1)
        expected_act = (torch.nn.functional.silu(eg) * eu).half()
        report(label + "-swiglu", act, expected_act)
        down = runtime._projection(
            act, cold, hot, layer._arvq_tensors[6:], 1 / 256, h, 2, 1
        )
        pa = activation_planes(expected_act)
        expected_down = torch.stack(
            [pa[i] @ w2[e].T for i, e in enumerate(mapped.tolist())]
        )
        report(label + "-down", down, expected_down)
        result = runtime.arvq_mlp(
            x,
            weights,
            ids,
            layer._arvq_lookups,
            layer._arvq_tensors,
            layer._arvq_alphas,
            128,
        )
        expected = (
            (expected_down.view(tokens, 2, h) * weights[..., None]).sum(1).to(x.dtype)
        )
        report(label + "-moe", result, expected, 0.006)
        totals.append(result.float())
        expected_totals.append(expected.float())
        # Reuse the same packed weights/scales but duplicate book 0 for both slots.
        duplicated = list(layer._arvq_tensors)
        shared = list(layer._arvq_tensors)
        for offset in (1, 7):
            shared[offset] = layer._arvq_tensors[offset][0].clone()
            duplicated[offset] = shared[offset].repeat(2, 1)

        def run(tensors, layer=layer):
            return runtime.arvq_mlp(
                x, weights, ids, layer._arvq_lookups, tensors, layer._arvq_alphas, 128
            )

        base = run(shared)
        assert torch.equal(base, run(duplicated))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(duplicated)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(base, captured)
        # Reorder slots, books, weights and scales together; preserve global IDs.
        permuted = list(layer._arvq_tensors)
        for offset, n, k in [(0, n13, h), (6, h, ish)]:
            packed = permuted[offset][:-1].reshape(2, n // 16, k // 64, 64)
            permuted[offset] = torch.cat(
                (packed.flip(0).flatten(), packed.new_zeros(1))
            )
            permuted[offset + 1] = permuted[offset + 1].flip(0).contiguous()
            permuted[offset + 2] = permuted[offset + 2].flip(0).contiguous()
        lookups = layer._arvq_lookups.clone()
        lookups[0] = torch.where(lookups[0] >= 0, 1 - lookups[0], -1)
        reordered = runtime.arvq_mlp(
            x, weights, ids, lookups, permuted, layer._arvq_alphas, 128
        )
        assert torch.equal(result, reordered)
        if tokens <= 4:
            monkeypatch.setenv("VLLM_ARVQ_FUSED_ACTIVATION_PACK", "1")
            assert torch.equal(run(duplicated), base)
            monkeypatch.setenv("VLLM_ARVQ_FUSED_ACTIVATION_PACK", "0")
    report(f"tp{tp}-t{tokens}-summed", sum(totals), sum(expected_totals), 0.006)


@GPU
def test_known_projection_uses_cold_slot():
    tensors, _, _ = fixture(128, 128)
    packed, books, scales = [t.cuda() for t in tensors]
    packed[1].copy_(packed[0])
    scales.fill_(56)
    books[0].fill_(0x22222222)
    books[1].fill_(0x99999999)
    dummy = torch.zeros(1, device="cuda", dtype=torch.int32)
    args = [packed, books, scales, dummy, dummy, dummy]
    x = torch.ones(2, 128, device="cuda", dtype=torch.float16)
    cold = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    hot = torch.full_like(cold, -1)
    out = runtime._projection(x, cold, hot, args, 1.0, 128, 2, 1)
    assert torch.equal(out[0], torch.full_like(out[0], -128.0))
    assert torch.equal(out[1], torch.full_like(out[1], 256.0))


@GPU
@pytest.mark.parametrize("compact", ["0", "1"])
def test_grouped_prefill_mixed_routes(compact, monkeypatch):
    from vllm.model_executor.layers.quantization.nvfp4_arvq_prefill import (
        grouped_cold_prefill,
    )

    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", compact)
    torch.manual_seed(31)
    f13, _, _ = fixture(256, 128)
    f2, _, _ = fixture(128, 128, seed=22)
    layer = layer_from(128, 128, f13, f2)
    x = torch.randn(40, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 2], [2, 0, 1]], device="cuda").repeat(20, 1)
    weights = torch.tensor([[0.2, 0.3, 0.5]], device="cuda").repeat(40, 1)

    def run(tensors, layer=layer):
        return grouped_cold_prefill(
            x,
            weights,
            ids,
            layer._arvq_lookups,
            tensors,
            layer._arvq_alphas,
            projection=runtime._projection,
            min_expert_tokens=4,
            chunk_tokens=8,
        )

    tensors = layer._arvq_tensors
    distinct = run(tensors)
    # Independent expert execution also checks route weights/scatter.
    separate = []
    for e in (0, 1, 2):
        mask_weights = weights * (ids == e)
        separate.append(
            grouped_cold_prefill(
                x,
                mask_weights,
                ids,
                layer._arvq_lookups,
                tensors,
                layer._arvq_alphas,
                projection=runtime._projection,
                min_expert_tokens=4,
                chunk_tokens=8,
            ).float()
        )
    report("grouped-mixed-" + compact, distinct, sum(separate), 0.006)
    shared, duplicate = list(tensors), list(tensors)
    for offset in (1, 7):
        shared[offset] = tensors[offset][0].clone()
        duplicate[offset] = shared[offset].repeat(2, 1)
    assert torch.equal(run(shared), run(duplicate))
    assert not torch.equal(distinct, run(duplicate))

    # Native mixed hot/cold routes and graph, with the same duplicated books.
    def native(t):
        return runtime.arvq_mlp(
            x, weights, ids, layer._arvq_lookups, t, layer._arvq_alphas, 8
        )

    expected = native(shared)
    assert torch.equal(expected, native(duplicate))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = native(duplicate)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(expected, out)


@GPU
def test_fused_gather_selects_expert_books():
    from vllm.model_executor.layers.quantization.nvfp4_arvq_cold_gather import (
        decode_gather,
    )
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
        _expert_codebooks,
    )

    f, pairs, scales = fixture(1024, 6144)
    reference = oracle(pairs, f[1], scales, 1 / 256)
    packed, books, cs = [t.cuda() for t in f]
    x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
    routes = torch.tensor([8, 0, 16, 8], device="cuda")
    for e in (0, 1):
        rows, weights = decode_gather(
            x,
            routes,
            packed[e],
            _expert_codebooks(books, e),
            cs[e],
            1 / 256,
            1024,
            6144,
            8,
        )
        assert torch.equal(rows, x[routes // 8].half())
        assert torch.equal(weights.cpu(), reference[e].half())


def test_expert_abi_never_aliases_shared():
    from types import SimpleNamespace

    old = SimpleNamespace(argtypes=[])
    shared, expert = SimpleNamespace(), SimpleNamespace()
    lib = SimpleNamespace(
        hybrid_launch=old, hybrid_launch_8x8=shared, hybrid_launch_8x8_expert=expert
    )
    assert runtime._launch_for_codebooks(lib, torch.empty(512)) is shared
    assert runtime._launch_for_codebooks(lib, torch.empty(1, 512)) is expert
    assert runtime._launch_for_codebooks(lib, torch.empty(2, 512)) is expert
    for shape in [(512, 1), (2, 384), (1024,), (2, 2, 128)]:
        with pytest.raises(ValueError, match="codebook"):
            runtime._launch_for_codebooks(lib, torch.empty(shape))
    del lib.hybrid_launch_8x8_expert
    with pytest.raises(RuntimeError, match="rebuild"):
        runtime._launch_for_codebooks(lib, torch.empty(2, 512))


@GPU
def test_unmodified_v2_library(monkeypatch):
    original = os.environ.get("ARVQ_V2_BASELINE_LIB")
    if original is None:
        pytest.skip("set ARVQ_V2_BASELINE_LIB to independently built original library")
    f13, _, _ = fixture(1024, 6144)
    f2, _, _ = fixture(6144, 512, seed=19)
    layer = layer_from(6144, 512, f13, f2)
    shared, duplicate = list(layer._arvq_tensors), list(layer._arvq_tensors)
    for offset in (1, 7):
        shared[offset] = layer._arvq_tensors[offset][0].clone()
        duplicate[offset] = shared[offset].repeat(2, 1)
    x = torch.randn(4, 6144, device="cuda", dtype=torch.bfloat16) * 0.25
    ids = torch.tensor([[2, 1, 0]] * 4, device="cuda")
    weights = torch.tensor([[0.2, 0.3, 0.5]] * 4, device="cuda")

    def run(tensors):
        return runtime.arvq_mlp(
            x, weights, ids, layer._arvq_lookups, tensors, layer._arvq_alphas, 128
        )

    new_v2, new_v3 = run(shared), run(duplicate)
    monkeypatch.setenv("VLLM_ARVQ_KERNEL_LIB", original)
    monkeypatch.setattr(runtime, "_LIB", None)
    old_v2 = run(shared)
    assert torch.equal(old_v2, new_v2)
    assert torch.equal(old_v2, new_v3)
