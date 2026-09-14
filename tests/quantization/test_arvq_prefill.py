# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped route ordering and native ARVQ reconstruction."""

import pytest
import torch

from vllm.model_executor.layers.quantization import nvfp4_arvq_prefill as prefill


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "scenario", ["mixed", "all_hot", "all_cold", "low_count", "zero"]
)
def test_grouped_preserves_mixed_route_order(monkeypatch, compact, scenario):
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "1" if compact else "0")
    torch.manual_seed(7)
    hidden, intermediate = 128, 128
    w13 = torch.randn(3, intermediate * 2, hidden).half() * 0.01
    w2 = torch.randn(3, hidden, intermediate).half() * 0.01
    x = torch.randn(4, hidden).half()
    ids = torch.tensor([[0, 2], [1, 0], [2, 0], [2, 0]])
    if scenario == "all_hot":
        ids.fill_(2)
    elif scenario == "all_cold":
        ids.fill_(0)
    elif scenario == "zero":
        ids.fill_(3)
    routing = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
    lookups = torch.tensor([[0, 1, -1, -1], [-1, -1, 0, -1]], dtype=torch.int32)
    tensors = []
    for n, k in ((256, 128), (128, 128)):
        tensors.extend(
            [
                torch.zeros(2 * (n // 16) * (k // 64) * 60 + 1, dtype=torch.int32),
                torch.zeros(384, dtype=torch.int32),
                torch.zeros(2, n // 16, k // 128, 16, dtype=torch.uint8),
                torch.zeros(1, n // 16, k // 64, 4, 32, dtype=torch.int32),
                torch.zeros(1, n, k // 64, dtype=torch.int32),
                torch.ones(1, 2 if n == 256 else 1),
            ]
        )
    decoded = []

    def decode(packed, codebooks, scales, alpha, n, k):
        decoded.append((n, k))
        return w13[0] if n == 256 else w2[0]

    monkeypatch.setattr(prefill, "dequantize_cold", decode)
    original_mm = torch.mm

    def mm(a, b, *, out_dtype=None):
        # CPU stand-in for CUDA's FP16 inputs / FP32 output GEMM.
        return original_mm(a.float(), b.float()).to(out_dtype or a.dtype)

    monkeypatch.setattr(torch, "mm", mm)

    def projection(rows, cold, hot, tensors, alpha, n, split, parts):
        output = torch.zeros(rows.shape[0], n)
        weights = w13 if n == 256 else w2
        for slot in range(rows.shape[0]):
            expert = int(cold[slot]) if cold[slot] >= 0 else 2 + int(hot[slot])
            if cold[slot] >= 0 or hot[slot] >= 0:
                output[slot] = rows[slot].float() @ weights[expert].float().T
        return output

    actual = prefill.grouped_cold_prefill(
        x,
        routing,
        ids,
        lookups,
        tensors,
        [1.0, 1.0],
        projection=projection,
        min_expert_tokens=99 if scenario == "low_count" else 3,
        chunk_tokens=2,
    )
    expected_routes = torch.empty(4, 2, hidden)
    for token in range(4):
        for slot in range(2):
            expert = int(ids[token, slot])
            if expert == 3:
                expected_routes[token, slot] = 0
                continue
            h13 = (x[token].float() @ w13[expert].float().T).half()
            gate, up = h13.chunk(2)
            act = (torch.nn.functional.silu(gate) * up).half()
            expected_routes[token, slot] = act.float() @ w2[expert].float().T
    expected = (expected_routes * routing[:, :, None]).sum(1).half()
    # CPU batched GEMM and per-route matvec accumulate in different orders;
    # a few intermediate FP16 values can land on opposite rounding midpoints.
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-3)
    assert decoded == (
        [(256, 128), (128, 128)] if scenario in ("mixed", "all_cold") else []
    )


def test_compact_retains_tiny_original_tail_split(monkeypatch):
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "1")
    calls = []

    def projection(rows, cold, hot, tensors, alpha, n, split, parts):
        calls.append((rows.shape[0], split, parts))
        return torch.zeros(rows.shape[0], n)

    result = prefill.grouped_cold_prefill(
        torch.ones(131, 16).half(),
        torch.ones(131, 8),
        torch.zeros(131, 8, dtype=torch.long),
        torch.tensor([[-1], [0]], dtype=torch.int32),
        [torch.empty(1, 2, 1)] * 12,
        [1.0, 1.0],
        projection=projection,
    )
    assert calls == [(1024, 8, 2), (1024, 2, 1), (24, 16, 2), (24, 2, 1)]
    assert result.count_nonzero() == 0


def test_compact_diagnostic_marker(monkeypatch):
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "toggle")
    monkeypatch.setattr(prefill.Path, "exists", lambda path: False)
    assert not prefill._compact_prefill_enabled()
    monkeypatch.setattr(prefill.Path, "exists", lambda path: True)
    assert prefill._compact_prefill_enabled()

    def unexpected(path):
        raise AssertionError("Normal modes must not inspect marker files")

    monkeypatch.setattr(prefill.Path, "exists", unexpected)
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "1")
    assert prefill._compact_prefill_enabled()
    monkeypatch.delenv("VLLM_ARVQ_COMPACT_PREFILL")
    assert not prefill._compact_prefill_enabled()


@pytest.mark.parametrize("tokens", [32, 64, 131])
def test_sorted_routes_preserve_scatter_and_split(monkeypatch, tokens):
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "1")
    calls = []

    def projection(rows, cold, hot, tensors, alpha, n, split, parts):
        calls.append((rows.shape[0], split, parts, hot.clone()))
        if parts == 2:
            return rows[:, :1].repeat(1, n)
        return (rows[:, :1].float() + hot[:, None]).repeat(1, n)

    args = (
        torch.arange(tokens).reshape(-1, 1).repeat(1, 16).half() / 100,
        torch.arange(1, 9).float().repeat(tokens, 1),
        torch.arange(tokens * 8).reshape(tokens, 8) % 7,
        torch.stack((torch.full((7,), -1), torch.arange(7))).int(),
        [torch.empty(1, 2, 1)] * 12,
        [1.0, 1.0],
    )
    monkeypatch.setenv("VLLM_ARVQ_SORT_NATIVE_PREFILL", "0")
    original = prefill.grouped_cold_prefill(*args, projection=projection)
    original_calls = calls[:]
    calls.clear()
    monkeypatch.setenv("VLLM_ARVQ_SORT_NATIVE_PREFILL", "1")
    sorted_result = prefill.grouped_cold_prefill(*args, projection=projection)
    assert torch.equal(original.view(torch.int16), sorted_result.view(torch.int16))
    assert [c[:3] for c in calls] == [c[:3] for c in original_calls]
    if tokens >= 64:
        assert bool((calls[0][3][1:] >= calls[0][3][:-1]).all())
    else:
        assert torch.equal(calls[0][3], original_calls[0][3])


def test_sorted_diagnostic_marker(monkeypatch):
    monkeypatch.setenv("VLLM_ARVQ_SORT_NATIVE_PREFILL", "toggle")
    monkeypatch.setattr(prefill.Path, "exists", lambda path: False)
    assert not prefill._sort_native_prefill_enabled()
    monkeypatch.setattr(prefill.Path, "exists", lambda path: True)
    assert prefill._sort_native_prefill_enabled()

    def unexpected(path):
        raise AssertionError("Normal modes must not inspect marker files")

    monkeypatch.setattr(prefill.Path, "exists", unexpected)
    monkeypatch.setenv("VLLM_ARVQ_SORT_NATIVE_PREFILL", "1")
    assert prefill._sort_native_prefill_enabled()
    monkeypatch.delenv("VLLM_ARVQ_SORT_NATIVE_PREFILL")
    assert not prefill._sort_native_prefill_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM120 GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequant_matches_independent_natural_codes(dtype):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 GPU")
    torch.manual_seed(9)
    n, k = 32, 128
    position = torch.arange(n * k // 8).reshape(n, k // 8)
    a, b = position * 17 % 256, position * 31 % 128
    natural = (a | (b << 8)).reshape(n // 16, 16, k // 64, 8)
    fragments = torch.stack(
        [
            natural[
                :, 8 * (j % 2) : 8 * (j % 2) + 8, :, 4 * (j // 2) : 4 * (j // 2) + 4
            ]
            .permute(0, 2, 1, 3)
            .reshape(-1, 32)
            for j in range(4)
        ],
        1,
    ).reshape(-1, 128)
    bit = torch.arange(128) * 15
    native = torch.zeros(fragments.shape[0], 61, dtype=torch.int64)
    indices = (bit // 32).expand(fragments.shape[0], -1)
    shifted = fragments << (bit % 32)
    native.scatter_add_(1, indices, shifted & 0xFFFFFFFF)
    native.scatter_add_(1, indices + 1, shifted >> 32)
    packed = torch.cat(
        (native[:, :60].reshape(-1).int(), torch.zeros(1, dtype=torch.int32))
    )
    cb = torch.randint(-(1 << 31), (1 << 31) - 1, (384,), dtype=torch.int32)
    scales = torch.tensor([0, 1, 8, 0x38, 0x7E], dtype=torch.uint8).repeat(7)[:n]
    scales = scales.reshape(n // 16, 1, 16)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    values = levels[(cb.long()[:, None] >> (4 * torch.arange(8))) & 15]
    expected = (values[a] + values[256 + b]).reshape(n, k)
    expected *= scales.reshape(n, 1).view(torch.float8_e4m3fn).float()
    expected *= 0.0137
    args = tuple(t.cuda() for t in (packed, cb, scales))
    actual = prefill.dequantize_cold(*args, 0.0137, n, k, dtype=dtype)
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prefill.dequantize_cold(*args, 0.0137, n, k, dtype=dtype)
    graph.replay()
    torch.testing.assert_close(captured, actual, atol=0, rtol=0)
