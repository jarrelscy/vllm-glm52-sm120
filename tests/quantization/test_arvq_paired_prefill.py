# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing and dispatch proofs for exact paired/wide hot prefill."""

import pytest
import torch

from vllm.model_executor.layers.quantization import (
    nvfp4_arvq_paired_policy as policy,
)
from vllm.model_executor.layers.quantization import (
    nvfp4_arvq_paired_runtime as paired,
)
from vllm.model_executor.layers.quantization import (
    nvfp4_arvq_prefill as prefill,
)
from vllm.model_executor.layers.quantization import (
    nvfp4_arvq_wide_runtime as wide,
)


def test_partition_preserves_hot_only_destinations():
    routes = torch.arange(1024)
    hot = torch.cat((torch.arange(16).repeat_interleave(32), torch.full((512,), -1)))
    cold = torch.cat((torch.full((512,), -1), torch.zeros(512, dtype=torch.int64)))
    batches = policy.partition([(routes, 8)], 4096, hot, cold)
    assert len(batches) == 2
    selected, split, enabled = batches[0]
    assert enabled and split == 8
    assert torch.equal(selected, routes[:512])
    assert torch.all(cold[selected] < 0) and torch.all(hot[selected] >= 0)
    assert torch.equal(batches[1][0], routes[512:]) and not batches[1][2]
    assert torch.equal(torch.cat([b[0] for b in batches]).sort().values, routes)


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 129, 2049, 2177, 4097])
def test_partition_rejects_decode_and_original_tails(tokens):
    routes = torch.arange(1024)
    hot = torch.zeros(1024, dtype=torch.int64)
    cold = torch.full_like(hot, -1)
    batches = policy.partition([(routes, 16)], tokens, hot, cold)
    assert len(batches) == 1 and batches[0][1:] == (16, False)
    assert torch.equal(batches[0][0], routes)


@pytest.mark.parametrize("small_total", [False, True])
def test_partition_rejects_low_collision_and_small_total(small_total):
    routes = torch.arange(1024)
    hot = (
        torch.cat((torch.zeros(256, dtype=torch.int64), torch.arange(768) + 1))
        if small_total
        else torch.arange(1024) // 16
    )
    batches = policy.partition([(routes, 8)], 2048, hot, torch.full_like(routes, -1))
    assert len(batches) == 1 and not batches[0][2]
    assert torch.equal(batches[0][0], routes)


@pytest.mark.parametrize("tokens,mode", [(2048, 0), (4096, 2)])
def test_wide_mode_uses_whole_prefill_count(tokens, mode):
    assert wide.mode_for_tokens(tokens) == mode


@pytest.mark.parametrize("tokens", [1, 8, 32, 128, 256, 512, 1024, 2177, 8192])
def test_wide_rejects_route_chunk_counts(tokens):
    with pytest.raises(ValueError):
        wide.mode_for_tokens(tokens)


def test_unqualified_shared_mode_rejected_before_library_load():
    with pytest.raises(ValueError):
        wide.WideHot(1)


@pytest.mark.parametrize(
    "tokens,paired_enabled,wide_enabled,expected",
    [
        (2048, True, False, "paired"),
        (2048, True, True, 0),
        (4096, True, True, 2),
        (2177, True, True, None),
        (2048, False, True, None),
    ],
)
def test_actual_helper_preserves_scatter_and_selects_owner_batch(
    monkeypatch, tokens, paired_enabled, wide_enabled, expected
):
    monkeypatch.setenv("VLLM_ARVQ_COMPACT_PREFILL", "1")
    monkeypatch.setenv("VLLM_ARVQ_SORT_NATIVE_PREFILL", "0")
    factories = []

    def projection(rows, cold, hot, tensors, alpha, n, split, parts):
        return (rows[:, :1].float() + hot[:, None]).repeat(1, n)

    def factory(mode):
        factories.append(mode)
        return projection

    monkeypatch.setattr(paired, "PairedHot", lambda: factory("paired"))
    monkeypatch.setattr(wide, "WideHot", factory)
    args = (
        torch.arange(tokens).reshape(-1, 1).repeat(1, 16).half() / 10000,
        torch.arange(1, 9).float().repeat(tokens, 1),
        torch.arange(tokens * 8).reshape(tokens, 8) % 16,
        torch.stack((torch.full((16,), -1), torch.arange(16))).int(),
        [torch.empty(1, 2, 1)] * 12,
        [1.0, 1.0],
    )
    monkeypatch.delenv("VLLM_ARVQ_PAIRED_HOT_PREFILL", raising=False)
    monkeypatch.delenv("VLLM_ARVQ_WIDE_HOT_PREFILL", raising=False)
    original = prefill.grouped_cold_prefill(*args, projection=projection)
    assert factories == []
    monkeypatch.setenv("VLLM_ARVQ_PAIRED_HOT_PREFILL", str(int(paired_enabled)))
    monkeypatch.setenv("VLLM_ARVQ_WIDE_HOT_PREFILL", str(int(wide_enabled)))
    actual = prefill.grouped_cold_prefill(*args, projection=projection)
    assert torch.equal(original.view(torch.int16), actual.view(torch.int16))
    if expected is None:
        assert factories == []
    else:
        assert factories and set(factories) == {expected}


@pytest.mark.parametrize("mode", [0, 2])
def test_wide_dispatches_register_kernel_and_reuses_pair_descriptors(monkeypatch, mode):
    from types import SimpleNamespace

    calls = []

    def record(name):
        def invoke(*args):
            calls.append((name, args))
            return 0

        return invoke

    lib = SimpleNamespace(
        hybrid_pack_register_pairs=record("pack_pairs"),
        hybrid_pack=record("pack"),
        wide_launch_register=record("register"),
    )
    monkeypatch.setattr(wide, "library", lambda: lib)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    projection = wide.WideHot(mode)
    rows = torch.zeros((33, 128), dtype=torch.float16)
    hot = torch.zeros(33, dtype=torch.int32)
    cold = torch.full_like(hot, -1)
    tensors = [torch.zeros(1)] * 6
    result = projection(rows, cold, hot, tensors, 1.0, 64, 8, 2)
    groups = projection.groups
    projection(rows, cold, hot, tensors, 1.0, 64, 2, 1)
    assert result.shape == (33, 64)
    assert projection.groups is groups and groups.shape == (2, 7)
    assert [name for name, _ in calls] == ["pack_pairs", "register", "pack", "register"]
    assert calls[1][1][10:16] == (64, 128, 33, 8, 2, mode)
    assert calls[3][1][10:16] == (64, 128, 33, 2, 1, mode)
