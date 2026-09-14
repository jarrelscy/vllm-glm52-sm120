# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU safety coverage for the opt-in grouped-prefill dispatch boundary."""

import sys
from types import ModuleType

import pytest
import torch

from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as hybrid


def test_disabled_small_and_overbudget_never_query_cuda(monkeypatch):
    def unexpected_capture_query():
        raise AssertionError("Ineligible calls must not query CUDA capture state")

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", unexpected_capture_query
    )
    monkeypatch.delenv("VLLM_ARVQ_GROUPED_PREFILL", raising=False)
    assert not hybrid._grouped_prefill_enabled(4096, 8, 6144)
    monkeypatch.setenv("VLLM_ARVQ_GROUPED_PREFILL", "1")
    assert not hybrid._grouped_prefill_enabled(2047, 8, 6144)
    assert not hybrid._grouped_prefill_enabled(8192, 8, 6144)


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("mode", ["1", "toggle"])
def test_actual_custom_op_dispatch_respects_capture(monkeypatch, capturing, mode):
    monkeypatch.setenv("VLLM_ARVQ_GROUPED_PREFILL", mode)
    monkeypatch.setattr(hybrid.Path, "exists", lambda self: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    module_name = "vllm.model_executor.layers.quantization.nvfp4_arvq_prefill"
    module = ModuleType(module_name)
    calls = []

    def grouped(x, weights, ids, lookups, tensors, alphas, **kwargs):
        calls.append("grouped")
        assert kwargs["projection"] is native
        assert kwargs["chunk_tokens"] == 128
        return x + 2

    def native(x, cold, hot, tensors, alpha, n, split, parts):
        calls.append("native")
        return torch.zeros((x.shape[0], n), dtype=torch.float32)

    module.grouped_cold_prefill = grouped
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(hybrid, "_projection", native)
    x = torch.ones((2048, 16), dtype=torch.bfloat16)
    ids = torch.zeros((2048, 1), dtype=torch.int32)
    weights = torch.ones((2048, 1))
    lookups = torch.tensor([[0], [-1]], dtype=torch.int32)
    # Only the native gate/up output width is read by this dispatcher.
    tensors = [torch.empty(1, 2, 1)] * 12
    result = hybrid.arvq_mlp(x, weights, ids, lookups, tensors, [1.0, 1.0], 128)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.eq(0 if capturing else 3).all()
    assert calls == (["native"] * 32 if capturing else ["grouped"])


@pytest.mark.parametrize("present", [False, True])
def test_diagnostic_marker_and_normal_mode(monkeypatch, present):
    reads = []

    def exists(path):
        reads.append(str(path))
        return present

    monkeypatch.setattr(hybrid.Path, "exists", exists)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("VLLM_ARVQ_GROUPED_PREFILL", "toggle")
    assert hybrid._grouped_prefill_enabled(2048, 8, 6144) is present
    assert reads == ["/dev/shm/vllm_arvq_grouped_prefill_on"]
    reads.clear()
    assert not hybrid._grouped_prefill_enabled(2047, 8, 6144)
    assert not hybrid._grouped_prefill_enabled(8192, 8, 6144)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not hybrid._grouped_prefill_enabled(2048, 8, 6144)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("VLLM_ARVQ_GROUPED_PREFILL", "1")
    assert hybrid._grouped_prefill_enabled(2048, 8, 6144)
    assert reads == []
