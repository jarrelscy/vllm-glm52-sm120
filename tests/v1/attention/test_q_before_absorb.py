# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate and meta-tensor collective/stride regression tests."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def helper():
    path = (
        Path(__file__).resolve().parents[3] / "vllm/v1/attention/ops/q_before_absorb.py"
    )
    tree = ast.parse(path.read_text())
    functions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    namespace = {"torch": torch, "os": os, "Any": Any, "logger": Mock()}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def fixtures():
    impl = type("FlashInferMLASparseSM120Impl", (), {})()
    impl.dcp_world_size = 4
    impl.supports_quant_query_input = False
    query = SimpleNamespace(
        shape=(2048, 16, 256),
        dtype=torch.bfloat16,
        is_cuda=True,
        stride=lambda: (4096, 256, 1),
    )
    weight = SimpleNamespace(
        shape=(16, 192, 512), dtype=torch.bfloat16, stride=lambda: (229376, 512, 1)
    )
    layer = SimpleNamespace(
        impl=impl,
        W_UK_T=weight,
        q_pad_num_heads=None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
    )
    return layer, query


def test_default_off_does_not_query_capture(helper, monkeypatch):
    monkeypatch.delenv("VLLM_GLM_Q_BEFORE_ABSORB", raising=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", Mock(side_effect=AssertionError)
    )
    layer, query = fixtures()
    assert not helper["q_before_absorb_eligible"](layer, query, False)


@pytest.mark.parametrize(
    "change",
    [
        "decode",
        "tail",
        "fp16",
        "cpu",
        "capture",
        "backend",
        "dcp",
        "padding",
        "fp4",
        "fp8",
        "quant_query",
        "q_stride",
        "w_stride",
    ],
)
def test_unsupported_paths_retain_original(helper, monkeypatch, change):
    monkeypatch.setenv("VLLM_GLM_Q_BEFORE_ABSORB", "1")
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: change == "capture"
    )
    layer, query = fixtures()
    if change == "decode":
        query.shape = (16, 16, 256)
    if change == "tail":
        query.shape = (2177, 16, 256)
    if change == "fp16":
        query.dtype = torch.float16
    if change == "cpu":
        query.is_cuda = False
    if change == "backend":
        layer.impl = SimpleNamespace()
    if change == "dcp":
        layer.impl.dcp_world_size = 2
    if change == "padding":
        layer.q_pad_num_heads = 32
    if change == "fp4":
        layer.is_aiter_triton_fp4_bmm_enabled = True
    if change == "fp8":
        layer.is_aiter_triton_fp8_bmm_enabled = True
    if change == "quant_query":
        layer.impl.supports_quant_query_input = True
    if change == "q_stride":
        query.stride = lambda: (5000, 256, 1)
    if change == "w_stride":
        layer.W_UK_T.stride = lambda: (98304, 512, 1)
    assert not helper["q_before_absorb_eligible"](layer, query, change == "quant_query")


@pytest.mark.parametrize("tokens", [2048, 4096])
def test_qualified_gate(helper, monkeypatch, tokens):
    monkeypatch.setenv("VLLM_GLM_Q_BEFORE_ABSORB", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layer, query = fixtures()
    query.shape = (tokens, 16, 256)
    assert helper["q_before_absorb_eligible"](layer, query, False)


@pytest.mark.parametrize("overlap", [False, True])
def test_collective_order_and_exact_bmm_strides(helper, monkeypatch, overlap):
    events: list[Any] = []

    def gather(tensor, dim):
        events.append(("gather", tuple(tensor.shape), dim))
        return torch.cat([tensor] * 4, dim=dim)

    group = SimpleNamespace(all_gather=gather)
    helper["get_dcp_group"] = lambda: group
    helper["dcp_comm_overlap_enabled"] = lambda: overlap
    helper["consume_pending_dcp_merge"] = lambda: events.append("consume")

    class Async:
        def __init__(self, actual_group, tensor, dim, flush_coalesce):
            assert actual_group is group and dim == 1 and flush_coalesce
            events.append("async_begin")
            self.tensor = tensor

        def wait_raw(self):
            events.append("async_wait")
            return torch.cat([self.tensor] * 4, dim=0)

    helper["AsyncAllGather"] = Async

    def precompute(metadata, tokens):
        events.append("precompute")
        return "indices"

    impl = SimpleNamespace(precompute_mqa_indices=precompute)
    query = torch.empty((2048, 16, 256), dtype=torch.bfloat16, device="meta")
    weight = torch.empty_strided(
        (16, 192, 512), (229376, 512, 1), dtype=torch.bfloat16, device="meta"
    )
    original = torch.bmm

    def bmm(q, w, out):
        assert q.shape == (16, 2048, 192) and q.stride() == (256, 4096, 1)
        assert w.shape == weight.shape and w.stride() == weight.stride()
        events.append("bmm")
        return original(q, w, out=out)

    monkeypatch.setattr(torch, "bmm", bmm)
    result, indices = helper["q_before_absorb"](query, weight, impl, None)
    assert result.shape == (2048, 64, 576)
    assert events.count("bmm") == 4
    if overlap:
        assert events[:3] == ["async_begin", "precompute", "async_wait"]
        assert indices == "indices"
    else:
        assert events[0] == "consume" and indices is None
    assert ("gather", (16, 192, 512), 0) in events
