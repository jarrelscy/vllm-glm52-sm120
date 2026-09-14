# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm/model_executor/layers/quantization/nvfp4_arvq_cold_scatter.py"
)
spec = importlib.util.spec_from_file_location("cold_scatter", PATH)
assert spec is not None and spec.loader is not None
scatter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scatter)


def metadata(shape, dtype):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device("cuda", 0),
        is_cuda=True,
        is_contiguous=lambda: True,
        data_ptr=lambda: 16,
        numel=lambda: shape[0],
    )


def test_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ARVQ_FUSED_COLD_SCATTER", raising=False)
    assert not scatter.batch_eligible(None, None, None)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "cpu",
        "device",
        "dtype",
        "stride",
        "width",
        "align",
        "empty",
        "route_count",
        "tokens",
    ],
)
def test_gate(monkeypatch, case):
    monkeypatch.setenv("VLLM_ARVQ_FUSED_COLD_SCATTER", "1")
    src = metadata((2048, 6144), torch.bfloat16)
    dst = metadata((4096, 6144), torch.float32)
    routes = metadata((32,), torch.int64)
    if case == "tokens":
        src.shape = (2177, 6144)
    if case == "cpu":
        src.is_cuda = False
    if case == "device":
        routes.device = torch.device("cuda", 1)
    if case == "dtype":
        src.dtype = torch.float16
    if case == "stride":
        routes.is_contiguous = lambda: False
    if case == "width":
        src.shape = (32, 512)
    if case == "align":
        dst.data_ptr = lambda: 4
    if case == "empty":
        routes.numel = lambda: 0
    if case == "route_count":
        routes.numel = lambda: 4097
    assert scatter.batch_eligible(src, routes, dst) == (case == "valid")


@pytest.mark.skipif(
    os.environ.get("ARVQ_COLD_SCATTER_GPU_TEST") != "1",
    reason="explicit GPU lease required",
)
@pytest.mark.parametrize("rows", [8, 32, 128, 1024])
def test_raw_bits_graph_no_scratch(rows, monkeypatch):
    monkeypatch.setenv("VLLM_ARVQ_FUSED_COLD_SCATTER", "1")
    torch.manual_seed(7731)
    routes = torch.randperm(2048, device="cuda")[:rows].contiguous()
    src = torch.randint(
        -(2**31), 2**31 - 1, (rows, 6144), device="cuda", dtype=torch.int32
    ).view(torch.float32)
    dst = torch.zeros((2048, 6144), device="cuda", dtype=torch.float32)
    ref = dst.clone()
    ref[routes] = src
    x = torch.empty((2048, 6144), device="cuda", dtype=torch.bfloat16)
    copy_rows = scatter.prepare(x, routes, dst)
    assert copy_rows is not None
    before = torch.accelerator.memory_allocated()
    copy_rows(src, routes)
    assert torch.accelerator.memory_allocated() == before
    assert torch.equal(dst.view(torch.int32), ref.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_copy = scatter.prepare(x, routes, dst)
        graph_copy(src, routes)
    dst.zero_()
    graph.replay()
    assert torch.equal(dst.view(torch.int32), ref.view(torch.int32))
