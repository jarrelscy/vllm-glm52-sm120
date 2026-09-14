# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/model_executor/layers/quantization/nvfp4_arvq_route_pack.py"
spec = importlib.util.spec_from_file_location("route_pack_policy", SOURCE)
assert spec is not None and spec.loader is not None
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_ARVQ_FUSED_GATE_PACK", raising=False)
    assert not policy.eligible(None, None, 8, True)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "topk",
        "fallback",
        "dtype",
        "stride",
        "large",
        "empty",
        "wide_off",
        "cpu",
        "device",
    ],
)
def test_eligibility(monkeypatch, case):
    monkeypatch.setenv("VLLM_ARVQ_FUSED_GATE_PACK", "1")
    monkeypatch.setenv("VLLM_ARVQ_WIDE_HOT_PREFILL", "1")
    # Metadata stubs exercise CUDA eligibility without allocating on a GPU.
    x = SimpleNamespace(
        is_cuda=case != "cpu",
        device=torch.device("cuda", 0),
        dtype=torch.float16 if case == "dtype" else torch.bfloat16,
        ndim=2,
        shape=(2, 6144),
        is_contiguous=lambda: case != "stride",
    )
    routes = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda", 1 if case == "device" else 0),
        dtype=torch.int64,
        ndim=1,
        is_contiguous=lambda: True,
        numel=lambda: 1025 if case == "large" else 0 if case == "empty" else 32,
    )
    if case == "wide_off":
        monkeypatch.setenv("VLLM_ARVQ_WIDE_HOT_PREFILL", "0")
    assert policy.eligible(
        x, routes, 7 if case == "topk" else 8, case != "fallback"
    ) == (case == "valid")


@pytest.mark.skipif(
    os.environ.get("ARVQ_ROUTE_PACK_GPU_TEST") != "1",
    reason="explicit GPU lease required",
)
@pytest.mark.parametrize("slots", [1, 17, 65, 1024])
def test_gpu_packed_bytes_and_capture(slots):
    lib = ctypes.CDLL(os.environ["ARVQ_ROUTE_PACK_TEST_LIB"])
    old = lib.hybrid_pack_register_pairs
    old.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    new = lib.hybrid_pack_routes_top8
    new.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    torch.manual_seed(91426)
    x = torch.randn((2048, 6144), device="cuda").bfloat16()
    x[0, :8] = torch.tensor(
        [0.0, -0.0, 0.25, -0.25, 6.0, -6.0, 65504.0, -65504.0], device="cuda"
    )
    routes = (torch.arange(slots, device="cuda", dtype=torch.int64) * 8).contiguous()
    cold = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
    hot = torch.arange(slots, device="cuda", dtype=torch.int32) // 32
    q = torch.empty((slots, 4, 768), device="cuda", dtype=torch.int32)
    s = torch.empty((slots, 4, 384), device="cuda", dtype=torch.uint8)
    g = torch.zeros(((slots + 31) // 32, 7), device="cuda", dtype=torch.int32)
    refq, refs, refg = torch.empty_like(q), torch.empty_like(s), torch.zeros_like(g)

    def ptr(t):
        return ctypes.c_void_p(t.data_ptr())

    def baseline():
        xr = x[routes // 8].half()
        return old(
            *map(ptr, [xr, refq, refs, cold, hot, refg]),
            6144,
            slots,
            4,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )

    def candidate():
        return new(
            *map(ptr, [x, q, s, cold, hot, g, routes]),
            6144,
            slots,
            4,
            8,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )

    assert baseline() == candidate() == 0
    assert torch.equal(q, refq) and torch.equal(s, refs) and torch.equal(g, refg)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    q.zero_()
    s.zero_()
    graph.replay()
    assert torch.equal(q, refq) and torch.equal(s, refs)
