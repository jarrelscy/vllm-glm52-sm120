# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for experimental dense P4 loading and opaque dispatch."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor.layers.quantization import nvfp4_p4_linear as dense


def decode_cpu(packed, scales, global_scale):
    _, tiles, groups, _, _ = packed.shape
    words = torch.empty((tiles, 16, groups, 8), dtype=torch.int32)
    for j in range(4):
        words[:, 8 * (j % 2) : 8 * (j % 2) + 8, :, 4 * (j // 2) : 4 * (j // 2) + 4] = (
            packed[0, :, :, j].reshape(tiles, groups, 8, 4).permute(0, 2, 1, 3)
        )
    shifts = torch.arange(8) * 4
    codes = ((words[..., None].long() >> shifts) & 15).reshape(tiles * 16, groups * 64)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    block_scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return levels[codes] * block_scales * global_scale


@pytest.mark.parametrize("zero", [False, True])
def test_chunked_quantization_layout_and_storage(zero):
    torch.manual_seed(5)
    weight = torch.randn(272, 128).bfloat16()
    if zero:
        weight.zero_()
    packed = dense.quantize_weight(weight)
    decoded = decode_cpu(*packed)
    assert torch.isfinite(decoded).all()
    assert sum(t.nbytes for t in packed) == weight.numel() * 9 // 16 + 4
    if zero:
        assert decoded.count_nonzero() == 0
    else:
        assert (decoded - weight.float()).norm() / weight.float().norm() < 0.13
    # Chunking must not change the single-matrix scale or native row ordering.
    assert packed[0].shape == (1, 17, 2, 4, 32)


def test_load_replaces_bf16_and_is_idempotent(monkeypatch):
    from vllm.model_executor import parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setenv("VLLM_NVFP4_P4_MAX_TOKENS", "4")
    layer = torch.nn.Module()
    method = dense.NvFp4P4LinearMethod()
    method.create_weights(
        layer, 64, [16], 64, 16, torch.bfloat16, weight_loader=lambda *args: None
    )
    layer.weight.data.normal_()
    method.process_weights_after_loading(layer)
    first = layer.weight
    method.process_weights_after_loading(layer)
    assert layer.weight is first
    assert all(p.dtype != torch.bfloat16 for p in layer.parameters())
    assert layer.p4_route_ids.shape == (2, 4)
    assert layer.p4_route_ids[0].eq(-1).all()
    assert layer.p4_route_ids[1].eq(0).all()


def test_opt_in_only_attention_output(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_NVFP4_P4_O_PROJ", raising=False)
    assert not dense.matches("model.layers.3.self_attn.o_proj")
    monkeypatch.setenv("VLLM_ENABLE_NVFP4_P4_O_PROJ", "1")
    assert dense.matches("model.layers.3.self_attn.o_proj")
    for suffix in (
        "self_attn.q_b_proj",
        "self_attn.fused_qkv_a_proj",
        "mlp.shared_experts.down_proj",
        "mlp.experts.o_proj",
    ):
        assert not dense.matches("model.layers.3." + suffix)


def test_fake_preserves_leading_dimensions():
    with FakeTensorMode():
        x = torch.empty(2, 3, 64, dtype=torch.bfloat16)
        packed = torch.empty(1, 1, 1, 4, 32, dtype=torch.int32)
        y = dense.dense_p4(
            x,
            packed,
            torch.empty(16, 1, dtype=torch.int32),
            torch.ones(1, 1),
            torch.empty(2, 4, dtype=torch.int32),
            4,
        )
        assert y.shape == (2, 3, 16)
        assert y.dtype == x.dtype


def test_runtime_dispatch_crosses_threshold(monkeypatch):
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as hybrid

    calls = []

    def projection(x, cold, hot, tensors, alpha, n, split, parts):
        calls.append(("native", x.shape[0], split))
        return torch.ones(x.shape[0], n)

    monkeypatch.setattr(hybrid, "_projection", projection)

    def dequant(weight, scales, global_scale, n, k):
        calls.append(("prefill", n, k))
        return torch.ones(n, k, dtype=torch.bfloat16)

    monkeypatch.setattr(dense, "_dequantize", dequant)
    packed = torch.empty(1, 1, 1, 4, 32, dtype=torch.int32)
    scales = torch.empty(16, 1, dtype=torch.int32)
    routes = torch.tensor([[-1] * 4, [0] * 4], dtype=torch.int32)
    for tokens in (1, 4, 5):
        out = dense.dense_p4(
            torch.ones(tokens, 64, dtype=torch.bfloat16),
            packed,
            scales,
            torch.ones(1, 1),
            routes,
            4,
        )
        assert out.shape == (tokens, 16)
        assert out.eq(1 if tokens <= 4 else 64).all()
    assert calls == [("native", 1, 8), ("native", 4, 2), ("prefill", 16, 64)]


def test_arvq_hook_precedes_optional_fp8(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
        NvFp4AqlmHybridConfig,
    )
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
        NvFp4ArvqHybridConfig,
    )

    config = object.__new__(NvFp4ArvqHybridConfig)
    config.aqlm_layer_books = {77: {}}
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    sentinel = object()
    monkeypatch.setattr(config, "_aqlm_layer_idx", lambda prefix: None)
    monkeypatch.setattr(
        NvFp4AqlmHybridConfig, "get_quant_method", lambda *args: sentinel
    )
    monkeypatch.setenv("VLLM_ENABLE_NVFP4_P4_O_PROJ", "1")
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.o_proj"),
        dense.NvFp4P4LinearMethod,
    )
    assert isinstance(
        config.get_quant_method(layer, "model.layers.77.self_attn.o_proj"),
        dense.NvFp4P4LinearMethod,
    )
    for prefix in (
        "model.layers.78.self_attn.o_proj",
        "model.layers.78.mtp_block.self_attn.o_proj",
        "model.layers.3.mtp_block.self_attn.o_proj",
    ):
        assert config.get_quant_method(layer, prefix) is sentinel
    assert config.get_quant_method(layer, "model.self_attn.q_b_proj") is sentinel
    monkeypatch.delenv("VLLM_ENABLE_NVFP4_P4_O_PROJ")
    assert config.get_quant_method(layer, "model.self_attn.o_proj") is sentinel
