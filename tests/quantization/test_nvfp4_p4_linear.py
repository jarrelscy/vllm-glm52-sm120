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


@pytest.mark.parametrize("native_max", [4, 8, 16])
def test_runtime_dispatch_crosses_threshold(monkeypatch, native_max):
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as hybrid

    calls = []

    def projection(x, cold, hot, tensors, alpha, n, split, parts):
        assert cold.shape == hot.shape == (x.shape[0],)
        assert cold.eq(-1).all() and hot.eq(0).all()
        calls.append(("native", x.shape[0], split))
        return torch.ones(x.shape[0], n)

    monkeypatch.setattr(hybrid, "_projection", projection)

    def dequant(weight, scales, global_scale, n, k):
        calls.append(("prefill", n, k))
        return torch.ones(n, k, dtype=torch.bfloat16)

    monkeypatch.setattr(dense, "_dequantize", dequant)
    packed = torch.empty(1, 1, 1, 4, 32, dtype=torch.int32)
    scales = torch.empty(16, 1, dtype=torch.int32)
    routes = torch.tensor([[-1] * native_max, [0] * native_max], dtype=torch.int32)
    token_cases = (1, 2, 4, 5, 8, 9, 16, 17)
    expected_splits = {1: 8, 2: 2, 4: 2, 5: 1, 8: 1, 9: 4, 16: 4, 17: 4}
    for tokens in token_cases:
        out = dense.dense_p4(
            torch.ones(tokens, 64, dtype=torch.bfloat16),
            packed,
            scales,
            torch.ones(1, 1),
            routes,
            native_max,
        )
        assert out.shape == (tokens, 16)
        assert out.eq(1 if tokens <= native_max else 64).all()
    assert calls == [
        ("native", tokens, expected_splits[tokens])
        if tokens <= native_max
        else ("prefill", 16, 64)
        for tokens in token_cases
    ]


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


def test_paired_dispatch_is_opt_in_and_preserves_single_token(monkeypatch):
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as hybrid

    calls = []

    def projection(x, cold, hot, tensors, alpha, n, split, parts):
        calls.append(("native", len(x), split))
        return torch.ones(len(x), n)

    def paired(x, weight, scales, global_scale, n, split):
        calls.append(("paired", len(x), split))
        return torch.ones(len(x), n)

    monkeypatch.setattr(hybrid, "_projection", projection)
    monkeypatch.setattr(dense, "_paired_projection", paired)
    weights = torch.empty(1, 1, 1, 4, 32, dtype=torch.int32)
    scales = torch.empty(16, 1, dtype=torch.int32)
    routes = torch.tensor([[-1] * 16, [0] * 16], dtype=torch.int32)
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_NVFP4_P4_PAIRED", str(int(enabled)))
        for m in (1, 2, 3, 4, 5, 7, 8, 9):
            dense.dense_p4(
                torch.ones(m, 64), weights, scales, torch.ones(1), routes, 16
            )
    assert calls[:8] == [
        ("native", m, s)
        for m, s in ((1, 8), (2, 2), (3, 2), (4, 2), (5, 1), (7, 1), (8, 1), (9, 4))
    ]
    assert calls[8:] == [
        ("native", 1, 8),
        ("paired", 2, 8),
        ("paired", 3, 4),
        ("paired", 4, 4),
        ("paired", 5, 2),
        ("paired", 7, 2),
        ("paired", 8, 2),
        ("paired", 9, 2),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires SM120 CUDA libraries"
)
@pytest.mark.parametrize("tokens", [2, 3, 4, 5, 7, 8, 9, 15, 16])
def test_paired_cuda_matches_native_and_replays_changed_activations(tokens):
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as hybrid

    torch.manual_seed(971 + tokens)
    # A partial final CTA and odd token counts exercise both independent guards.
    n, k = 80, 512
    cpu_weights = dense.quantize_weight(torch.randn(n, k).bfloat16())
    weight, scales, global_scale = [t.cuda() for t in cpu_weights]
    decoded = decode_cpu(*cpu_weights)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
    cold = torch.full((tokens,), -1, device="cuda", dtype=torch.int32)
    hot = torch.zeros_like(cold)
    tensors = [weight, weight, scales, weight, scales, global_scale]
    split = 8 if tokens <= 2 else 4 if tokens <= 4 else 2
    dense._paired_projection(x, weight, scales, global_scale, n, split)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dense._paired_projection(x, weight, scales, global_scale, n, split)
    for amplitude in (0.3, 1.7):
        x.normal_().mul_(amplitude)
        graph.replay()
        expected = hybrid._projection(x, cold, hot, tensors, 0.0, n, split, 1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
        # Independent ordinary GEMM checks both token outputs and all row scales.
        residual = x.cpu().float()
        activation = torch.zeros_like(residual)
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        for plane in range(4):
            maxima = residual.reshape(tokens, -1, 16).abs().amax(-1)
            exponents = torch.ceil(torch.log2((maxima / 6).clamp_min(2**-20))).clamp(
                -6, 8
            )
            scale = torch.exp2(exponents).repeat_interleave(16, -1)
            norm = residual / scale
            code = sum(
                (norm.abs() > t).long() for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5)
            )
            quantized = levels[code] * norm.sign() * scale
            activation += quantized / (16**plane)
            residual = (residual - quantized) * 16
        oracle = activation @ decoded.T
        assert (actual.cpu() - oracle).norm() / oracle.norm() < 3e-6


def test_paired_still_dequantizes_above_native_limit(monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_P4_PAIRED", "1")
    calls = []

    def dequant(weight, scales, global_scale, n, k):
        calls.append((n, k))
        return torch.ones(n, k)

    monkeypatch.setattr(dense, "_dequantize", dequant)
    out = dense.dense_p4(
        torch.ones(17, 64),
        torch.empty(1, 1, 1, 4, 32, dtype=torch.int32),
        torch.empty(16, 1, dtype=torch.int32),
        torch.ones(1),
        torch.empty(2, 16, dtype=torch.int32),
        16,
    )
    assert calls == [(16, 64)]
    assert out.shape == (17, 16) and out.eq(64).all()
