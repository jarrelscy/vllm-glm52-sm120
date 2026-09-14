# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialized ARVQ TP slicing and complete graph-captured routed MLP."""

import pytest
import torch

from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
    ArvqExpertsMoEMethod,
    NvFp4ArvqHybridConfig,
    arvq_mlp,
)


def _method(rank=0, tp=4):
    method = object.__new__(ArvqExpertsMoEMethod)
    method._tp = tp
    method._tpr = rank
    method.n_cold = 1
    method.n_nvfp4 = 1
    method._chunk_tokens = 2
    return method


@pytest.mark.parametrize("rank", range(4))
def test_serialized_tensor_parallel_loaders(rank):
    method = _method(rank)
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 128, 128, torch.bfloat16)
    for name, param in layer.named_parameters():
        shape = list(param.shape)
        if name.startswith("arvq_w13_") and name.endswith(("packed", "scales")):
            shape[1] *= 4
        if name.startswith("arvq_w2_") and name.endswith(("packed", "scales")):
            shape[2] *= 4
        if name.startswith("nvfp4_w13_") and not name.endswith("scale2"):
            shape[1] *= 4
        if name.startswith("nvfp4_w2_") and not name.endswith("scale2"):
            shape[2] *= 4
        source = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        source = source.to(param.dtype)
        param.weight_loader(param, source)
        if name.startswith("arvq_w13_") and name.endswith(("packed", "scales")):
            width = param.shape[1] // 2
            gate = source[:, rank * width : (rank + 1) * width]
            up = source[:, (4 + rank) * width : (5 + rank) * width]
            assert torch.equal(param, torch.cat((gate, up), dim=1))
        if name.startswith("arvq_w2_") and name.endswith(("packed", "scales")):
            width = param.shape[2]
            assert torch.equal(param, source[:, :, rank * width : (rank + 1) * width])


def test_reject_incompatible_checkpoint_marker():
    with pytest.raises(ValueError, match="Unsupported ARVQ"):
        NvFp4ArvqHybridConfig.from_config({"arvq": {"activation_planes": 1}})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM120 GPU")
def test_custom_mlp_graph_chunk_and_compile():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 GPU")
    method = _method(tp=1)
    torch.manual_seed(41)
    with torch.device("cuda"):
        layer = torch.nn.Module()
        method.create_weights(layer, 2, 128, 128, torch.bfloat16)
    for name, param in layer.named_parameters():
        if name == "hyb_kind":
            param.data.copy_(torch.tensor([0, 2], device="cuda", dtype=torch.int8))
        elif name.endswith(("global", "scale2")):
            param.data.fill_(0.25)
        elif "scales" in name or "bscale" in name:
            param.data.copy_(
                torch.randint(40, 65, param.shape, device="cuda", dtype=torch.uint8)
            )
        elif param.dtype == torch.uint32:
            value = torch.randint(
                -(2**31), 2**31, param.shape, device="cuda", dtype=torch.int32
            ).view(torch.uint32)
            param.data.copy_(value)
        else:
            param.data.copy_(
                torch.randint(0, 256, param.shape, device="cuda", dtype=torch.uint8)
            )
    method.process_weights_after_loading(layer)
    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([[0.3, 0.7]] * 5, device="cuda")
    ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], device="cuda")

    def run(x, weights, ids):
        return arvq_mlp(
            x,
            weights,
            ids,
            layer._arvq_lookups,
            layer._arvq_tensors,
            layer._arvq_alphas,
            2,
        )

    expected = run(x, weights, ids)
    assert torch.isfinite(expected).all()
    separate = torch.cat(
        [run(x[i : i + 1], weights[i : i + 1], ids[i : i + 1]) for i in range(len(x))]
    )
    assert torch.equal(expected, separate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(x, weights, ids)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)
    traced = torch.compile(run, backend="eager", fullgraph=True)
    assert torch.equal(expected, traced(x, weights, ids))


@pytest.mark.parametrize("draft", [False, True])
def test_real_model_loader_remaps_serialized_arvq(monkeypatch, draft):
    from types import SimpleNamespace

    from vllm.model_executor.models import deepseek_mtp, deepseek_v2

    module = deepseek_mtp if draft else deepseek_v2
    owner = module.DeepSeekMTP if draft else module.DeepseekV2Model
    layer_id = 78 if draft else 3
    prefix = f"model.layers.{layer_id}.mlp.experts."
    params = {}
    inputs = []
    for projection in ("w13", "w2"):
        for suffix in ("packed", "scales", "codebooks", "global"):
            name = f"arvq_{projection}_{suffix}"
            target = prefix + "routed_experts." + name
            params[target] = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            inputs.append((prefix + name, torch.tensor([3.0])))
    model = SimpleNamespace(
        use_mha=False,
        config=SimpleNamespace(
            n_routed_experts=256, n_shared_experts=1, num_hidden_layers=78
        ),
        num_redundant_experts=0,
        model=SimpleNamespace(mtp_start_layer_idx=78, num_mtp_layers=1),
        named_parameters=lambda: iter(params.items()),
        _rewrite_spec_layer_name=lambda index, name: name,
    )
    monkeypatch.setattr(
        module, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(module, "get_pp_missing_layer_names", lambda model: set())
    if hasattr(module, "is_pp_missing_parameter"):
        monkeypatch.setattr(module, "is_pp_missing_parameter", lambda *args: False)
    monkeypatch.setattr(
        module,
        "get_spec_layer_idx_from_weight_name",
        lambda config, name: layer_id if draft else None,
    )
    loaded = owner.load_weights(model, inputs)
    assert loaded == set(params)
    assert all(torch.equal(p, torch.tensor([3.0])) for p in params.values())
