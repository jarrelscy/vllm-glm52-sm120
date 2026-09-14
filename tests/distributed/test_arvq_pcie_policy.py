# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for independent fused-decode and copy-engine prefill routing."""

from types import SimpleNamespace

import torch

from vllm.distributed.device_communicators import b12x_pcie_all_reduce as adapter
from vllm.model_executor.layers.pcie_fused_ar_rms import (
    _run_b12x_fused,
    defer_mlp_all_reduce,
)


def test_explicit_zero_keeps_fused_channel(monkeypatch):
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "0")
    monkeypatch.setenv("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "12288")
    monkeypatch.setattr(
        adapter,
        "_load_b12x_recommended_max_bytes",
        lambda: lambda *args, **kwargs: 999999,
    )
    assert adapter._oneshot_limits(4) == (0, 12288, 12288)


def test_dma_size_gate_and_fp32_capacity(monkeypatch):
    monkeypatch.setenv("VLLM_PCIE_DMA_MIN_BYTES", "24MB")
    assert adapter._dma_min_bytes() == 24 * 1024**2
    monkeypatch.setenv("VLLM_PCIE_DMA_MIN_BYTES", "off")
    assert adapter._dma_min_bytes() is None
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, get_hidden_size=lambda: 6144
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    capacities = adapter._dma_capacity_plan()
    assert capacities == {torch.bfloat16: 4096 * 6144, torch.float32: 4096 * 6144}


def test_fused_warmup_preplans_and_runs_without_cuda(monkeypatch):
    calls = []
    runtime = SimpleNamespace(
        for_stream=lambda stream: SimpleNamespace(should_allreduce=lambda x: True),
        prepare_graph_fused_add_rms_norm=lambda *args, **kwargs: calls.append("plan"),
        all_reduce_fused_add_rms_norm=lambda *args, **kwargs: calls.append("run"),
    )
    comm = SimpleNamespace(
        supports_fused_add_rms_norm=lambda: True,
        fused_max_bytes=12288,
        _runtime=runtime,
        _runtime_stream=lambda: None,
        _is_capturing=True,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.empty(1, 6144, dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    weight = torch.empty(6144, dtype=x.dtype)
    out = torch.empty_like(x)
    assert _run_b12x_fused(comm, x, residual, weight, 1e-6, out)
    assert calls == ["plan", "run"]
    calls.clear()
    larger = torch.empty(2, 6144, dtype=torch.bfloat16)
    assert not _run_b12x_fused(
        comm,
        larger,
        torch.empty_like(larger),
        weight,
        1e-6,
        torch.empty_like(larger),
    )
    assert calls == []
    comm.fused_max_bytes = 0
    assert not _run_b12x_fused(comm, x, residual, weight, 1e-6, out)
    assert calls == []


def test_moe_defers_only_combined_late_reduction():
    config = SimpleNamespace(
        is_sequence_parallel=False, skip_final_all_reduce=False, tp_size=4, ep_size=1
    )
    experts = SimpleNamespace(
        moe_config=config, _fused_output_is_reduced=False, router=None
    )
    mlp = SimpleNamespace(experts=experts)
    assert defer_mlp_all_reduce(mlp)
    assert config.skip_final_all_reduce
    assert not defer_mlp_all_reduce(mlp)
    config.skip_final_all_reduce = False
    experts._fused_output_is_reduced = True
    assert not defer_mlp_all_reduce(mlp)
    assert not config.skip_final_all_reduce


def test_large_shape_fallback_is_compile_visible(monkeypatch):
    from vllm.model_executor.layers import pcie_fused_ar_rms as fusion

    monkeypatch.setenv("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "12288")
    monkeypatch.setattr(fusion, "tensor_model_parallel_all_reduce", lambda x: x * 4)

    class Norm:
        def __call__(self, x, residual):
            summed = x + residual
            return summed * 0.5, summed

    def run(x, residual):
        return fusion.fused_ar_rms_norm(x, residual, Norm())

    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    x = torch.randn(4, 6144, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    compiled = torch.compile(run, backend=backend, fullgraph=True)
    output, updated = compiled(x, residual)
    expected = x * 4 + residual
    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(output, expected * 0.5)
    assert len(graphs) == 1
    assert not any(
        "glm_pcie_fused_ar_rms" in str(node.target) for node in graphs[0].graph.nodes
    )


def test_model_initializes_aux_state_before_fusion_hook():
    # Execute the real constructor body with tiny CPU module factories. Avoid
    # distributed model initialization and the compile decorator, while keeping
    # constructor ordering and the hook invocation under test.
    import ast
    from pathlib import Path

    import vllm.model_executor.models.deepseek_v2 as model

    tree = ast.parse(Path(model.__file__).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2Model"
    )
    constructor = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = []

    def hook(self):
        assert self.aux_hidden_state_layers == ()
        assert self.pcie_fuse_final_norm is False
        calls.append(self)

    cls.decorator_list = []
    cls.bases = [ast.parse("torch.nn.Module", mode="eval").body]
    cls.body = [constructor]
    namespace = dict(vars(model))
    namespace.update(
        VocabParallelEmbedding=lambda *args, **kwargs: torch.nn.Identity(),
        RMSNorm=lambda *args, **kwargs: torch.nn.Identity(),
        make_layers=lambda *args, **kwargs: (0, 0, torch.nn.ModuleList()),
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        pcie_fused_ar_rms_enabled=lambda: True,
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), model.__file__, "exec"),
        namespace,
    )
    tiny_model = namespace["DeepseekV2Model"]
    tiny_model._enable_pcie_fused_ar_rms = hook
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                vocab_size=16,
                num_hidden_layers=0,
                rms_norm_eps=1e-6,
                model_type="deepseek",
            )
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    instance = tiny_model(vllm_config=config)
    assert calls == [instance]


def test_single_row_keeps_fused_custom_op(monkeypatch):
    from vllm.model_executor.layers import pcie_fused_ar_rms as fusion

    monkeypatch.setenv("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "12288")
    calls = []

    def fused(x, residual, weight, epsilon):
        calls.append(x.shape)
        residual.add_(x * 4)
        return residual * 0.5

    monkeypatch.setattr(torch.ops.vllm, "glm_pcie_fused_ar_rms", fused)
    x = torch.ones(1, 6144, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    norm = SimpleNamespace(weight=torch.ones(6144), variance_epsilon=1e-6)
    out, updated = fusion.fused_ar_rms_norm(x, residual, norm)
    assert calls == [x.shape]
    assert updated is residual
    torch.testing.assert_close(out, torch.full_like(x, 2.5))
    torch.testing.assert_close(residual, torch.full_like(x, 5))
    torch.testing.assert_close(x, torch.ones_like(x))
