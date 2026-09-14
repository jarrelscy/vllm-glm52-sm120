# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU FX tests for singleton-range PCIe fusion and residual semantics."""

import operator

import pytest
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.fusion.b12x_allreduce_rms import B12xAllReduceRMSFusionPass
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.config.utils import Range


def _graph(group="tp:0", dtype=torch.bfloat16, only_norm=False, extra_ar_user=False):
    graph = fx.Graph()
    x, residual, weight = [
        graph.placeholder(name) for name in ("x", "residual", "weight")
    ]
    for node, shape in ((x, (1, 6144)), (residual, (1, 6144)), (weight, (6144,))):
        node.meta["val"] = torch.empty(shape, dtype=dtype)
    reduced = graph.call_function(torch.ops.vllm.all_reduce.default, (x, group))
    reduced.meta["val"] = torch.empty_like(x.meta["val"])
    norm = graph.call_function(
        torch.ops.vllm_ir.fused_add_rms_norm.default,
        (reduced, residual, weight, 1e-6),
    )
    norm.meta["val"] = (
        torch.empty_like(x.meta["val"]),
        torch.empty_like(x.meta["val"]),
    )
    output = graph.call_function(operator.getitem, (norm, 0))
    output.meta["val"] = norm.meta["val"][0]
    new_residual = graph.call_function(operator.getitem, (norm, 1))
    new_residual.meta["val"] = norm.meta["val"][1]
    values = (output,) if only_norm else (output, new_residual)
    if extra_ar_user:
        values += (reduced,)
    graph.output(values)
    return graph


def test_range_selection_and_large_graph_preserved():
    fusion = B12xAllReduceRMSFusionPass("tp:0", 6144, 12288)
    assert fusion.is_applicable_for_range(Range(1, 1))
    for interval in (Range(1, 4096), Range(2, 4096), Range(4, 4)):
        assert not fusion.is_applicable_for_range(interval)
    graph = _graph()
    before = str(graph)
    if fusion.is_applicable_for_range(Range(1, 4096)):
        fusion(graph)
    assert str(graph) == before
    assert not B12xAllReduceRMSFusionPass("tp:0", 6144, 0).is_applicable_for_range(
        Range(1, 1)
    )


@pytest.mark.parametrize(
    "kwargs", [{"group": "dcp:0"}, {"dtype": torch.float32}, {"extra_ar_user": True}]
)
def test_ineligible_pair_is_unchanged(kwargs):
    graph = _graph(**kwargs)
    before = str(graph)
    B12xAllReduceRMSFusionPass("tp:0", 6144, 12288)(graph)
    assert str(graph) == before


def test_old_residual_consumer_prevents_inplace_rewrite():
    graph = _graph()
    residual = next(node for node in graph.nodes if node.target == "residual")
    output = next(node for node in graph.nodes if node.op == "output")
    output.args = (output.args[0] + (residual,),)
    before = str(graph)
    B12xAllReduceRMSFusionPass("tp:0", 6144, 12288)(graph)
    assert str(graph) == before


@pytest.mark.parametrize("only_norm", [False, True])
def test_functionalized_return_and_inplace_lowering(only_norm):
    graph = _graph(only_norm=only_norm)
    B12xAllReduceRMSFusionPass("tp:0", 6144, 12288)(graph)
    nodes = [node for node in graph.nodes if node.target == auto_functionalized]
    assert len(nodes) == 1
    assert not any(
        node.target == torch.ops.vllm.all_reduce.default for node in graph.nodes
    )

    # CPU arithmetic stand-in for communication, exercising the actual HOP and
    # residual-mutation schema without GPUs or process groups.
    def kernel(x, residual, weight, epsilon):
        summed = x.float() * 4 + residual.float()
        residual.copy_(summed)
        norm = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + epsilon)
        return (norm.to(weight.dtype) * weight).to(x.dtype)

    x = torch.randn(1, 6144, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(6144, dtype=x.dtype)
    original_residual = residual.clone()
    expected_residual = residual.clone()
    expected = kernel(x, expected_residual, weight, 1e-6)
    library = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        library.impl("glm_pcie_fused_ar_rms", kernel)
        result = fx.GraphModule({}, graph)(x, residual, weight)
        torch.testing.assert_close(result[0], expected)
        if not only_norm:
            torch.testing.assert_close(result[1], expected_residual)
        # Functionalized call preserves the input residual.
        torch.testing.assert_close(residual, original_residual)
        fixer = object.__new__(FixFunctionalizationPass)
        fixer.nodes_to_remove = []
        fixer.defunctionalize(graph, nodes[0], {1: "residual"})
        # The normal pass removes queued nodes at the end.
        for node in fixer.nodes_to_remove:
            graph.erase_node(node)
        graph.lint()
        lowered = fx.GraphModule({}, graph)(x, residual, weight)
        torch.testing.assert_close(lowered[0], expected)
        torch.testing.assert_close(residual, expected_residual)
        if not only_norm:
            assert lowered[1] is residual
    finally:
        library._destroy()
