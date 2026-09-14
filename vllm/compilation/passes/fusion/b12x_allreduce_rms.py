# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select PCIe AR+RMS fusion after vLLM specializes the compile range."""

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


class B12xAllReduceRMSFusionPass(InductorPass):
    def __init__(self, group_name: str, hidden_size: int, max_bytes: int):
        self.group_name = group_name
        self.hidden_size = hidden_size
        self.max_bytes = max_bytes

    def uuid(self) -> str:
        return self.hash_source(self) + self.hash_dict(dict(vars(self)))

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # Only the measured one-row geometry. A dynamic [1,4096] graph must
        # retain the ordinary path even if it can also be called with one row.
        return (
            compile_range.start == compile_range.end == 1
            and self.hidden_size == 6144
            and self.max_bytes >= 12288
        )

    def __call__(self, graph: fx.Graph) -> None:
        norm_op = torch.ops.vllm_ir.fused_add_rms_norm.default
        ar_op = torch.ops.vllm.all_reduce.default
        fused_op = torch.ops.vllm.glm_pcie_fused_ar_rms.default
        matched = 0
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != norm_op:
                continue
            arguments = {
                arg.name: arg.default_value for arg in norm_op._schema.arguments
            }
            arguments.update(
                zip((arg.name for arg in norm_op._schema.arguments), node.args)
            )
            arguments.update(node.kwargs)
            reduced = arguments["x"]
            residual = arguments["x_residual"]
            weight = arguments["weight"]
            if (
                not isinstance(reduced, fx.Node)
                or reduced.target != ar_op
                or len(reduced.users) != 1
                or arguments["variance_size"] is not None
                or not isinstance(residual, fx.Node)
                or not isinstance(weight, fx.Node)
            ):
                continue
            # The final cleanup restores in-place residual mutation. Preserve
            # functional semantics if another node still needs the old value.
            if len(residual.users) != 1:
                continue
            ar_args = dict(zip(("tensor", "group_name"), reduced.args))
            ar_args.update(reduced.kwargs)
            if ar_args.get("group_name") != self.group_name:
                continue
            x = ar_args["tensor"]
            x_val = x.meta.get("val")
            residual_val = residual.meta.get("val")
            weight_val = weight.meta.get("val")
            if (
                not isinstance(x_val, torch.Tensor)
                or not isinstance(residual_val, torch.Tensor)
                or not isinstance(weight_val, torch.Tensor)
                or x_val.dtype != torch.bfloat16
                or residual_val.dtype != x_val.dtype
                or weight_val.dtype != x_val.dtype
                or x_val.ndim != 2
                or x_val.shape[-1] != self.hidden_size
                or weight_val.shape != (self.hidden_size,)
                or self.hidden_size * x_val.element_size() > self.max_bytes
                or x is residual
            ):
                continue
            # IR returns (normed, new_residual); auto_functionalized returns
            # (custom_op_return, mutated_residual), so all tuple users remain
            # valid, including final-norm sites that only consume element 0.
            node.target = auto_functionalized
            node.args = (fused_op,)
            node.kwargs = {
                "x": x,
                "residual": residual,
                "weight": weight,
                "epsilon": arguments["epsilon"],
            }
            graph.erase_node(reduced)
            matched += 1
        if matched:
            graph.lint()
            logger.info_once(
                "B12X single-row compile range: fused %d all-reduce/RMS pairs",
                matched,
            )
