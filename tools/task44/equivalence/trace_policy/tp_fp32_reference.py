# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager diagnostic: retain row-parallel partial outputs until reduction."""

import ast
import types
from pathlib import Path

import torch


def clone_without_output_cast(module, name, dense=False):
    tree = ast.parse(Path(module.__file__).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []

    class Change(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and len(node.args) == 1
                and ast.unparse(node.args[0]) == "x.dtype"
            ):
                return node.func.value
            if dense and ast.unparse(node.func) == "torch.empty":
                # The reference prefill branch preallocates its row-parallel
                # output: removing terminal casts alone leaves BF16 partials.
                for keyword in node.keywords:
                    if (
                        keyword.arg == "dtype"
                        and ast.unparse(keyword.value) == "x.dtype"
                    ):
                        keyword.value = ast.parse("torch.float32", mode="eval").body
            if dense and ast.unparse(node.func) == "torch.nn.functional.linear":
                return (
                    ast.parse("torch.mm(flat, decoded.T, out_dtype=torch.float32)")
                    .body[0]
                    .value
                )
            return node

    node = Change().visit(node)
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(module))
    exec(compile(code, module.__file__, "exec"), namespace)
    return namespace[name]


def install(model):
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as arvq
    from vllm.model_executor.layers.quantization import nvfp4_p4_linear as dense

    dense_fp32 = clone_without_output_cast(dense, "dense_p4", dense=True)
    if not getattr(arvq, "_task44_fp32", False):
        arvq.arvq_mlp = clone_without_output_cast(arvq, "arvq_mlp")
        arvq._task44_fp32 = True
        # This diagnostic intentionally supports only small eager batches.
        arvq._grouped_prefill_enabled = lambda *args: False

    def apply(method, layer, x, bias=None):
        kind = type(method).__name__
        if kind == "NvFp4P4LinearMethod":
            out = dense_fp32(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global,
                layer.p4_route_ids,
                layer.p4_native_max_tokens,
            )
        elif kind == "UnquantizedLinearMethod":
            out = torch.mm(
                x.reshape(-1, x.shape[-1]), layer.weight.T, out_dtype=torch.float32
            ).reshape(*x.shape[:-1], layer.weight.shape[0])
        else:
            raise RuntimeError(f"Unsupported FP32 row-parallel method: {kind}")
        return out if bias is None else out + bias.float()

    def cast_after(module, args, output):
        dtype = args[0].dtype
        if isinstance(output, tuple):
            return (output[0].to(dtype), *output[1:])
        return output.to(dtype)

    count = 0
    for child in model.modules():
        if isinstance(child, RowParallelLinear):
            # Quant methods may be shared; only replace this instance's method.
            import copy

            child.quant_method = copy.copy(child.quant_method)
            child.quant_method.apply = types.MethodType(apply, child.quant_method)
            if child.reduce_results:
                child.register_forward_hook(cast_after)
            count += 1
        elif type(child).__name__ == "DeepseekV2MoE":
            child.register_forward_hook(cast_after)
    print(f"TASK44 FP32 partial reductions installed: {count} row linears", flush=True)
