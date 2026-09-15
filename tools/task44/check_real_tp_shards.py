# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real layer-3 hot/cold expert reconstruction through production loaders."""

import ast
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.arvq_reference import cold_rows, hot_rows

torch.set_num_threads(4)
root = Path("/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-initial-8x8")
mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
prefix = "model.layers.3.mlp.experts."
# Execute the actual production loader helpers without initializing a MoE layer.
source = Path("/opt/vllm/vllm/model_executor/layers/quantization/tp_hybrid_moe.py")
tree = ast.parse(source.read_text())
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in ("_shard_range", "_gateup_loader", "_rowk_loader")
]
namespace = {"torch": torch}
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace
)


def load(suffix, expert=False):
    name = prefix + suffix
    with safe_open(root / mapping[name], framework="pt", device="cpu") as f:
        return f.get_slice(name)[:1] if expert else f.get_tensor(name)


def shard(tensor, rank, proj):
    axis = 1 if proj == "w13" else 2
    shape = list(tensor.shape)
    shape[axis] //= 4
    parameter = torch.nn.Parameter(torch.empty(shape, dtype=tensor.dtype), False)
    loader = (
        namespace["_gateup_loader"](rank, 4, tensor.shape[1] // 2, axis)
        if proj == "w13"
        else namespace["_rowk_loader"](rank, 4, axis)
    )
    loader(parameter, tensor)
    return parameter.detach()


def decode(packed, scales, proj, cold, codebooks=None, global_scale=None):
    if cold:
        n, k = packed.shape[1] * 16, packed.shape[2] * 64
        words = packed.contiguous().view(torch.int32).flatten()
        rows = [
            cold_rows(words, codebooks, scales, n, k, 0, i, min(i + 1024, n))
            for i in range(0, n, 1024)
        ]
        return torch.cat(rows) * global_scale.item()
    e, n, k2 = packed.shape
    k = k2 * 2
    native = (
        packed.contiguous()
        .view(torch.int32)
        .reshape(e, n // 16, 2, 8, k // 64, 2, 4)
        .permute(0, 1, 4, 5, 2, 3, 6)
        .contiguous()
    )
    rows = [
        hot_rows(
            native, scales.contiguous().view(torch.int32), n, k, 0, i, min(i + 1024, n)
        )
        for i in range(0, n, 1024)
    ]
    weight = torch.cat(rows)
    parts = 2 if proj == "w13" else 1
    return weight * global_scale.reshape(parts).repeat_interleave(n // parts)[:, None]


results = []
for cold in (True, False):
    for proj in ("w13", "w2"):
        stem = ("arvq_" if cold else "nvfp4_") + proj
        packed = load(stem + "_packed", expert=True)
        scales = load(stem + ("_scales" if cold else "_bscale"), expert=True)
        books = load(stem + "_codebooks") if cold else None
        alpha = load(stem + ("_global" if cold else "_scale2"), expert=not cold)
        full = decode(packed, scales, proj, cold, books, alpha)
        pieces = [
            decode(
                shard(packed, rank, proj),
                shard(scales, rank, proj),
                proj,
                cold,
                books,
                alpha,
            )
            for rank in range(4)
        ]
        reconstructed = (
            torch.cat(
                [p[: p.shape[0] // 2] for p in pieces]
                + [p[p.shape[0] // 2 :] for p in pieces]
            )
            if proj == "w13"
            else torch.cat(pieces, dim=1)
        )
        torch.testing.assert_close(reconstructed, full, rtol=0, atol=0)
        results.append(
            {
                "cold": cold,
                "projection": proj,
                "shape": list(full.shape),
                "bit_exact": True,
            }
        )
        del full, pieces, reconstructed
print(json.dumps(results, indent=2))
