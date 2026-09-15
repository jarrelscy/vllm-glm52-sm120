# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure early BF16 rounding of real TP-sharded dense FP4 partial outputs."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import _projection
from vllm.model_executor.layers.quantization.nvfp4_p4_linear import quantize_weight

torch.set_num_threads(4)
root = Path("/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-initial-8x8")
trace = torch.load(
    "/opt/task44-traces/arvq-task44-b16-pp-strict/rank0-step0033-pos996.pt",
    map_location="cpu",
    weights_only=True,
)
x = trace["records"]["model.layers.0.self_attn.o_proj:input"].cuda().half()[None]
name = "model.layers.0.self_attn.o_proj.weight"
mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(root / mapping[name], framework="pt", device="cpu") as f:
    packed, scales, alpha = [v.cuda() for v in quantize_weight(f.get_tensor(name))]
n = packed.shape[1] * 16
cold = torch.full((1,), -1, device="cuda", dtype=torch.int32)
hot = torch.zeros_like(cold)


def project(activation, weight, block_scales):
    tensors = [weight, weight, block_scales, weight, block_scales, alpha]
    return _projection(activation.contiguous(), cold, hot, tensors, 0.0, n, 8, 1)


full = project(x, packed, scales).bfloat16()
parts = [
    project(a, w.contiguous(), s.contiguous())
    for a, w, s in zip(x.chunk(4, -1), packed.chunk(4, 2), scales.chunk(4, -1))
]
results = []
for rounding in (True, False):
    values = [p.bfloat16().float() if rounding else p for p in parts]
    result = torch.stack(values).sum(0).bfloat16()
    error = result.float() - full.float()
    results.append(
        {
            "round_partials_to_bf16": rounding,
            "unequal": int((result != full).sum()),
            "elements": full.numel(),
            "max_abs": error.abs().max().item(),
            "relative_l2": (error.norm() / full.float().norm()).item(),
        }
    )
print(json.dumps(results, indent=2))
