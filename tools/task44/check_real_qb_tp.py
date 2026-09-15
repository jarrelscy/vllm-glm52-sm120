# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full versus TP-sliced BF16 Q projection on a captured real activation."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(4)
root = Path("/data/models/jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid-initial-8x8")
trace = torch.load(
    "/opt/task44-traces/arvq-task44-b13-pp-trace/rank0-step0000-pos31.pt",
    map_location="cpu",
    weights_only=True,
)
row = trace["records"]["model.layers.0.self_attn.q_a_layernorm:output"]
x = row.cuda().bfloat16().repeat(32, 1)
name = "model.layers.0.self_attn.q_b_proj.weight"
mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(root / mapping[name], framework="pt", device="cpu") as f:
    weight = f.get_tensor(name).cuda()
reference = (x[:1].double() @ weight.double().T).bfloat16()[0]
results = []
for reduced_precision in (True, False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision
    )
    full = torch.nn.functional.linear(x, weight)[-1]
    sharded = torch.cat(
        [torch.nn.functional.linear(x, part)[-1] for part in weight.chunk(4, dim=0)]
    )
    results.append(
        {
            "bf16_reduced_precision": reduced_precision,
            "full_vs_tp_unequal": int((full != sharded).sum()),
            "full_vs_tp_max_abs": (full.float() - sharded.float()).abs().max().item(),
            "full_vs_fp64_unequal": int((full != reference).sum()),
            "tp_vs_fp64_unequal": int((sharded != reference).sum()),
            "elements": full.numel(),
        }
    )
print(json.dumps(results, indent=2))
