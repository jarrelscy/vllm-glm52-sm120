# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure dense dispatch differences on identical activation and weight rows."""

import json
import os

import torch

from vllm.model_executor.layers.quantization.nvfp4_p4_linear import (
    _dequantize,
    dense_p4,
    quantize_weight,
)

torch.manual_seed(44015)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
os.environ["VLLM_ARVQ_REFERENCE_WEIGHTS"] = "0"
os.environ["VLLM_NVFP4_P4_PAIRED"] = "1"
results = []
for k in (4096, 16384):
    weight = (torch.randn(256, k) * 0.01).bfloat16()
    packed, scales, global_scale = [v.cuda() for v in quantize_weight(weight)]
    x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16)
    routes = torch.zeros((2, 32), device="cuda", dtype=torch.int32)
    routes[0].fill_(-1)
    small = dense_p4(x[:16], packed, scales, global_scale, routes, 16)
    fallback = dense_p4(x, packed, scales, global_scale, routes, 16)[:16]
    uniform = dense_p4(x, packed, scales, global_scale, routes, 32)[:16]
    local_weight = _dequantize(packed, scales, torch.ones_like(global_scale), 256, k)
    scaled_after = (
        torch.mm(x, local_weight.T, out_dtype=torch.float32) * global_scale
    ).to(x.dtype)[:16]
    for name, actual in (
        ("bf16_fallback", fallback),
        ("fp4_32_rows", uniform),
        ("bf16_global_scale_after_gemm", scaled_after),
    ):
        error = actual.float() - small.float()
        results.append(
            {
                "k": k,
                "comparison": name,
                "max_abs": error.abs().max().item(),
                "relative_l2": (error.norm() / small.float().norm()).item(),
                "unequal": (actual != small).sum().item(),
                "total": small.numel(),
            }
        )
print(json.dumps(results, indent=2))
