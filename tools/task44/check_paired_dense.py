# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paired dense FP4 MMA against the independent matrix reference."""

import json
import os

import torch

from vllm.model_executor.layers.quantization.arvq_reference import hot_rows, projection
from vllm.model_executor.layers.quantization.nvfp4_p4_linear import (
    _dequantize,
    _paired_projection,
    dense_p4,
    quantize_weight,
)

torch.manual_seed(44015)
torch.set_num_threads(4)
# Compare the reference against FP32-accumulating BF16 GEMM. Default truncated
# BF16 reduction is a separately measured approximation, not reference error.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
results = []
for k in (4096, 16384):
    n = 64
    weight = (torch.randn(n, k) * 0.01).bfloat16()
    packed, scales, global_scale = [v.to("cuda") for v in quantize_weight(weight)]
    expected_weights = (
        hot_rows(packed, scales, n, k, 0) * global_scale.item()
    ).bfloat16()
    torch.testing.assert_close(
        _dequantize(packed, scales, global_scale, n, k),
        expected_weights,
        rtol=0,
        atol=0,
    )
    tensors = [packed, packed, scales, packed, scales, global_scale]
    for slots in (2, 4, 5, 8, 16):
        x = torch.randn(slots, k, device="cuda", dtype=torch.float16)
        cold = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
        hot = torch.zeros_like(cold)
        expected = projection(x, cold, hot, tensors, 0.0, n, 1, 1)
        split = 8 if slots <= 2 else 4 if slots <= 4 else 2
        actual = _paired_projection(x, packed, scales, global_scale, n, split)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=0.001)
        results.append(
            {"k": k, "slots": slots, "max_abs": (actual - expected).abs().max().item()}
        )
    for slots in (2, 4, 16, 32):
        x = torch.randn(slots, k, device="cuda", dtype=torch.bfloat16)
        routes = torch.stack(
            (
                torch.full((slots,), -1, device="cuda", dtype=torch.int32),
                torch.zeros(slots, device="cuda", dtype=torch.int32),
            )
        )
        os.environ["VLLM_NVFP4_P4_PAIRED"] = "1"
        os.environ["VLLM_ARVQ_REFERENCE_WEIGHTS"] = "0"
        actual = dense_p4(x, packed, scales, global_scale, routes, 16)
        os.environ["VLLM_ARVQ_REFERENCE_WEIGHTS"] = "1"
        expected = dense_p4(x, packed, scales, global_scale, routes, 16)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.01, atol=0.005
        )
        os.environ["VLLM_ARVQ_REFERENCE_WEIGHTS"] = "0"
print(json.dumps(results, indent=2))
