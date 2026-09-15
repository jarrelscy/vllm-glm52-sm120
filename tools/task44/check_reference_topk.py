# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference selector checks, including relative bounds, ties, and empty rows."""

import importlib.util

import torch

spec = importlib.util.spec_from_file_location(
    "task44_reference",
    "/opt/vllm/vllm/model_executor/layers/quantization/arvq_reference.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select = module.stable_indexer_topk

logits = torch.tensor(
    [[100, 1, 4, 4, 2, 100], [9, 9, 9, 9, 9, 9], [1, 2, 3, 4, 5, 6]],
    dtype=torch.float32,
)
starts = torch.tensor([1, 2, 4], dtype=torch.int32)
ends = torch.tensor([5, 6, 4], dtype=torch.int32)
output = torch.empty((3, 3), dtype=torch.int32)
expected = torch.tensor([[3, 2, 1], [2, 1, 0], [-1, -1, -1]], dtype=torch.int32)
for _ in range(100):
    select(logits, starts, ends, output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

short = torch.empty((3, 8), dtype=torch.int32)
select(logits, starts, ends, short)
expected_short = torch.tensor(
    [[3, 2, 1, 0, -1, -1, -1, -1], [3, 2, 1, 0, -1, -1, -1, -1], [-1] * 8],
    dtype=torch.int32,
)
torch.testing.assert_close(short, expected_short, rtol=0, atol=0)
print("Reference top-k: ties, relative bounds, short and empty rows PASS")

if torch.cuda.is_available():
    from vllm import _custom_ops as ops

    torch.manual_seed(44015)
    scores = torch.stack([torch.randperm(8192) for _ in range(6)]).cuda().float()
    lower = torch.tensor([7, 10, 0, 10, 15, 0], device="cuda", dtype=torch.int32)
    upper = torch.tensor(
        [8192, 8099, 4000, 2048, 128, 0], device="cuda", dtype=torch.int32
    )
    native = torch.empty((6, 2048), device="cuda", dtype=torch.int32)
    reference = torch.empty_like(native)
    select(scores, lower, upper, reference)
    ops.top_k_per_row_prefill(
        scores, lower, upper, native, 6, scores.stride(0), 1, 2048
    )
    torch.testing.assert_close(
        native.sort(-1).values, reference.sort(-1).values, rtol=0, atol=0
    )
    select(scores, torch.zeros_like(lower), upper, reference)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    torch.ops._C.persistent_topk(
        scores, upper.reshape(3, 2), native, workspace, 2048, scores.shape[1]
    )
    torch.testing.assert_close(
        native.sort(-1).values, reference.sort(-1).values, rtol=0, atol=0
    )
    print("CUDA prefill/persistent decode selected sets match reference PASS")
