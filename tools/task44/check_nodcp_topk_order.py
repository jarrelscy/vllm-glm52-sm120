# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check non-DCP prefill top-k selection and ordering on fixed logits."""

import json

import torch

from vllm import _custom_ops as ops

torch.manual_seed(44015)
results = []
for rows in (1, 32):
    for tied in (False, True):
        logits = torch.randn(rows, 8192, device="cuda", dtype=torch.float32)
        if tied:
            logits = logits.round()
        starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        ends = torch.full_like(starts, logits.shape[1])
        out = torch.empty((rows, 2048), device="cuda", dtype=torch.int32)

        def run(logits=logits, starts=starts, ends=ends, out=out):
            ops.top_k_per_row_prefill(
                logits, starts, ends, out, logits.shape[0], logits.stride(0), 1, 2048
            )
            return out.clone()

        baseline = run()
        baseline_set = baseline.sort(-1).values
        order_changes = set_changes = 0
        for _ in range(100):
            current = run()
            order_changes += int(not torch.equal(current, baseline))
            set_changes += int(not torch.equal(current.sort(-1).values, baseline_set))
        results.append(
            {
                "rows": rows,
                "tied": tied,
                "repeats": 100,
                "order_changes": order_changes,
                "set_changes": set_changes,
            }
        )
print(json.dumps(results, indent=2))
