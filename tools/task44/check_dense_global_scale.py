# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression: shared scale makes NVFP4 independent of column sharding."""

import torch

from vllm.model_executor.layers.quantization.arvq_reference import hot_rows
from vllm.model_executor.layers.quantization.nvfp4_p4_linear import quantize_weight

torch.set_num_threads(4)
torch.manual_seed(44015)
w = torch.randn(32, 512).to(torch.bfloat16)
w[:, :128] *= 4
maximum = w.abs().amax().float()


def decode(weight, amax):
    packed, scales, global_scale = quantize_weight(weight, amax)
    n, k = weight.shape
    return hot_rows(packed, scales, n, k, 0) * global_scale.item()


full = decode(w, maximum)
shared = torch.cat([decode(shard, maximum) for shard in w.chunk(4, dim=1)], dim=1)
local = torch.cat([decode(shard, None) for shard in w.chunk(4, dim=1)], dim=1)
torch.testing.assert_close(full, shared, rtol=0, atol=0)
assert not torch.equal(full, local), "Fixture must expose per-shard scale changes"
print("PASS: shared scale is bit-exact under TP4 column partitioning")
