# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 NCCL parity against the independently decoded plane reference."""

import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist

spec = importlib.util.spec_from_file_location(
    "fixtures",
    Path(__file__).resolve().parents[3]
    / "tests/quantization/test_arvq_expert_books.py",
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.accelerator.set_device_index(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
os.environ["ARVQ_V3_PARITY_REPORT"] = f"/results/distributed-parity-rank{rank}.jsonl"
torch.manual_seed(77)
h, intermediate, tokens = 6144, 2048, 4
f13, p13, s13 = fixtures.fixture(2 * intermediate, h)
f2, p2, s2 = fixtures.fixture(h, intermediate, seed=19)
layer = fixtures.layer_from(h, intermediate, f13, f2, rank, world)
x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.25
ids = torch.tensor([[2, 0], [0, 2]] * 2, device="cuda")
weights = torch.tensor([[0.3, 0.7]] * tokens, device="cuda")
for t in (x, ids, weights):
    dist.broadcast(t, 0)
slots = layer._arvq_lookups[0, ids.flatten()].long().tolist()
ish = intermediate // world
ref13 = fixtures.oracle(p13, f13[1], s13, 1 / 256)
ref13 = torch.cat(
    (
        ref13[:, rank * ish : (rank + 1) * ish],
        ref13[:, intermediate + rank * ish : intermediate + (rank + 1) * ish],
    ),
    1,
).cuda()
ref2 = fixtures.oracle(p2, f2[1], s2, 1 / 256)[
    :, :, rank * ish : (rank + 1) * ish
].cuda()
px = fixtures.activation_planes(x.half().repeat_interleave(2, 0))
h13 = torch.stack([px[i] @ ref13[e].T for i, e in enumerate(slots)]).half()
gate, up = h13.chunk(2, -1)
act = (torch.nn.functional.silu(gate) * up).half()
pa = fixtures.activation_planes(act)
down = torch.stack([pa[i] @ ref2[e].T for i, e in enumerate(slots)])
expected = (down.view(tokens, 2, h) * weights[..., None]).sum(1).to(x.dtype)
actual = fixtures.runtime.arvq_mlp(
    x, weights, ids, layer._arvq_lookups, layer._arvq_tensors, layer._arvq_alphas, 128
)
fixtures.report(f"tp{world}-rank{rank}-local", actual, expected, 0.006)
dist.all_reduce(actual)
dist.all_reduce(expected)
fixtures.report(f"tp{world}-rank{rank}-nccl-sum", actual, expected, 0.006)
print(f"TP{world} rank {rank}: reference parity passed", flush=True)
dist.destroy_process_group()
