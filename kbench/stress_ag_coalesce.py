#!/usr/bin/env python3
"""Real-NCCL bitwise stress for VLLM_GLM_COMM_COALESCE (AsyncAllGather
defer + single-group flush), 4 ranks.

Per iteration, mirrors the prod full-indexer-layer schedule:
  candidates AG (fp32, deferred) -> q compute stand-in -> q AG (bf16,
  flush_coalesce=True, launches both in one NCCL group) -> independent
  compute -> wait(q) -> wait(idx)
vs the serial reference (two plain all_gather_into_tensor calls). Also
exercises the self-heal paths (wait() before flush; stale defer) and
shared-idx layers (flush with empty pending). Bit-compares every output
every iteration.

  docker run --rm --gpus all --entrypoint /bin/bash --shm-size 4g \
    -e NCCL_MAX_NCHANNELS=4 -e NCCL_BUFFSIZE=1048576 -e NCCL_ALGO=RING,TREE \
    -v <wt>/vllm:/opt/vllm/vllm-overlays:ro -v <wt>/kbench:/work/kbench:ro \
    glm52-vision-sm120:latest -c 'source /opt/vllm/.venv/bin/activate && \
      torchrun --nproc-per-node=4 /work/kbench/stress_ag_coalesce.py'
(mount the worktree dcp_comm_overlap.py over the installed one)
"""
import argparse
import os

os.environ.setdefault("VLLM_GLM_COMM_OVERLAP", "1")
os.environ.setdefault("VLLM_GLM_COMM_COALESCE", "1")

import torch
import torch.distributed as dist

from vllm.v1.attention.ops.dcp_comm_overlap import (
    _PENDING_COALESCE,
    AsyncAllGather,
    dcp_comm_coalesce_enabled,
)


class GroupShim:
    def __init__(self, world_size, device_group):
        self.world_size = world_size
        self.device_group = device_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1500)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    group = GroupShim(world, dist.group.WORLD)
    assert dcp_comm_coalesce_enabled()

    B = 4
    fails = 0
    for it in range(args.iters):
        gen = torch.Generator(device="cpu").manual_seed(it * 131 + rank)
        idx = torch.randn((B, 2560, 2), generator=gen).to(dev)  # fp32
        q = (torch.randn((B, 16, 576), generator=gen) * 3).bfloat16().to(dev)

        # serial reference
        ref_idx = torch.empty((world * B, 2560, 2), dtype=torch.float32, device=dev)
        dist.all_gather_into_tensor(ref_idx.view(world * B, -1),
                                    idx.view(B, -1).contiguous(),
                                    group=group.device_group)
        ref_q = torch.empty((world * B, 16, 576), dtype=torch.bfloat16, device=dev)
        dist.all_gather_into_tensor(ref_q.view(world * B, -1),
                                    q.view(B, -1).contiguous(),
                                    group=group.device_group)

        mode = it % 4
        if mode < 2:
            # normal schedule (defer -> flush-coalesce)
            ag_i = AsyncAllGather(group, idx, dim=1)
            _ = torch.empty(2048, 512, device=dev).mul_(1.5)  # churn
            ag_q = AsyncAllGather(group, q, dim=1, flush_coalesce=True)
            _ = torch.empty(1024, 512, device=dev).add_(1.0)  # churn
            got_q = ag_q.wait()
            got_i = ag_i.wait()
        elif mode == 2:
            # self-heal: wait() before any flush (solo launch)
            ag_i = AsyncAllGather(group, idx, dim=1)
            got_i = ag_i.wait()
            ag_q = AsyncAllGather(group, q, dim=1, flush_coalesce=True)
            got_q = ag_q.wait()
        else:
            # shared-idx layer: flush with empty pending
            assert not _PENDING_COALESCE
            ag_q = AsyncAllGather(group, q, dim=1, flush_coalesce=True)
            got_q = ag_q.wait()
            ag_i = AsyncAllGather(group, idx, dim=1)
            got_i = ag_i.wait()

        # AsyncAllGather does dim=1 concat epilogue; rebuild reference layout
        ref_i_d1 = (
            ref_idx.view(world, B, 2560, 2).movedim(0, 1).reshape(B, world * 2560, 2)
        )
        ref_q_d1 = (
            ref_q.view(world, B, 16, 576).movedim(0, 1).reshape(B, world * 16, 576)
        )
        if not torch.equal(got_i, ref_i_d1) or not torch.equal(got_q, ref_q_d1):
            fails += 1
            print(f"[rank{rank}] iter {it} mode {mode}: MISMATCH", flush=True)
        assert not _PENDING_COALESCE, f"pending leak at iter {it}"

        if it % 500 == 0 and rank == 0:
            print(f"iter {it}: OK so far ({fails} fails)", flush=True)

    total = torch.tensor([fails], device=dev)
    dist.all_reduce(total)
    if rank == 0:
        print(f"DONE: {args.iters} iters, total mismatches across ranks: "
              f"{int(total.item())} -> {'PASS' if total.item() == 0 else 'FAIL'}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
