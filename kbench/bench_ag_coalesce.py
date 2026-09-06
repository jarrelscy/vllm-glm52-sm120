#!/usr/bin/env python3
"""Is coalescing AG(idx)+AG(q) into one NCCL group better than the current
comm-overlap schedule? 4-rank microbench on real GLM-5.3 decode sizes.

Schedules compared (per 'layer', B=4 decode tokens):
  A) overlap (prod, VLLM_GLM_COMM_OVERLAP=1):
       AG(idx) async -> q-compute kernels (~8us) -> AG(q) async ->
       idx-independent work (~5us, precompute_mqa_indices stand-in) ->
       wait(q) -> wait(idx)
  B) coalesced:
       q-compute kernels -> [coalescing_manager: AG(idx) + AG(q)] ->
       idx-independent work -> wait(all)
  C) serial baseline (no overlap, no coalesce)

Run:
  docker run --rm --gpus all --entrypoint /bin/bash --shm-size 4g \
    -e NCCL_MAX_NCHANNELS=4 -e NCCL_BUFFSIZE=1048576 -e NCCL_ALGO=RING,TREE \
    -v .../kbench:/work/kbench:ro glm52-vision-sm120:latest -c \
    'source /opt/vllm/.venv/bin/activate && torchrun --nproc-per-node=4 \
       /work/kbench/bench_ag_coalesce.py'
"""
import os

import torch
import torch.distributed as dist

ITERS = 300
WARMUP = 40
B = 4


def dummy_compute(x, n):
    for _ in range(n):
        x = x * 1.0001 + 0.01
    return x


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    # real-ish decode payloads (per rank)
    idx = torch.randint(0, 1 << 30, (B, 2560), dtype=torch.int32, device=dev)
    q = torch.randn(B, 16, 576, dtype=torch.bfloat16, device=dev)
    idx_out = torch.empty((world * B, 2560), dtype=torch.int32, device=dev)
    q_out = torch.empty((world * B, 16, 576), dtype=torch.bfloat16, device=dev)
    comp = torch.randn(4096, 512, device=dev, dtype=torch.bfloat16)

    def sched_overlap():
        w1 = dist.all_gather_into_tensor(idx_out, idx, group=group, async_op=True)
        c = dummy_compute(comp, 6)  # ~q-side compute
        w2 = dist.all_gather_into_tensor(q_out, q, group=group, async_op=True)
        c = dummy_compute(c, 4)  # precompute indices stand-in
        w2.wait()
        w1.wait()
        return c

    def sched_coalesced():
        c = dummy_compute(comp, 6)
        with dist._coalescing_manager(group=group, async_ops=True) as cm:
            dist.all_gather_into_tensor(idx_out, idx, group=group)
            dist.all_gather_into_tensor(q_out, q, group=group)
        c = dummy_compute(c, 4)
        cm.wait()
        return c

    def sched_serial():
        c = dummy_compute(comp, 6)
        dist.all_gather_into_tensor(idx_out, idx, group=group)
        c = dummy_compute(c, 4)
        dist.all_gather_into_tensor(q_out, q, group=group)
        return c

    results = {}
    for name, fn in (
        ("overlap", sched_overlap),
        ("coalesced", sched_coalesced),
        ("serial", sched_serial),
    ):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        results[name] = s.elapsed_time(e) * 1000 / ITERS
        dist.barrier()

    if rank == 0:
        print(
            f"[rank0] per-layer us: "
            + " ".join(f"{k}={v:.2f}" for k, v in results.items())
            + f" | coalesce-vs-overlap delta={results['overlap']-results['coalesced']:+.2f}us"
            + f" (x24 idx layers/step = {(results['overlap']-results['coalesced'])*24/1000:+.3f}ms)",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
