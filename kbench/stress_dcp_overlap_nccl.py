#!/usr/bin/env python3
"""Real-NCCL determinism stress for VLLM_GLM_COMM_OVERLAP (AsyncAllGather +
deferred DCP top-k merge), vllm/v1/attention/ops/dcp_comm_overlap.py.

The shipped test (tests/kernels/attention/test_dcp_comm_overlap.py) fakes the
collective on a single GPU, so real ProcessGroupNCCL semantics -- side-stream
enqueue, allocator interaction with in-flight collectives, capture behavior --
were never exercised. This harness replays the EXACT prod schedule with a real
2-rank NCCL group and heavy allocator churn between enqueue and wait:

  per layer (async path, mirrors sparse_attn_indexer + mla_attention):
    1. pack:    packed = f(iter) written on the compute stream
    2. AG(idx): AsyncAllGather(packed) enqueued; the caller's reference to
                `packed` is DROPPED immediately (prod: _merge_dcp_topk_global
                _async's closure captures only the AsyncAllGather, not packed)
    3. q-side compute: fresh torch.empty allocations + writes on the compute
                stream (bmm/cat stand-ins) -- the allocator churn that could
                reuse `packed`'s block while NCCL still reads it
    4. AG(q):   AsyncAllGather(q) enqueued
    5. consume: gathered = AG(idx).wait(); merge -> topk buffer
    6. convert: reads the topk buffer (index-conversion stand-in)
    7. q_g = AG(q).wait(); out = g(q_g, converted)

  sync path: identical kernels, identical inputs, serial collectives
  (dist.all_gather_into_tensor without async_op), i.e. the base stack order.

Bit-compare (a) async vs sync every iteration and (b) async vs iteration 0
across iterations. Any mismatch = the overlap path is not a pure scheduling
change on real NCCL.

Bisect knobs:
  --hold-input   keep the AG input tensor alive until wait() (candidate fix)
  --no-churn     drop the allocator churn in the overlap window
  --graph        capture one async iteration in a CUDA graph and replay.
                 CAVEAT: capturing raw torch.distributed NCCL collectives
                 without vLLM's coordinated graph_capture() machinery can
                 hang (observed co-tenant on busy GPUs, 2026-09-05); the
                 eager mode is the load-bearing check here — prod exercises
                 the captured schedule directly.

Run (co-tenant safe, ~1 GiB/GPU):
  torchrun --nproc-per-node=2 kbench/stress_dcp_overlap_nccl.py --iters 2000
  (CUDA_VISIBLE_DEVICES picks the GPUs)
"""
import argparse
import importlib.util
import os
import pathlib
import sys

import torch
import torch.distributed as dist

HERE = pathlib.Path(__file__).resolve().parent
_MOD = HERE.parent / "vllm/v1/attention/ops/dcp_comm_overlap.py"
spec = importlib.util.spec_from_file_location("dcp_ovl", str(_MOD))
dcp_ovl = importlib.util.module_from_spec(spec)
sys.modules["dcp_ovl"] = dcp_ovl
spec.loader.exec_module(dcp_ovl)
AsyncAllGather = dcp_ovl.AsyncAllGather


class Group:
    """Minimal stand-in for the DCP GroupCoordinator fields AsyncAllGather
    uses, plus the sync base-communicator all_gather for the reference."""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = dist.group.WORLD

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        # DeviceCommunicatorBase.all_gather, verbatim reshape epilogue.
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device)
        dist.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group)
        output_tensor = output_tensor.reshape(
            (self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1:])


def make_layer_inputs(rank, layer, dev, rows=22, topk=2048, qt=4, qh=20,
                      qd=576):
    """Deterministic per-(rank, layer) base inputs, created once."""
    g = torch.Generator().manual_seed(1000 * rank + layer)
    pack_src = torch.randn(rows, topk, 2, generator=g).to(dev)
    q_src = (torch.randn(qt, qh, qd, generator=g) / 8).to(torch.bfloat16).to(dev)
    bmm_w = (torch.randn(qh, qd, qd, generator=g) / qd**0.5).to(
        torch.bfloat16).to(dev)
    return pack_src, q_src, bmm_w


def one_layer(group, ins, topk_buffer, *, overlap, hold_input, churn,
              churn_bufs):
    pack_src, q_src, bmm_w = ins
    dev = pack_src.device

    # 1. pack (compute-stream write into a fresh allocation, like prod's
    #    torch.empty + pack kernel)
    packed = torch.empty_like(pack_src)
    packed.copy_(pack_src)
    packed.mul_(1.0000001)  # a real kernel touching every element

    held = None
    if overlap:
        ag_idx = AsyncAllGather(group, packed, dim=1)
        if hold_input:
            held = packed
        del packed  # prod drops its reference here
    else:
        gathered_idx = group.all_gather(packed, dim=1)

    # 3. q-side compute: allocations + kernels in the overlap window
    q_nope = q_src.transpose(0, 1)  # (qh, qt, qd)
    q_bmm = torch.empty_like(q_nope)
    torch.bmm(q_nope, bmm_w, out=q_bmm)
    q_cat = torch.cat([q_bmm.transpose(0, 1), q_src], dim=-1)
    if churn:
        # extra fresh allocations sized near `packed` to invite block reuse
        for cb in churn_bufs:
            t = torch.empty(cb, device=dev)
            t.fill_(float(cb))
            del t

    # 4. AG(q)
    if overlap:
        ag_q = AsyncAllGather(group, q_cat, dim=1)
        # 5. consume deferred merge
        gathered_idx = ag_idx.wait()
    else:
        q_g = group.all_gather(q_cat, dim=1)

    # merge stand-in: deterministic reduction of the gathered candidates
    # (fixed order, content-sensitive) into the persistent topk buffer
    merged = gathered_idx[..., 0] * 3.0 + gathered_idx[..., 1]
    topk_buffer.copy_(merged[:, : topk_buffer.shape[1]])

    # 6. convert stand-in: reads the buffer
    conv = topk_buffer * 0.5 + 1.0

    # 7. wait on AG(q), combine
    if overlap:
        q_g = ag_q.wait()
    out = q_g.float().sum(dim=(1, 2)) + conv.sum(dim=1)[: q_g.shape[0]]
    if held is not None:
        del held
    return out.clone(), gathered_idx.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hold-input", action="store_true")
    ap.add_argument("--no-churn", action="store_true")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl")
    group = Group(world, rank)

    layers = [make_layer_inputs(rank, l, dev) for l in range(args.layers)]
    topk_buffer = torch.zeros(22, 1024, device=dev)
    churn_bufs = [22 * 2048 * 2, 22 * 2048, 1 << 16, 3 << 16]

    def run(overlap):
        outs = []
        for ins in layers:
            o, gi = one_layer(
                group, ins, topk_buffer, overlap=overlap,
                hold_input=args.hold_input, churn=not args.no_churn,
                churn_bufs=churn_bufs)
            outs.append((o, gi))
        return outs

    # references from the serial path
    ref = run(False)
    torch.cuda.synchronize()
    dist.barrier()

    bad_vs_sync = bad_vs_first = 0
    first_async = None

    if args.graph:
        # warmup then capture ONE async pass; replay with fixed inputs
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run(True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap = run(True)
        for it in range(args.iters):
            g.replay()
            torch.cuda.synchronize()
            cur = [(o.clone(), gi.clone()) for o, gi in cap]
            if first_async is None:
                first_async = cur
            ok_sync = all(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for a, b in zip(cur, ref))
            ok_first = all(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for a, b in zip(cur, first_async))
            bad_vs_sync += not ok_sync
            bad_vs_first += not ok_first
    else:
        for it in range(args.iters):
            cur = run(True)
            torch.cuda.synchronize()
            if first_async is None:
                first_async = cur
            ok_sync = all(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for a, b in zip(cur, ref))
            ok_first = all(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for a, b in zip(cur, first_async))
            bad_vs_sync += not ok_sync
            bad_vs_first += not ok_first

    flags = (f"hold_input={args.hold_input} churn={not args.no_churn} "
             f"graph={args.graph}")
    tag = "RACE" if (bad_vs_sync or bad_vs_first) else "ok  "
    print(f"[{tag}] rank{rank} {flags}: vs-sync {bad_vs_sync}/{args.iters}, "
          f"vs-first-async {bad_vs_first}/{args.iters}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if (bad_vs_sync or bad_vs_first) else 0)


if __name__ == "__main__":
    main()
