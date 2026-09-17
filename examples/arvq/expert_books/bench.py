# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated full routed-MLP v2/v3 timing; these are NOT model generation TPS.

Run with the serving venv. Use torchrun --nproc-per-node=4 for TP4 NCCL.
Duplicate real shared books supplied via --books; no checkpoint is modified.
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import arvq_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cold", type=int, default=173)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 32, 128, 256])
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.accelerator.set_device_index(rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(100 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, hot = args.cold, 256 - args.cold
    h, ish = 6144, 2048 // world
    stored = np.load(args.books)
    tensors = []
    for proj, n, k, parts in [("w13", 2 * ish, h, 2), ("w2", h, ish, 1)]:
        packed = torch.randint(
            -(2**31),
            2**31 - 1,
            (e * (n // 16) * (k // 64) * 64 + 1,),
            device="cuda",
            dtype=torch.int32,
        )
        book = torch.from_numpy(stored[proj]).cuda()
        scales = torch.full(
            (e, n // 16, k // 128, 16), 48, device="cuda", dtype=torch.uint8
        )
        hw = torch.full(
            (hot, n // 16, k // 64, 4, 32), 0x22222222, device="cuda", dtype=torch.int32
        )
        hs = torch.full((hot, n, k // 64), 0x38383838, device="cuda", dtype=torch.int32)
        hg = torch.full((hot, parts), 1 / 256, device="cuda")
        tensors.extend([packed, book, scales, hw, hs, hg])
    duplicate = list(tensors)
    for offset in (1, 7):
        duplicate[offset] = tensors[offset].repeat(e, 1)
    lookup = torch.full((2, 256), -1, device="cuda", dtype=torch.int32)
    lookup[0, :e] = torch.arange(e, device="cuda", dtype=torch.int32)
    lookup[1, e:] = torch.arange(hot, device="cuda", dtype=torch.int32)
    # Fixed model inputs/routing on every rank, but rank-local packed shards.
    torch.manual_seed(600)
    records = []
    for tokens in args.tokens:
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.25
        routing = torch.rand(tokens, 256, device="cuda").topk(8, dim=-1).indices
        weights = torch.rand(tokens, 8, device="cuda").softmax(-1)
        if world > 1:
            for t in (x, routing, weights):
                dist.broadcast(t, 0)
        for route_case in ("mixed", "cold"):
            ids = routing if route_case == "mixed" else routing % e

            def run(ts, x=x, weights=weights, ids=ids):
                out = arvq_mlp(x, weights, ids, lookup, ts, [1 / 256, 1 / 256], 128)
                if world > 1:
                    dist.all_reduce(out)
                return out

            base, candidate = run(tensors), run(duplicate)
            torch.accelerator.synchronize()
            assert torch.equal(base, candidate), (world, rank, tokens, route_case)
            assert torch.isfinite(base).all()
            for graph_mode in (False, True):
                functions = []
                for ts in (tensors, duplicate):
                    for _ in range(3):
                        run(ts)
                    if graph_mode:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = run(ts)
                        functions.append((graph.replay, output, graph))
                    else:
                        functions.append((lambda ts=ts: run(ts), None, None))
                for index in range(len(functions)):
                    functions[index][0]()
                torch.accelerator.synchronize()
                if graph_mode:
                    assert torch.equal(functions[0][1], functions[1][1])
                samples = [[], []]
                repetitions = 20 if tokens <= 32 else 5
                # Alternate order to reduce monotonic clock/thermal bias.
                for trial in range(8):
                    for which in [0, 1] if trial % 2 == 0 else [1, 0]:
                        if world > 1:
                            dist.barrier()
                        start, end = (
                            torch.Event(enable_timing=True),
                            torch.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(repetitions):
                            functions[which][0]()
                        end.record()
                        end.synchronize()
                        ms = start.elapsed_time(end) / repetitions
                        if world > 1:
                            slowest = torch.tensor(ms, device="cuda")
                            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
                            ms = slowest.item()
                        samples[which].append(ms)
                b, c = map(statistics.median, samples)
                row = dict(
                    tp=world,
                    tokens=tokens,
                    route_case=route_case,
                    graphs=graph_mode,
                    baseline_ms=b,
                    expert_ms=c,
                    ratio=c / b,
                    bit_exact=True,
                    cold_experts=e,
                    book_bytes_shared=4096,
                    book_bytes_expert=e * 4096,
                    allocated_bytes=torch.accelerator.memory_allocated(),
                    baseline_samples_ms=samples[0],
                    expert_samples_ms=samples[1],
                )
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    records.append(row)
                    Path(args.out).write_text(json.dumps(records, indent=2))
                del functions
                if graph_mode:
                    del graph
                    gc.collect()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
