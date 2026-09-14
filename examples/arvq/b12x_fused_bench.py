# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual TP4 remote-push fused AR+add+RMSNorm and BF16 copy-engine DMA.
Run only with idle serving GPUs. Captured resets prevent accumulated residual/input
corruption. All measured arms use the same reset copies. No production flag edits.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist


def cmd(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as e:
        return repr(e)


def snapshot():
    return cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pstate,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active,memory.used",
            "--format=csv,noheader",
        ]
    )


def exchange(v, group):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, v, group=group)
    return values


def cpucheck(actual, reference, group):
    a = actual.cpu()
    af = a.float()
    rf = reference.float()
    digest = hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest()
    hashes = exchange(digest, group)
    r = {
        "finite": bool(torch.isfinite(af).all()),
        "nonzero": bool(torch.count_nonzero(a)),
        "identical_across_ranks": len(set(hashes)) == 1,
        "relative_l2": float((af - rf).norm() / rf.norm().clamp_min(1e-12)),
        "max_abs_error": float((af - rf).abs().max()),
    }
    assert (
        r["finite"]
        and r["nonzero"]
        and r["identical_across_ranks"]
        and r["relative_l2"] < 0.025
    ), r
    return r


def make_graph(fn, stream, group, repeats, context=lambda: contextlib.nullcontext()):
    with torch.cuda.stream(stream):
        for _ in range(4):
            fn()
    stream.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with context(), torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            fn()
    for _ in range(4):
        graph.replay()
    torch.accelerator.synchronize()
    return graph


def time_pair(graphs, group, reps, iters, capture_repeats):
    samples = {k: [] for k in graphs}
    names = list(graphs)
    for rep in range(reps):
        for name in names if rep % 2 == 0 else names[::-1]:
            torch.accelerator.synchronize()
            dist.barrier(group=group)
            s = torch.Event(enable_timing=True)
            z = torch.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                graphs[name].replay()
            z.record()
            z.synchronize()
            us = s.elapsed_time(z) * 1000 / (iters * capture_repeats)
            samples[name].append(max(exchange(us, group)))
    return {
        "slowest_rank_us_samples": samples,
        "median_us": {k: statistics.median(v) for k, v in samples.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--capture-repeats", type=int, default=8)
    ap.add_argument("--skip-dma", action="store_true")
    ap.add_argument("--skip-fused", action="store_true")
    ap.add_argument("--fused-tokens", type=int, nargs="+", default=[1, 4, 5, 16])
    args = ap.parse_args()
    os.environ["B12X_PCIE_TP4_REMOTE_PUSH"] = "1"
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    group = dist.new_group(backend="gloo")
    assert dist.get_world_size() == 4
    import b12x.comm.pcie.pcie_oneshot as oneshot_module
    from b12x.comm.pcie import AllReduce, DmaAllReduce

    from vllm import _custom_ops as ops

    root = Path(oneshot_module.__file__).resolve().parents[3]
    result = {
        "command": " ".join(__import__("sys").argv),
        "b12x_commit": os.environ.get("B12X_SOURCE_COMMIT")
        or cmd(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "b12x_worktree": str(root),
        "source_file_sha256": hashlib.sha256(
            Path(oneshot_module.__file__).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "gpu_before": snapshot() if rank == 0 else None,
        "world": 4,
        "hidden": 6144,
        "reset_contract": (
            "Every captured operation resets input; fused arms also reset residual. "
            "Copies included equally in both arms."
        ),
        "environment": {
            k: os.environ[k]
            for k in [
                "B12X_PCIE_TP4_REMOTE_PUSH",
                "CUTE_DSL_ARCH",
                "NCCL_P2P_LEVEL",
                "NCCL_IB_DISABLE",
            ]
        },
        "geometry_overrides": {
            k: os.environ.get(k)
            for k in ["B12X_PCIE_FUSED_THREADS", "B12X_PCIE_FUSED_CTAS_PER_ROW"]
        },
        "points": [],
    }

    def save():
        if rank == 0:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.skip_fused:
        ar = AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda", rank),
            eager_buffer_bytes=16 * 6144 * 2,
            max_size=16 * 6144 * 2,
            rank_data_bytes=1 << 20,
            single_channel=True,
            max_concurrent_channels=1,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cases = []
        # Prepare all variants before capture; live token counts are not keys.
        from b12x.comm.pcie._oneshot_cute import get_fused_oneshot_launcher

        for tokens in args.fused_tokens:
            gen = torch.Generator().manual_seed(20114 + tokens)
            base = (
                (torch.randn(tokens, 6144, generator=gen) * (rank + 1) / 4)
                .bfloat16()
                .cuda()
            )
            res = torch.randn(tokens, 6144, generator=gen).bfloat16().cuda()
            weight = torch.randn(6144, generator=gen).bfloat16().cuda()
            x = torch.empty_like(base)
            r = torch.empty_like(res)
            out = torch.empty_like(base)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ar.prepare_graph_fused_add_rms_norm(x, stream=stream)
                x.copy_(base)
                r.copy_(res)
                ar.all_reduce_fused_add_rms_norm(
                    x, r, weight, 1e-6, out=out, residual_out=r, stream=stream
                )
            stream.synchronize()
            channel = ar.for_stream(stream)
            state = channel._ext._state(channel._ptr)
            transport = channel._ext._fused_launch_config(state, x)
            assert transport[0] == "stage_remote_push", transport
            cases.append((tokens, base, res, weight, x, r, out, transport))
        misses = get_fused_oneshot_launcher.cache_info().misses
        for tokens, base, res, weight, x, r, out, transport in cases:
            torch.accelerator.reset_peak_memory_stats()

            def baseline(x=x, base=base, r=r, res=res, weight=weight):
                x.copy_(base)
                r.copy_(res)
                dist.all_reduce(x)
                ops.fused_add_rms_norm(x, r, weight, 1e-6)

            def fused(x=x, base=base, r=r, res=res, weight=weight, out=out):
                x.copy_(base)
                r.copy_(res)
                ar.all_reduce_fused_add_rms_norm(
                    x, r, weight, 1e-6, out=out, residual_out=r, stream=stream
                )

            with torch.cuda.stream(stream):
                baseline()
            stream.synchronize()
            ref = x.cpu()
            ref_r = r.cpu()
            bg = make_graph(baseline, stream, group, args.capture_repeats)
            fg = make_graph(
                fused,
                stream,
                group,
                args.capture_repeats,
                context=lambda: ar.capture(stream=stream),
            )
            fg.replay()
            torch.accelerator.synchronize()
            checks = {
                "output": cpucheck(out, ref, group),
                "residual": cpucheck(r, ref_r, group),
            }
            timing = time_pair(
                {"nccl_plus_vllm_fused_norm": bg, "b12x_remote_push_fused": fg},
                group,
                args.reps,
                args.iters,
                args.capture_repeats,
            )
            fg.replay()
            torch.accelerator.synchronize()
            checks["after_replays"] = cpucheck(out, ref, group)
            assert get_fused_oneshot_launcher.cache_info().misses == misses, (
                "Unexpected compilation during frozen graph execution"
            )
            point = {
                "kind": "fused_decode",
                "tokens": tokens,
                "bytes": base.numel() * 2,
                "transport": transport,
                "correctness": checks,
                "compiled_launcher_cache_misses": misses,
                "max_torch_allocated_bytes": max(
                    exchange(torch.accelerator.max_memory_allocated(), group)
                ),
                **timing,
            }
            point["speedup_nccl_over_b12x"] = (
                point["median_us"]["nccl_plus_vllm_fused_norm"]
                / point["median_us"]["b12x_remote_push_fused"]
            )
            result["points"].append(point)
            if rank == 0:
                print(json.dumps(point), flush=True)
            save()
            del bg, fg, baseline, fused
        cases.clear()
        ar.close()
        del ar, cases
        torch.accelerator.empty_cache()
    if not args.skip_dma:
        # Two payload buffers; DMA aliases input/output after each reset.
        # BF16 wire, no FP8 compression or accuracy mismatch hidden in the comparison.
        dma = DmaAllReduce(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda", rank),
            max_bytes=4096 * 6144 * 2,
            fp8="off",
        )
        for tokens in [2048, 4096]:
            torch.accelerator.empty_cache()
            torch.accelerator.reset_peak_memory_stats()
            gen = torch.Generator().manual_seed(714 + rank + tokens)
            base = torch.randn(tokens, 6144, generator=gen).bfloat16().cuda()
            buf = torch.empty_like(base)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def baseline(buf=buf, base=base):
                buf.copy_(base)
                dist.all_reduce(buf)

            def candidate(buf=buf, base=base):
                buf.copy_(base)
                dma.all_reduce(buf, out=buf)

            with torch.cuda.stream(stream):
                baseline()
            stream.synchronize()
            ref = buf.cpu()
            bg = make_graph(baseline, stream, group, args.capture_repeats)
            dg = make_graph(candidate, stream, group, args.capture_repeats)
            dg.replay()
            torch.accelerator.synchronize()
            checks = cpucheck(buf, ref, group)
            timing = time_pair(
                {"nccl_bf16": bg, "b12x_copy_engine_bf16": dg},
                group,
                args.reps,
                args.iters,
                args.capture_repeats,
            )
            dg.replay()
            torch.accelerator.synchronize()
            checks["after_replays"] = cpucheck(buf, ref, group)
            raw_scratch = 2 * 3 * dma.shard_capacity
            point = {
                "kind": "dma_prefill",
                "tokens": tokens,
                "bytes": base.numel() * 2,
                "wire_mode": dma._fp8,
                "transport": "copy-engine ring reduce-scatter/all-gather, BF16",
                "inplace_input_output": True,
                "correctness": checks,
                "raw_scratch_bytes_excluding_flags": raw_scratch,
                "max_torch_allocated_bytes": max(
                    exchange(torch.accelerator.max_memory_allocated(), group)
                ),
                **timing,
            }
            point["speedup_nccl_over_b12x"] = (
                point["median_us"]["nccl_bf16"]
                / point["median_us"]["b12x_copy_engine_bf16"]
            )
            result["points"].append(point)
            if rank == 0:
                print(json.dumps(point), flush=True)
            save()
            del bg, dg, base, buf, baseline, candidate
            torch.accelerator.empty_cache()
        dma.close()
    result["gpu_after"] = snapshot() if rank == 0 else None
    save()
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
