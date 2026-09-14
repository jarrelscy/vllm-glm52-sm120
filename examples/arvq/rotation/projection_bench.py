# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialize and execute rotated cold ARVQ with the existing native FP4 MMA."""

# ruff: noqa: B023
# Benchmark closures are executed synchronously before the loop advances.
import argparse
import ctypes
import json
import statistics

import torch
from fit_codebooks import LEVELS
from fit_rotation import ROOT, rotate, source
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def bind(path, name, types):
    f = getattr(ctypes.CDLL(str(path)), name)
    f.argtypes = types
    f.restype = ctypes.c_int
    return f


ptr = ctypes.c_void_p
integer = ctypes.c_int


def pointers(ts):
    return [t.data_ptr() for t in ts]


def checked(result):
    assert result == 0, result


def timing(fn, banks=1):
    for b in range(banks):
        fn(b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for b in range(banks):
            fn(b)
    for _ in range(3):
        g.replay()
    values = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        start.record()
        for _ in range(4):
            g.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (4 * banks))
    return statistics.median(values)


def decoded_source(proj):
    cb, _, path, family = source(3, proj)
    with safe_open(path, framework="pt") as f:
        c = f.get_slice(family + "_codes")
        s = f.get_slice(family + "_scales")
        if proj == "gateup":
            ids = torch.cat([c[2:3, 0:1, :512, :], c[2:3, 0:1, 2048:2560, :]], dim=2)
            scales = torch.cat([s[2:3, :512], s[2:3, 2048:2560]], dim=1)
        else:
            ids = c[2:3, 0:1, :, :64]
            scales = s[2:3, :]
    return (
        cb[ids.long() & 65535].reshape(scales.numel(), -1) * scales.reshape(-1, 1)
    ).cuda()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from assign_api import assign, pack

    oldpack = bind(ROOT / "hybrid.so", "hybrid_pack", [ptr] * 3 + [integer] * 3 + [ptr])
    rotpack = bind(
        ROOT / "rotation.so", "rotation_pack", [ptr] * 5 + [integer] * 3 + [ptr]
    )
    launch = bind(
        ROOT / "hybrid.so",
        "hybrid_launch",
        [ptr] * 12 + [ctypes.c_float] + [integer] * 6 + [ptr],
    )
    torch.set_num_threads(8)
    torch.manual_seed(714)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for proj in ["gateup", "down"]:
        original = decoded_source(proj)
        n, k = original.shape
        split = 16 if proj == "gateup" else 2
        for block in [0, 32, 128]:
            artifact = torch.load(
                ROOT / f"fit_l3_{proj}_b{block}.pt", weights_only=True
            )
            signs = artifact["signs"].cuda()
            w = rotate(original, signs, block)
            global_scale = artifact["global_scale"]
            # Fitted normalization used rms scale then beta. Recover beta.
            beta = artifact["metrics"]["beta"]
            desired = w.reshape(n, -1, 128).square().mean(-1).sqrt() * beta
            scales = (desired / global_scale).clamp(2**-9, 448).to(torch.float8_e4m3fn)
            scaled = (
                (w.reshape(n, -1, 128) / (scales.float() * global_scale)[:, :, None])
                .reshape(-1, 8)
                .contiguous()
            )
            c0 = artifact["c0"].cuda()
            c1 = artifact["c1"].cuda()
            a, b = assign(scaled, c0, c1, refine=3)
            packed = pack(a, b, n, k)
            native_sc = (
                scales.view(torch.uint8)
                .reshape(n // 16, 16, k // 128)
                .permute(0, 2, 1)
                .contiguous()
            )
            levels = torch.tensor(LEVELS, device="cuda")
            cb = torch.cat([c0, c1])
            bits = (cb[:, :, None] - levels).abs().argmin(-1)
            packed_cb = (
                (bits.long() << (torch.arange(8, device="cuda") * 4))
                .sum(-1)
                .to(torch.int32)
            )
            reconstructed = (c0[a.long()] + c1[b.long()]).reshape(n, -1, 128) * (
                scales.float() * global_scale
            )[:, :, None]
            reconstructed = reconstructed.reshape(n, k)
            payload = {
                "packed": packed.cpu().view(torch.int32),
                "codebooks": packed_cb.cpu(),
                "scales": native_sc.cpu(),
                "signs": signs.cpu(),
                "global_scale": torch.tensor(global_scale),
            }
            path = ROOT / f"prototype_l3_{proj}_b{block}.safetensors"
            save_file(
                payload,
                str(path),
                metadata={
                    "format": artifact["format"],
                    "block": str(block),
                    "transform": artifact["transform"],
                    "source": "decoded AQLM expert local2; TP0",
                },
            )
            reopened = load_file(str(path))
            assert all(
                torch.equal(value, reopened[key]) for key, value in payload.items()
            )
            bpw = (
                sum(t.numel() * t.element_size() for t in payload.values())
                * 8
                / (n * k)
            )
            assert bpw < 2
            # >272 MiB payload pool, distinct allocations/addresses; duplicated
            # real quantized weights, so measures cache eviction, not diversity.
            pool_count = (
                int(272 * 1024**2 / (packed.numel() * 4 + native_sc.numel())) // 8 + 1
            ) * 8
            pool_w = packed[:-1].repeat(pool_count)
            pool_w = torch.cat(
                [pool_w, torch.zeros(1, device="cuda", dtype=torch.uint32)]
            )
            pool_s = native_sc.repeat(pool_count, 1, 1, 1).contiguous()
            dummy = torch.zeros(4, device="cuda", dtype=torch.int32)
            hot_scale = torch.ones(1, device="cuda")
            for tokens in [1, 4]:
                slots = tokens * 8
                torch.manual_seed(714 + tokens)
                x = torch.randn(
                    tokens, k, device="cuda", dtype=torch.float16
                ).repeat_interleave(8, 0)
                q = torch.empty(slots, 4, k // 8, device="cuda", dtype=torch.int32)
                qs = torch.empty(slots, 4, k // 16, device="cuda", dtype=torch.uint8)
                partial = torch.empty(slots, n, split, device="cuda")
                out = torch.empty(slots, n, device="cuda")
                hi = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
                ids = [
                    (torch.arange(slots, device="cuda", dtype=torch.int32) % 8 + b * 8)
                    for b in range(pool_count // 8)
                ]

                def run(bank, pool_w=pool_w, pool_s=pool_s):
                    stream = torch.cuda.current_stream().cuda_stream
                    if block:
                        checked(
                            rotpack(
                                *pointers([x, signs, ids[bank], q, qs]),
                                k,
                                slots,
                                block,
                                stream,
                            )
                        )
                    else:
                        checked(oldpack(*pointers([x, q, qs]), k, slots, 4, stream))
                    checked(
                        launch(
                            *pointers(
                                [
                                    pool_w,
                                    packed_cb,
                                    pool_s,
                                    dummy,
                                    dummy,
                                    hot_scale,
                                    q,
                                    qs,
                                    ids[bank],
                                    hi,
                                    partial,
                                    out,
                                ]
                            ),
                            global_scale,
                            n,
                            k,
                            slots,
                            split,
                            4,
                            1,
                            stream,
                        )
                    )

                run(0)
                # Decode actual four-plane operand for independent matmul oracle.
                nib = (
                    q.long()[:, :, :, None] >> (torch.arange(8, device="cuda") * 4)
                ) & 15
                decoded = levels[nib].reshape(slots, 4, k) * qs.view(
                    torch.float8_e4m3fn
                ).float().repeat_interleave(16, -1)
                rx = (
                    decoded
                    / torch.tensor([1, 16, 256, 4096], device="cuda")[None, :, None]
                ).sum(1)
                reference = rx @ reconstructed.T
                error = float((out - reference).norm() / reference.norm())
                assert error < 2e-5, (proj, block, error)
                source_output = x.float() @ original.T
                row = {
                    "projection": proj,
                    "block": block,
                    "tokens": tokens,
                    "slots": slots,
                    "payload_bpw": bpw,
                    "pool_bytes": pool_w.numel() * 4 + pool_s.numel(),
                    "pool_count": pool_count,
                    "kernel_relative_l2": error,
                    "output_relative_l2": float(
                        (out - source_output).norm() / source_output.norm()
                    ),
                    "routing": (
                        "8 distinct cold addresses, shared across token "
                        "positions; identical weight payloads"
                    ),
                    "warm_pack_plus_mma_us": timing(run),
                    "rotating_pack_plus_mma_us": timing(run, len(ids)),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                (ROOT / "projection_results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
            del pool_w, pool_s
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
