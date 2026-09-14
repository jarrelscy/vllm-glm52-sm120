# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import gc
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from arvq8x8_data import LEVELS, decoded_weights, load, reconstructed_activations

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("ARVQ8X8_OUTPUT_DIR", str(ROOT)))
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(51426)
torch.backends.cuda.matmul.allow_tf32 = False
libs = [
    ctypes.CDLL(
        os.environ.get(
            "ARVQ_BASE_KERNEL_LIB",
            str(
                ROOT.parents[2]
                / "vllm/model_executor/layers/quantization/arvq/hybrid.so"
            ),
        )
    ),
    ctypes.CDLL(str(ROOT / "hybrid_8x8.so")),
]
for lib in libs:
    lib.hybrid_launch.argtypes = (
        [ctypes.c_void_p] * 12
        + [ctypes.c_float]
        + [ctypes.c_int] * 6
        + [ctypes.c_void_p]
    )
    lib.hybrid_launch.restype = ctypes.c_int
    lib.hybrid_pack.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    lib.hybrid_pack.restype = ctypes.c_int
libs[1].expand_indices.argtypes = [ctypes.c_void_p] * 2 + [
    ctypes.c_longlong,
    ctypes.c_int,
    ctypes.c_void_p,
]
libs[1].expand_indices.restype = ctypes.c_int


def ptrs(ts):
    return [ctypes.c_void_p(t.data_ptr()) for t in ts]


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def check(err):
    assert err == 0, err


def routes(tokens, mode, bank, mixed):
    cold = []
    hot = []
    for t in range(tokens):
        offset = t if mode == "unique" else 0
        nc = 6 if mixed else 8
        cold.extend(
            [(bank * 11 + 17 + offset * nc + j) % 195 for j in range(nc)]
            + [-1] * (8 - nc)
        )
        hot.extend(
            [-1] * nc + [(bank * 7 + 9 + offset * 2 + j) % 61 for j in range(8 - nc)]
        )
    return tuple(torch.tensor(v, device="cuda", dtype=torch.int32) for v in (cold, hot))


def buffers(slots, n, k, split):
    return [
        torch.empty(shape, device="cuda", dtype=dtype)
        for shape, dtype in [
            ((slots, 4, k // 8), torch.int32),
            ((slots, 4, k // 16), torch.uint8),
            ((slots, n, split), torch.float32),
            ((slots, n), torch.float32),
        ]
    ]


def projection(d, x, ids, b, variant, split):
    q, s, part, out = b
    check(libs[0].hybrid_pack(*ptrs([x, q, s]), x.shape[1], len(x), 4, stream()))
    cw = (
        d["cold_packed"]
        if variant == 0
        else d["packed_matched"]
        if variant == 1
        else d["packed_full"]
    )
    cb = d["cold_cb"] if variant == 0 else d["cb_full"]
    check(
        libs[int(variant > 0)].hybrid_launch(
            *ptrs(
                [
                    cw,
                    cb,
                    d["cold_scales"],
                    d["hot_fragment"],
                    d["hot_native_scales"],
                    d["hot_scale2"],
                    q,
                    s,
                    *ids,
                    part,
                    out,
                ]
            ),
            d["alpha"],
            d["N"],
            d["K"],
            len(x),
            split,
            4,
            d["hot_scale2"].shape[1],
            stream(),
        )
    )
    return out


def measurements(call_sets):
    graphs = []
    for calls in call_sets:
        for fn in calls:
            fn()
    torch.accelerator.synchronize()
    for calls in call_sets:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for fn in calls:
                fn()
        graphs.append(graph)
    for _ in range(3):
        for graph in graphs:
            graph.replay()
    result = [[] for _ in graphs]
    # Alternate order every round, same warm-up and loop count for all variants.
    for repeat in range(8):
        for v in (
            range(len(graphs)) if repeat % 2 == 0 else reversed(range(len(graphs)))
        ):
            a, b = (
                torch.Event(device="cuda", enable_timing=True),
                torch.Event(device="cuda", enable_timing=True),
            )
            a.record()
            for _ in range(2):
                graphs[v].replay()
            b.record()
            b.synchronize()
            result[v].append(a.elapsed_time(b) * 1000 / (2 * len(call_sets[v])))
    return [{"median_us": statistics.median(r), "samples_us": r} for r in result]


def decode_full(d, expert):
    n, k = d["N"], d["K"]
    tiles = n // 16
    groups = k // 64
    words = d["packed_full"].reshape(195, tiles, groups, 64)[expert].long()
    ids = torch.stack((words & 65535, (words >> 16) & 65535), -1).reshape(
        tiles, groups, 4, 32
    )
    codes0 = d["cb_full"][:256].long()[ids & 255]
    codes1 = d["cb_full"][256:].long()[ids >> 8]
    shifts = torch.arange(8, device="cuda") * 4
    levels = torch.tensor(LEVELS, device="cuda")
    f = (
        levels[(codes0[..., None] >> shifts) & 15]
        + levels[(codes1[..., None] >> shifts) & 15]
    )
    natural = torch.empty(tiles, 16, groups, 64, device="cuda")
    for j in range(4):
        block = (
            f[:, :, j]
            .reshape(tiles, groups, 8, 4, 8)
            .reshape(tiles, groups, 8, 32)
            .permute(0, 2, 1, 3)
        )
        natural[
            :, 8 * (j % 2) : 8 * (j % 2) + 8, :, 32 * (j // 2) : 32 * (j // 2) + 32
        ] = block
    scale = (
        d["cold_scales"][expert]
        .permute(0, 2, 1)
        .reshape(n, k // 128)
        .view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(128, -1)
    )
    return natural.reshape(n, k) * scale * d["alpha"]


def clocks():
    return (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ]
        )
        .decode()
        .strip()
    )


data = load()
footprint = {}
for name, d in data.items():
    d["alpha"] = float(d["cold_global"])
    n, k = d["N"], d["K"]
    words = 195 * (n // 16) * (k // 64) * 64
    d["packed_matched"] = torch.empty(words, device="cuda", dtype=torch.uint32)
    d["packed_full"] = torch.empty_like(d["packed_matched"])
    for full, key in [(0, "packed_matched"), (1, "packed_full")]:
        check(
            libs[1].expand_indices(
                *ptrs([d["cold_packed"], d[key]]), words, full, stream()
            )
        )
    # Upper128 codewords are random valid FP4 vectors: speed/correctness only,
    # no claim of an optimized256-entry residual dictionary or model quality.
    extra = torch.randint(0, 2**32, (128,), device="cuda", dtype=torch.int64).to(
        torch.uint32
    )
    d["cb_full"] = torch.cat([d["cold_cb"], extra.view(torch.int32)])
    oldbytes = (
        d["cold_packed"].nbytes + d["cold_scales"].nbytes + d["cold_cb"].nbytes + 4
    )
    newbytes = (
        d["packed_matched"].nbytes + d["cold_scales"].nbytes + d["cb_full"].nbytes + 4
    )
    footprint[name] = {
        "shape": [n, k],
        "experts": 195,
        "old_bytes": oldbytes,
        "new_bytes": newbytes,
        "old_bpw": oldbytes * 8 / (195 * n * k),
        "new_bpw": newbytes * 8 / (195 * n * k),
    }
    # Full256-index independent decoder oracle on one cold expert each projection.
    cid = torch.tensor([17], device="cuda", dtype=torch.int32)
    hid = torch.full_like(cid, -1)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    bf = buffers(1, n, k, 8 if name == "gateup" else 2)
    actual = projection(d, x, (cid, hid), bf, 2, 8 if name == "gateup" else 2)
    expected = reconstructed_activations(x) @ decode_full(d, 17).T
    error = float((actual - expected).norm() / expected.norm())
    assert error < 2e-5, error
    footprint[name]["full256_oracle_l2"] = error
    del expected, actual, bf
    gc.collect()
    torch.accelerator.empty_cache()
report = {
    "footprint": footprint,
    "scope": (
        "Actual layer3 TP3 weights, synthetic FP16 inputs;8+7 versus matched "
        "8+8 (<128 residual IDs) versus full256 synthetic upper codewords. "
        "Same P4/two MMAs/splits/order. Eight interleaved alternating-order "
        "samples, identical warmups."
    ),
    "clock_start": clocks(),
    "rows": [],
}
for tokens in [1, 4, 8, 32]:
    for mixed in [True, False]:
        for mode in ["reused", "unique"] if tokens > 1 else ["reused"]:
            slots = tokens * 8
            sg = 16 if slots <= 32 else 8
            sd = 2
            x = torch.randn(tokens, 6144, device="cuda", dtype=torch.float16)
            rw = torch.softmax(torch.randn(tokens, 8, device="cuda"), -1)
            banks = [routes(tokens, mode, bank, mixed) for bank in range(48)]
            bg = buffers(slots, 1024, 6144, sg)
            bd = buffers(slots, 6144, 512, sd)

            def pipeline(ids, v, x=x, bg=bg, sg=sg, bd=bd, sd=sd, tokens=tokens, rw=rw):
                y = projection(
                    data["gateup"], x.repeat_interleave(8, 0), ids, bg, v, sg
                ).half()
                act = torch.nn.functional.silu(y[:, :512]) * y[:, 512:]
                down = projection(data["down"], act, ids, bd, v, sd)
                return (down.reshape(tokens, 8, 6144) * rw[:, :, None]).sum(1)

            px0 = x.repeat_interleave(8, 0)
            g0 = projection(data["gateup"], px0, banks[0], bg, 0, sg).clone()
            g1 = projection(data["gateup"], px0, banks[0], bg, 1, sg).clone()
            assert torch.equal(g0.view(torch.int32), g1.view(torch.int32))
            gh = g0.half()
            act0 = torch.nn.functional.silu(gh[:, :512]) * gh[:, 512:]
            d0 = projection(data["down"], act0, banks[0], bd, 0, sd).clone()
            d1 = projection(data["down"], act0, banks[0], bd, 1, sd).clone()
            assert torch.equal(d0.view(torch.int32), d1.view(torch.int32))
            baseline = pipeline(banks[0], 0).clone()
            matched = pipeline(banks[0], 1).clone()
            assert torch.equal(baseline.view(torch.int32), matched.view(torch.int32)), (
                tokens,
                mixed,
                mode,
            )
            # Original independent decoded-weight oracle on both matched projections.
            ci, hi = [z[:8] for z in banks[0]]
            w = decoded_weights(data["gateup"], ci, hi, True)
            qx = reconstructed_activations(x.repeat_interleave(8, 0)[:8])
            oracle = torch.bmm(w, qx.unsqueeze(-1)).squeeze(-1)
            actual = projection(
                data["gateup"], x.repeat_interleave(8, 0), banks[0], bg, 1, sg
            )[:8]
            err = float((actual - oracle).norm() / oracle.norm())
            assert err < 2e-5, err
            del w, qx, oracle, actual
            gc.collect()
            torch.accelerator.empty_cache()
            row = {
                "tokens": tokens,
                "slots": slots,
                "routes": "6cold2hot" if mixed else "8cold",
                "routing": mode,
                "matched_bitwise_equal": True,
                "matched_gate_oracle_l2": err,
                "split_gateup": sg,
                "split_down": sd,
            }
            for label, ids_set in [("warm", [banks[0]] * 48), ("rotating", banks)]:
                row[label] = measurements(
                    [
                        [lambda ids=ids, v=v: pipeline(ids, v) for ids in ids_set]
                        for v in range(3)
                    ]
                )
            if not mixed:
                for proj, b, k, n, sp in [
                    ("gateup", bg, 6144, 1024, sg),
                    ("down", bd, 512, 6144, sd),
                ]:
                    px = torch.randn(slots, k, device="cuda", dtype=torch.float16)
                    for label, ids_set in [
                        ("warm", [banks[0]] * 48),
                        ("rotating", banks),
                    ]:
                        row[proj + "_" + label] = measurements(
                            [
                                [
                                    lambda ids=ids,
                                    v=v,
                                    proj=proj,
                                    px=px,
                                    b=b,
                                    sp=sp: projection(data[proj], px, ids, b, v, sp)
                                    for ids in ids_set
                                ]
                                for v in range(3)
                            ]
                        )
            report["rows"].append(row)
            report["clock_end"] = clocks()
            (OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "mixed": mixed,
                        "routing": mode,
                        "warm": [z["median_us"] for z in row["warm"]],
                        "rotating": [z["median_us"] for z in row["rotating"]],
                    }
                ),
                flush=True,
            )
print("COMPLETE", flush=True)
