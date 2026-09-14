# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight BF16 and existing FP8 W8A16 CUDA graph microbenchmarks."""

import gc
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

root = Path(os.environ.get("DENSE_OUTPUT_DIR", "."))
repository = Path(__file__).resolve().parents[3]
model = Path(os.environ["DENSE_MODEL_DIR"])
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
ext = load(
    name="dense_fp8_w8a16_ext",
    sources=[str(repository / "csrc/quantization/aqlm_moe/fp8_linear_v4.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
torch.manual_seed(913)


def measure(fn, weights, x):
    for w in weights:
        fn(x, *w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in weights:
            _y = fn(x, *w)
    for _ in range(3):
        g.replay()
    timings = []
    for _ in range(5):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(20):
            g.replay()
        b.record()
        b.synchronize()
        timings.append(a.elapsed_time(b) * 1000 / 20 / len(weights))
    del g
    return sorted(timings)[2]


results = []
for suffix in [
    "self_attn.o_proj.weight",
    "mlp.shared_experts.gate_up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
]:
    matches = [k for k in index if ".layers.3." in k and k.endswith(suffix)]
    if not matches and "gate_up" in suffix:
        parts = []
        for p in ["gate_proj", "up_proj"]:
            key = next(
                k
                for k in index
                if ".layers.3." in k
                and k.endswith("mlp.shared_experts." + p + ".weight")
            )
            with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
                parts.append(f.get_tensor(key)[:512])
        wcpu = torch.cat(parts)
    else:
        key = matches[0]
        with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
            wcpu = f.get_tensor(key)
        if "o_proj" in suffix:
            wcpu = wcpu[:, :4096].contiguous()
        elif "down_proj" in suffix:
            wcpu = wcpu[:, :512].contiguous()
        else:
            wcpu = torch.cat(
                [wcpu[:512], wcpu[wcpu.shape[0] // 2 : wcpu.shape[0] // 2 + 512]]
            )
    w = wcpu.cuda()
    N, K = w.shape
    x = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)
    sf = (w.float().abs().amax(1) / 448).clamp_min(1e-8)
    q = (
        (w.float() / sf[:, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )

    def bf(x, w):
        return torch.nn.functional.linear(x, w)

    def fp(x, q, s):
        return ext.fp8_w8a16_gemv((x.float() * (2**-6)).half().contiguous(), q, s).to(
            x.dtype
        )

    bfcompiled = torch.compile(bf, fullgraph=True)
    ref = bf(x, w).float()
    actual = fp(x, q, sf).float()
    deq = torch.nn.functional.linear(
        x.float(), q.view(torch.float8_e4m3fn).float() * sf[:, None]
    )
    row = {
        "name": suffix,
        "shape": [N, K],
        "weight_relative_l2": (
            (q.view(torch.float8_e4m3fn).float() * sf[:, None] - w.float()).norm()
            / w.float().norm()
        ).item(),
        "output_relative_l2": ((actual - ref).norm() / ref.norm()).item(),
        "kernel_relative_l2_vs_dequant": ((actual - deq).norm() / deq.norm()).item(),
    }
    for name, fn, base in [
        ("bf16_eager", bf, (w,)),
        ("bf16_inductor", bfcompiled, (w,)),
        ("fp8_w8a16", fp, (q, sf)),
    ]:
        row[name + "_warm_us"] = measure(fn, [base], x)
        count = math.ceil(
            272 * 1024**2 / sum(z.numel() * z.element_size() for z in base)
        )
        pool = [base] + [tuple(z.clone() for z in base) for _ in range(count - 1)]
        row[name + "_rotating_us"] = measure(fn, pool, x)
        row[name + "_pool_count"] = count
        del pool
        gc.collect()
        torch.cuda.empty_cache()
    results.append(row)
    print(json.dumps(row), flush=True)
    (root / "dense_micro.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "M": 1,
                "pool_min_MiB": 272,
                "results": results,
            },
            indent=2,
        )
    )
    del w, q, sf, x, ref, actual, deq
    gc.collect()
    torch.cuda.empty_cache()
