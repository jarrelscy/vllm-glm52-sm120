# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual layer3 dense weights, native NVFP4 with four activation planes."""

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(os.environ.get("DENSE_OUTPUT_DIR", "."))
MODEL = Path(os.environ["DENSE_MODEL_DIR"])
INDEX = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def load_weight(suffix):
    def read(suffix):
        key = next(k for k in INDEX if ".layers.3." in k and k.endswith(suffix))
        with safe_open(str(MODEL / INDEX[key]), framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    if "gate_up" in suffix:
        return torch.cat(
            [
                read("mlp.shared_experts." + p + ".weight")[:512]
                for p in ("gate_proj", "up_proj")
            ]
        )
    weight = read(suffix)
    return weight[:, : 4096 if "o_proj" in suffix else 512].contiguous()


def quantize(weight):
    n, k = weight.shape
    block = weight.float().reshape(n, -1, 16)
    maxima = block.abs().amax(-1) / 6
    global_scale = (maxima.max() / 448).clamp_min(1e-12)
    scales = (maxima / global_scale).clamp_min(2**-9).to(torch.float8_e4m3fn)
    norm = block / (scales.float() * global_scale)[:, :, None]
    codes = sum(
        (norm.abs() > threshold).int()
        for threshold in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    )
    codes |= (norm < 0).int() << 3
    levels = torch.tensor(LEVELS, device=weight.device)
    decoded = (levels[codes] * scales.float()[:, :, None] * global_scale).reshape(n, k)
    flat = codes.reshape(n, k).to(torch.uint8)
    packed = (flat[:, ::2] | (flat[:, 1::2] << 4)).contiguous()
    words = (
        packed.view(torch.int32)
        .reshape(1, n // 16, 16, k // 64, 8)
        .permute(0, 1, 3, 2, 4)
    )
    fragment = torch.stack(
        [
            words[
                :, :, :, 8 * (j % 2) : 8 * (j % 2) + 8, 4 * (j // 2) : 4 * (j // 2) + 4
            ].reshape(1, n // 16, k // 64, 32)
            for j in range(4)
        ],
        dim=3,
    ).contiguous()
    return (
        fragment,
        scales.contiguous().view(torch.int32),
        global_scale.reshape(1, 1),
    ), decoded


def reconstruct_activation(value):
    residual = value.float().clone()
    decoded = torch.zeros_like(residual)
    levels = torch.tensor(LEVELS, device=value.device)
    for plane in range(4):
        scaled = residual * (16**plane)
        maxima = scaled.reshape(value.shape[0], -1, 16).abs().amax(-1)
        exponent = torch.ceil(torch.log2((maxima / 6).clamp_min(2**-20))).clamp(-6, 8)
        scale = torch.exp2(exponent).repeat_interleave(16, -1)
        norm = scaled / scale
        codes = sum(
            (norm.abs() > threshold).int()
            for threshold in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        )
        codes |= (norm < 0).int() << 3
        quantized = levels[codes] * scale / (16**plane)
        residual -= quantized
        decoded += quantized
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--oproj-only", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "nvfp4_dense.json"))
    args = parser.parse_args()
    tokens = args.tokens
    torch.manual_seed(913)
    torch.set_num_threads(8)
    capture_stream = torch.cuda.Stream()
    lib = ctypes.CDLL(os.environ["VLLM_ARVQ_KERNEL_LIB"])
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
    cold_id = torch.full((tokens,), -1, device="cuda", dtype=torch.int32)
    hot_id = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    dummy = torch.zeros(384, device="cuda", dtype=torch.int32)

    def ptr(x):
        return ctypes.c_void_p(x.data_ptr())

    def check(code):
        if code:
            raise RuntimeError(f"CUDA error {code}")

    results = []
    for suffix in [
        "self_attn.o_proj.weight",
        "mlp.shared_experts.gate_up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    ]:
        if args.oproj_only and "o_proj" not in suffix:
            continue
        weight = load_weight(suffix)
        n, k = weight.shape
        x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
        base, decoded = quantize(weight)
        base = tuple(t.cuda() for t in base)
        reference = torch.nn.functional.linear(x.cpu().float(), weight.float())
        quantized_reference = torch.nn.functional.linear(
            reconstruct_activation(x.cpu().half()), decoded
        )
        packed_x = torch.empty((tokens, 4, k // 8), device="cuda", dtype=torch.int32)
        scales_x = torch.empty((tokens, 4, k // 16), device="cuda", dtype=torch.uint8)
        out = torch.empty((tokens, n), device="cuda", dtype=torch.float32)
        partial = torch.empty((tokens, n, 16), device="cuda", dtype=torch.float32)

        def native(
            weights,
            split,
            cast=True,
            x=x,
            packed_x=packed_x,
            scales_x=scales_x,
            k=k,
            n=n,
            partial=partial,
            out=out,
        ):
            # Resolve the capture stream at call time, not before graph capture.
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            half_x = x.half()
            check(
                lib.hybrid_pack(
                    ptr(half_x), ptr(packed_x), ptr(scales_x), k, tokens, 4, stream
                )
            )
            hw, hs, hg = weights
            tensors = [
                dummy,
                dummy,
                dummy,
                hw,
                hs,
                hg,
                packed_x,
                scales_x,
                cold_id,
                hot_id,
                partial,
                out,
            ]
            check(
                lib.hybrid_launch(
                    *map(ptr, tensors), 0.0, n, k, tokens, split, 4, 1, stream
                )
            )
            return out.to(x.dtype) if cast else out

        raw = native(base, 8, False).cpu()
        error = float((raw - quantized_reference).norm() / quantized_reference.norm())
        assert error < 2e-5, (suffix, error)
        weight_error = float((decoded - weight.float()).norm() / weight.float().norm())
        output_error = float((raw - reference).norm() / reference.norm())
        del weight, decoded, reference, quantized_reference, raw
        torch.cuda.empty_cache()
        count = math.ceil(272 * 1024**2 / sum(t.nbytes for t in base))
        pool = [base] + [tuple(t.clone() for t in base) for _ in range(count - 1)]

        def measure(weights, split):
            for w in weights:
                native(w, split)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                for w in weights:
                    native(w, split)
            for _ in range(3):
                graph.replay()
            timings = []
            for _ in range(5):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                for _ in range(20):
                    graph.replay()
                b.record()
                b.synchronize()
                timings.append(a.elapsed_time(b) * 1000 / (20 * len(weights)))
            return sorted(timings)[2]

        timings = []
        for split in [s for s in (1, 2, 4, 8, 16) if s <= k // 64]:
            timings.append(
                {
                    "split": split,
                    "warm_us": measure([base], split),
                    "rotating_us": measure(pool, split),
                }
            )
        row = {
            "name": suffix,
            "shape": [n, k],
            "weight_relative_l2": weight_error,
            "output_relative_l2": output_error,
            "kernel_relative_l2_vs_quantized_oracle": error,
            "pool_count": count,
            "pool_bytes": sum(t.nbytes for t in base) * count,
            "bpw": sum(t.nbytes for t in base) * 8 / (n * k),
            "timings": timings,
            "best": min(timings, key=lambda r: r["rotating_us"]),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "tokens": tokens,
                    "scope": (
                        "actual BF16 weights quantized NVFP4; BF16-to-FP16 + "
                        "P4pack + hot native GEMV + split/global reduce + "
                        "BF16cast, no communication"
                    ),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        del pool, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
