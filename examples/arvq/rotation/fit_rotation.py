# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched rotation ablation on held-out decoded AQLM experts, not donor accuracy."""

import argparse
import json
import time
from pathlib import Path

import torch
from fit_codebooks import fit, nearest, source
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent


def rotate(x, signs, block):
    if not block:
        return x.clone()
    y = (x * signs).reshape(-1, block)
    step = 1
    while step < block:
        v = y.reshape(-1, block // (2 * step), 2, step)
        y = torch.stack(
            (v[:, :, 0] + v[:, :, 1], v[:, :, 0] - v[:, :, 1]), dim=2
        ).reshape(-1, block)
        step *= 2
    return (y / block**0.5).reshape_as(x)


def sample(layer, projection, experts, rows):
    cb, _, path, family = source(layer, projection)
    # Read selected rows only; avoid loading all expert codes.
    weights = []
    with safe_open(path, framework="pt") as f:
        codes = f.get_slice(family + "_codes")
        scales = f.get_slice(family + "_scales")
        for e in experts:
            n = codes.get_shape()[2]
            ids = torch.linspace(0, n - 1, rows).long().tolist()
            c = torch.cat([codes[e : e + 1, 0:1, r : r + 1, :] for r in ids]).reshape(
                rows, -1
            )
            s = torch.cat([scales[e : e + 1, r : r + 1] for r in ids]).reshape(rows, 1)
            w = cb[c.long() & 65535].reshape(rows, -1) * s
            # Down uses TP rank zero's local input partition.
            weights.append(w[:, :512] if projection == "down" else w)
    return torch.cat(weights).cuda()


def normalize(w, global_scale=None):
    # Same true E4M3 block128 scale policy for every transform.
    blocks = w.reshape(-1, 128)
    desired = blocks.square().mean(1).sqrt().clamp_min(1e-12)
    if global_scale is None:
        global_scale = float(desired.median())
    scale = (desired / global_scale).clamp(2**-9, 448).to(
        torch.float8_e4m3fn
    ).float() * global_scale
    return (blocks / scale[:, None]).reshape(-1, 8), scale, global_scale


def assign_all(x, c0, c1):
    result = []
    for chunk in x.split(8192):
        i = nearest(chunk, c0)
        j = nearest(chunk - c0[i], c1)
        for _ in range(3):
            i = nearest(chunk - c1[j], c0)
            j = nearest(chunk - c0[i], c1)
        result.append(c0[i] + c1[j])
    return torch.cat(result)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[3, 40, 77])
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    report = []
    for layer in args.layers:
        for projection in ["gateup", "down"]:
            train = sample(layer, projection, [0, 1], 256)
            test = sample(layer, projection, [2, 3], 128)
            k = train.shape[1]
            gen = torch.Generator().manual_seed(
                91426 + layer * 2 + (projection == "down")
            )
            signs = (
                (torch.randint(0, 2, (k,), generator=gen) * 2 - 1).to(torch.int8).cuda()
            )
            torch.manual_seed(72)
            activations = torch.randn(16, k, device="cuda")
            for block in [0, 32, 128]:
                start = time.perf_counter()
                tw = rotate(train, signs, block)
                vw = rotate(test, signs, block)
                rx = rotate(activations, signs, block)
                exact = rel(vw @ rx.T, test @ activations.T)
                assert exact < 2e-6, exact
                x, _, glob = normalize(tw)
                torch.manual_seed(13)
                subset = x[torch.randperm(len(x), device="cuda")[:32768]]
                trials = []
                for beta in [0.5, 0.75, 1.0]:
                    torch.manual_seed(91426)
                    f = fit(subset, beta, iterations=8)
                    trials.append(f)
                f = min(trials, key=lambda t: t["relative_l2"])
                c0 = f["c0"].cuda()
                c1 = f["c1"].cuda()
                # Fold beta into serialized global; quantize actual stored scales.
                vx, scale, _ = normalize(vw, glob)
                approx = assign_all(vx / f["beta"], c0, c1).reshape(-1, 128)
                reconstructed = (approx * (scale * f["beta"])[:, None]).reshape_as(vw)
                row = {
                    "layer": layer,
                    "projection": projection,
                    "block": block,
                    "weight_relative_l2": rel(reconstructed, vw),
                    "output_relative_l2": rel(
                        reconstructed @ rx.T, test @ activations.T
                    ),
                    "exact_transform_relative_l2": exact,
                    "beta": f["beta"],
                    "train_relative_l2": f["relative_l2"],
                    "elapsed_seconds": time.perf_counter() - start,
                    "train_experts": [0, 1],
                    "test_experts": [2, 3],
                    "train_rows": 512,
                    "test_rows": 256,
                    "fit_vectors": len(subset),
                    "synthetic_activation_rows": 16,
                }
                artifact = {
                    "format": "arvq_rht_v1" if block else "arvq_matched_v1",
                    "transform": "input_signed_normalized_hadamard"
                    if block
                    else "identity",
                    "block": block,
                    "signs": signs.cpu() if block else torch.empty(0, dtype=torch.int8),
                    "global_scale": glob * f["beta"],
                    "c0": f["c0"],
                    "c1": f["c1"],
                    "scale_group": 128,
                    "index_bits": [8, 7],
                    "vector_size": 8,
                    "source": "decoded AQLM; no original donor; uncalibrated",
                    "metrics": row,
                }
                torch.save(artifact, ROOT / f"fit_l{layer}_{projection}_b{block}.pt")
                report.append(row)
                (ROOT / "fit_results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(row), flush=True)
            del train, test
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
