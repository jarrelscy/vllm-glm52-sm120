# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit FP4-constrained additive RVQ to an existing GLM AQLM dictionary.

This is an AQLM-transcoding feasibility experiment, not donor-weight quantization.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(os.environ.get("ARVQ_FIT_DIR", str(Path.cwd() / "arvq-fits")))
MODEL = Path(
    os.environ.get(
        "ARVQ_SOURCE_MODEL",
        "/data/huggingface/hub/models--jarrelscy--GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m/snapshots/2b883d28bb9dd13a9511e2bd45a8ad1cbacbad74",
    )
)
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def source(layer, projection):
    prefix = f"model.layers.{layer}.mlp.experts."
    family = "w13" if projection == "gateup" else "w2c"
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = prefix + family + "_codebooks"
    with safe_open(MODEL / index[name], framework="pt") as f:
        cb = f.get_tensor(name)[0].float()
        scales = f.get_tensor(prefix + family + "_scales").float()
    return cb, scales, MODEL / index[prefix + family + "_codes"], prefix + family


def nearest(x, c):
    # No N-vector norm needed for argmin. Bounded 65536 x256 workspace.
    return (c.square().sum(1)[None, :] - 2 * x @ c.T).argmin(1)


def means(x, ids, count, previous):
    sums = torch.zeros_like(previous)
    sums.index_add_(0, ids, x)
    counts = torch.bincount(ids, minlength=count).float()
    return torch.where(
        counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], previous
    )


def project(c):
    levels = torch.tensor(LEVELS, device=c.device)
    return levels[(c[:, :, None] - levels).abs().argmin(-1)]


def kmeans(x, count, iterations=12):
    c = x[torch.randperm(x.shape[0], device=x.device)[:count]].clone()
    for _ in range(iterations):
        ids = nearest(x, c)
        c = means(x, ids, count, c)
    return c


def fit(a, beta, iterations=10):
    x = a / beta
    c0 = project(kmeans(x, 256))
    i = nearest(x, c0)
    c1 = project(kmeans(x - c0[i], 128))
    j = nearest(x - c0[i], c1)
    history = []
    best = None
    for step in range(iterations):
        i = nearest(x - c1[j], c0)
        c0 = project(means(x - c1[j], i, 256, c0))
        j = nearest(x - c0[i], c1)
        c1 = project(means(x - c0[i], j, 128, c1))
        error = ((x - c0[i] - c1[j]).square().sum() / x.square().sum()).item()
        history.append(error)
        if best is None or error < best[0]:
            best = (error, c0.clone(), c1.clone(), i.clone(), j.clone())
    error, c0, c1, i, j = best
    # Coordinate refinement for the serialized translation table.
    for _ in range(3):
        i = nearest(x - c1[j], c0)
        j = nearest(x - c0[i], c1)
    error = ((x - c0[i] - c1[j]).square().sum() / x.square().sum()).item()
    return dict(
        beta=beta,
        c0=c0.cpu(),
        c1=c1.cpu(),
        translation=(i | (j << 8)).to(torch.int32).cpu(),
        relative_l2=error**0.5,
        history_mse=history,
        used_first=int(i.unique().numel()),
        used_second=int(j.unique().numel()),
    )


def main():
    global ROOT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(3, 78)))
    ap.add_argument("--source-model", type=Path, default=MODEL)
    ap.add_argument("--fit-dir", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=91426)
    args = ap.parse_args()
    ROOT = args.fit_dir
    MODEL = args.source_model
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = []
    for layer in args.layers:
        for projection in ["gateup", "down"]:
            a, scales, _, _ = source(layer, projection)
            a = a.cuda()
            trials = []
            for beta in [0.5, 0.75, 1.0]:
                torch.cuda.synchronize()
                t = time.perf_counter()
                f = fit(a, beta)
                torch.cuda.synchronize()
                f["fit_seconds"] = time.perf_counter() - t
                trials.append(f)
                print(
                    json.dumps(
                        dict(
                            layer=layer,
                            projection=projection,
                            **{
                                k: v
                                for k, v in f.items()
                                if not isinstance(v, torch.Tensor)
                            },
                        )
                    ),
                    flush=True,
                )
            winner = min(trials, key=lambda x: x["relative_l2"])
            # Global scalar keeps most weight block scales in E4M3's normal range.
            # It is a four-byte metadata field, comfortably inside the 2bpw budget.
            winner["global_scale"] = float(scales.median()) * winner["beta"]
            winner["layer"] = layer
            winner["projection"] = projection
            winner["seed"] = args.seed
            winner["fit_layer_order"] = args.layers
            winner["rng_policy"] = (
                "One torch.manual_seed before all layers; do not "
                "independently reseed layers"
            )
            winner["source"] = "decoded AQLM dictionary; not original donor weights"
            winner["fit_sweep_seconds"] = sum(x["fit_seconds"] for x in trials)
            torch.save(winner, ROOT / f"fit_l{layer}_{projection}.pt")
            report.append(
                {k: v for k, v in winner.items() if not isinstance(v, torch.Tensor)}
            )
            (ROOT / "fit_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
