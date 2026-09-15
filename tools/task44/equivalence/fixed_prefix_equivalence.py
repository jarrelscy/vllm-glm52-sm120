# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary comparisons at fixed token prefixes across serving arms."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/jarrelscy/homeassistant/benchmarks")
from token_dist_ab import needle_doc, traj_messages

ROOT = Path("/home/jarrelscy/glm52/dcp-groundtruth/task44/equivalence")
ROOT.mkdir(exist_ok=True)
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = "glm-5.3"


def post(route, body):
    r = urllib.request.Request(
        "http://localhost:8001" + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    with urllib.request.urlopen(r, timeout=14400) as f:
        return json.load(f)


def tokenize(messages):
    return post(
        "/tokenize", dict(model=MODEL, messages=messages, add_generation_prompt=True)
    )["tokens"]


def completion(ids, full=True):
    started = time.monotonic()
    result = post(
        "/v1/completions",
        dict(
            model=MODEL,
            prompt=ids,
            max_tokens=8,
            temperature=0,
            logprobs=154880 if full else 20,
            return_token_ids=True,
            return_tokens_as_token_ids=True,
        ),
    )
    result["_task44_wall_seconds"] = time.monotonic() - started
    return result


def seed():
    short = tokenize(traj_messages("torch-tensor-parallelism", 1))
    chosen = completion(short, False)["choices"][0]["token_ids"]
    prompts = {
        "short-p0": short,
        "short-p2": short + chosen[:2],
        "short-p3": short + chosen[:3],
    }
    prompts["trajectory26k"] = tokenize(traj_messages("torch-tensor-parallelism", 19))
    doc, _ = needle_doc(180000, seed=444)
    long = tokenize(
        [
            dict(
                role="user",
                content=doc + "\nSummarize the document and list the secret codewords.",
            )
        ]
    )
    # Keep the assistant generation header at the end while setting an exact length.
    assert len(long) > 131000, len(long)
    prompts["long131k"] = long[:130968] + long[-32:]
    (ROOT / "prefixes.json").write_text(json.dumps(prompts))
    print("seeded", {k: len(v) for k, v in prompts.items()}, flush=True)


def save(tag, name, ids, response):
    choice = response["choices"][0]
    tops = choice["logprobs"]["top_logprobs"]
    rows = []
    for top in tops:
        parsed = {int(k.split(":")[-1]): v for k, v in top.items()}
        assert len(parsed) == 154880, f"Not full vocabulary: {len(parsed)}"
        row = np.full(max(parsed) + 1, -np.inf, dtype=np.float32)
        for token, lp in parsed.items():
            row[token] = lp
        rows.append(row)
    path = ROOT / tag
    path.mkdir(exist_ok=True)
    np.savez_compressed(
        path / (name + ".npz"),
        logprobs=np.stack(rows),
        chosen=np.array(choice["token_ids"]),
        prefix_hash=np.array(hashlib.sha256(json.dumps(ids).encode()).hexdigest()),
        wall_seconds=np.array(response.get("_task44_wall_seconds", float("nan"))),
    )
    print(
        tag,
        name,
        "prompt",
        len(ids),
        "steps",
        len(rows),
        "vocab",
        len(rows[0]),
        flush=True,
    )


def capture(tag, scope):
    prompts = json.loads((ROOT / "prefixes.json").read_text())
    if scope == "prod":
        prompts = {k: v for k, v in prompts.items() if k != "long131k"}
    if scope == "short":
        prompts = {k: v for k, v in prompts.items() if k.startswith("short")}
    for name, ids in prompts.items():
        if name == "short-p0":
            subprocess.run(
                ["docker", "exec", tag, "touch", f"/opt/task44-traces/{tag}/enabled"],
                check=True,
            )
        save(tag, name, ids, completion(ids))
        if name == "short-p0":
            subprocess.run(
                [
                    "docker",
                    "exec",
                    tag,
                    "rm",
                    "-f",
                    f"/opt/task44-traces/{tag}/enabled",
                ],
                check=True,
            )
    if scope != "short":
        subprocess.run(
            ["docker", "exec", tag, "touch", f"/opt/task44-traces/{tag}/enabled"],
            check=True,
        )
        names = [
            "short-p0",
            "trajectory26k",
            "short-p2",
            "trajectory26k" if scope == "prod" else "long131k",
            "short-p3",
            "trajectory26k",
            "short-p0",
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            futures = [
                (i, n, pool.submit(completion, prompts[n])) for i, n in enumerate(names)
            ]
            for i, n, f in futures:
                save(tag, f"batch7-{i}-{n}", prompts[n], f.result())
        subprocess.run(
            ["docker", "exec", tag, "rm", "-f", f"/opt/task44-traces/{tag}/enabled"],
            check=True,
        )


def compare(a, b):
    report = []
    for f in sorted((ROOT / a).glob("*.npz")):
        other = ROOT / b / f.name
        if not other.exists():
            continue
        x, y = np.load(f), np.load(other)
        assert x["prefix_hash"] == y["prefix_hash"]
        for i in range(min(len(x["chosen"]), len(y["chosen"]))):
            if i and x["chosen"][i - 1] != y["chosen"][i - 1]:
                break
            lx, ly = (
                x["logprobs"][i].astype("float64"),
                y["logprobs"][i].astype("float64"),
            )
            px, py = np.exp(lx), np.exp(ly)
            px /= px.sum()
            py /= py.sum()
            finite = np.isfinite(lx) & np.isfinite(ly)
            kl = float(
                np.sum(
                    px[finite]
                    * (
                        (lx[finite] - np.log(np.exp(lx).sum()))
                        - (ly[finite] - np.log(np.exp(ly).sum()))
                    )
                )
            )
            report.append(
                dict(
                    probe=f.stem,
                    step=i,
                    chosen_a=int(x["chosen"][i]),
                    chosen_b=int(y["chosen"][i]),
                    tv=float(np.abs(px - py).sum() / 2),
                    kl=kl,
                    max_lp_delta=float(np.abs(lx[finite] - ly[finite]).max()),
                    prob_sum_a=float(np.exp(lx).sum()),
                    prob_sum_b=float(np.exp(ly).sum()),
                )
            )
    (ROOT / f"{a}-vs-{b}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


p = argparse.ArgumentParser()
p.add_argument("mode", choices=["seed", "capture", "compare"])
p.add_argument("tags", nargs="*")
p.add_argument("--scope", choices=["short", "all", "prod"], default="all")
a = p.parse_args()
if a.mode == "seed":
    seed()
elif a.mode == "capture":
    capture(a.tags[0], a.scope)
else:
    compare(*a.tags)
