# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request, fresh-cache probes with explicit observer controls."""

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path("/home/jarrelscy/glm52/dcp-groundtruth/task44")
name = sys.argv[1]
assert name.startswith("arvq-task44-")
marker = ROOT / "layer-traces" / name / "enabled"
output = ROOT / "equivalence" / name
output.mkdir(exist_ok=True)
key = os.environ["VLLM_API_KEY"]
prompts = json.loads((ROOT / "equivalence/prefixes.json").read_text())


def request(route, body=None):
    headers = {"Authorization": "Bearer " + key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        "http://localhost:8001" + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        return response.read()


def hits():
    data = request("/metrics").decode()
    return sum(
        float(line.split()[-1])
        for line in data.splitlines()
        if line.startswith("vllm:prefix_cache_hits_total")
    )


report = []
for probe, base, traced in [
    ("p0-untraced", "short-p0", False),
    ("p0-traced", "short-p0", True),
    ("p2-traced", "short-p2", True),
]:
    marker.unlink(missing_ok=True)
    request("/reset_prefix_cache", {})
    before = hits()
    if traced:
        marker.write_text(probe)
    ids = prompts[base]
    started = time.monotonic()
    response = json.loads(
        request(
            "/v1/completions",
            {
                "model": "glm-5.3",
                "prompt": ids,
                "max_tokens": 8,
                "temperature": 0,
                "logprobs": 154880,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        )
    )
    elapsed = time.monotonic() - started
    marker.unlink(missing_ok=True)
    delta = hits() - before
    choice = response["choices"][0]
    rows = []
    for top in choice["logprobs"]["top_logprobs"]:
        row = np.full(154880, -np.inf, dtype=np.float32)
        assert len(top) == 154880
        for token, lp in top.items():
            row[int(token.split(":")[-1])] = lp
        rows.append(row)
    np.savez_compressed(
        output / (probe + ".npz"),
        logprobs=np.stack(rows),
        chosen=np.array(choice["token_ids"]),
        prefix_hash=np.array(hashlib.sha256(json.dumps(ids).encode()).hexdigest()),
        wall_seconds=np.array(elapsed),
    )
    item = {
        "probe": probe,
        "prompt_tokens": len(ids),
        "prefix_cache_hits_delta": delta,
        "wall_seconds": elapsed,
    }
    report.append(item)
    print(name, json.dumps(item), flush=True)
a = np.load(output / "p0-untraced.npz")
b = np.load(output / "p0-traced.npz")
observer = {
    "chosen_exact": bool(np.array_equal(a["chosen"], b["chosen"])),
    "logprobs_exact": bool(np.array_equal(a["logprobs"], b["logprobs"])),
}
(output / "capture_report.json").write_text(
    json.dumps({"probes": report, "observer": observer}, indent=2)
)
print(name, "observer", json.dumps(observer), flush=True)
