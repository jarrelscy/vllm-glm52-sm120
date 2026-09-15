# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align target rows by actual token prefix, not speculative step number."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jarrelscy/glm52/dcp-groundtruth/task44")
torch.set_num_threads(4)


def load_rows(arm, rank, probe, prompt_ids, generated):
    rows = {}
    prompt_len = len(prompt_ids)
    for path in sorted((ROOT / "layer-traces" / arm).glob(f"rank{rank}-step*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("probe") != probe:
            continue
        positions = data["positions"].tolist()
        token_ids = data["records"]["input_token_ids"].tolist()
        assert len(positions) == len(token_ids)
        valid_prefix = True
        for index, (position, token) in enumerate(zip(positions, token_ids)):
            if position < prompt_len:
                expected = prompt_ids[position]
            elif position - prompt_len < len(generated):
                expected = int(generated[position - prompt_len])
            else:
                break
            valid_prefix = valid_prefix and token == expected
            if not valid_prefix:
                break
            if position in rows:
                # Repeated speculative verification is preserved, not overwritten.
                rows[position].append((data, index, path.name))
            else:
                rows[position] = [(data, index, path.name)]
    return rows


def leaves(value, path=""):
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        return [
            item
            for index, v in enumerate(value)
            for item in leaves(v, path + f"[{index}]")
        ]
    return []


def compare(a, b, probe):
    root = ROOT / "equivalence"
    x = np.load(root / a / (probe + ".npz"))
    y = np.load(root / b / (probe + ".npz"))
    assert x["prefix_hash"] == y["prefix_hash"]
    common = []
    for tx, ty in zip(x["chosen"], y["chosen"]):
        if tx != ty:
            break
        common.append(int(tx))
    prompts = json.loads((root / "prefixes.json").read_text())
    prompt = prompts["short-p0" if probe.startswith("p0") else "short-p2"]
    results = []
    coverage = []
    for rank in range(4):
        left = load_rows(a, rank, probe, prompt, common)
        right = load_rows(b, rank, probe, prompt, common)
        positions = sorted(left.keys() & right.keys())
        coverage.append(
            {
                "rank": rank,
                "positions": positions,
                "left_only": sorted(left.keys() - right.keys()),
                "right_only": sorted(right.keys() - left.keys()),
            }
        )
        for position in positions:
            ld, li, lf = left[position][0]
            rd, ri, rf = right[position][0]
            for key, value in ld["records"].items():
                if key not in rd["records"]:
                    raise ValueError("Missing capture " + key)
                ls, rs = leaves(value), leaves(rd["records"][key])
                assert len(ls) == len(rs)
                for (suffix, left_value), (rsuffix, right_value) in zip(ls, rs):
                    assert suffix == rsuffix
                    if left_value.ndim == 0 or right_value.ndim == 0:
                        continue
                    assert left_value.shape[0] == len(
                        ld["positions"]
                    ) and right_value.shape[0] == len(rd["positions"]), (
                        key,
                        left_value.shape,
                        right_value.shape,
                    )
                    left_value, right_value = left_value[li], right_value[ri]
                    assert left_value.shape == right_value.shape, (
                        key,
                        left_value.shape,
                        right_value.shape,
                    )
                    if torch.equal(left_value, right_value):
                        continue
                    delta = left_value.double() - right_value.double()
                    row = {
                        "rank": rank,
                        "position": position,
                        "key": key + suffix,
                        "left_trace": lf,
                        "right_trace": rf,
                        "unequal": int((left_value != right_value).sum()),
                        "elements": left_value.numel(),
                        "max_abs": float(delta.abs().max()),
                        "relative_l2": float(
                            delta.norm() / right_value.double().norm().clamp_min(1e-30)
                        ),
                    }
                    if key.endswith((":selected_indices", ":routes:ids")):
                        row["same_sorted_values"] = bool(
                            torch.equal(
                                left_value.sort().values, right_value.sort().values
                            )
                        )
                    results.append(row)
    report = {
        "left": a,
        "right": b,
        "probe": probe,
        "common_generated_tokens": common,
        "coverage": coverage,
        "differences": results,
    }
    output = ROOT / "layer-traces" / f"{a}-vs-{b}-{probe}-rows.json"
    output.write_text(json.dumps(report, indent=2))
    print("Saved", output, "differing tensors", len(results))
    for position in sorted(
        {right_value["position"] for right_value in results if right_value["rank"] == 0}
    ):
        first = next(
            right_value
            for right_value in results
            if right_value["rank"] == 0 and right_value["position"] == position
        )
        print("first differing captured operation", json.dumps(first))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument(
        "--probe", default="p0-traced", choices=["p0-traced", "p2-traced"]
    )
    args = parser.parse_args()
    compare(args.left, args.right, args.probe)
