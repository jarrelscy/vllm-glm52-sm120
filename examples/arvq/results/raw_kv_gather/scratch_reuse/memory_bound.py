# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative explicit-tensor liveness bound, excluding resident/shared storage."""

import json
from pathlib import Path

PART = 4096 * 16 * 512 * 2
LOCAL_Q = 4096 * 16 * 576 * 2
MAP = 4096 * 2048 * 4 + 4096 * 4 + 4096
LSE = 4096 * 16 * 4
BASELINE_PEAK = 635191296


def phases(blocks):
    peer_bytes = blocks * 64 * 656
    gathered = 4 * peer_bytes
    # Include old + newly allocated mapping, all four LSE outputs and their
    # final stack simultaneously, and table even where lifetimes do not overlap.
    allowance = 2 * MAP + 8 * LSE + blocks * 4
    out = {
        "gather": 5 * peer_bytes,
        "absorb_concat": gathered + PART + LOCAL_Q,
    }
    stashed = 0
    for peer in range(4):
        external_with_new_output = peer - stashed + 1
        out[f"attention_peer{peer}"] = (
            gathered + LOCAL_Q + external_with_new_output * PART + allowance
        )
        stashed = min(peer + 1, ((peer + 1) * peer_bytes) // PART)
        out[f"stash_peer{peer}"] = out[f"attention_peer{peer}"]
    out["merge"] = gathered + LOCAL_Q + (4 - stashed) * PART + PART + allowance
    return out


def report():
    rows = []
    for blocks in range(1025, 2049):
        p = phases(blocks)
        rows.append(
            {
                "blocks": blocks,
                "context_min": (blocks - 1) * 256 + 1,
                "context_max": blocks * 256,
                "bound_bytes": max(p.values()),
                "limiting_phase": max(p, key=p.get),
            }
        )
    worst = max(rows, key=lambda r: r["bound_bytes"])
    assert worst["bound_bytes"] < BASELINE_PEAK
    return {
        "scope": "explicit tensor allocations, not CUDA allocator/opaque backend proof",
        "assumptions": [
            "current-stream backend returns fresh contiguous output and separate LSE",
            "backend shared workspace already allocated, "
            "no context-dependent hidden allocation",
            "rank-major gather owns exactly four peer slices; no extra gather copy",
            "all peer buffers use rounded block count, not unrounded context",
            "up to two index/count/empty mappings counted concurrently",
            "all output LSEs plus stack counted in every loop/merge phase",
        ],
        "baseline_measured_peak_bytes": BASELINE_PEAK,
        "worst": worst,
        "minimum_margin_bytes": BASELINE_PEAK - worst["bound_bytes"],
        "all_1024_rounded_block_counts": rows,
    }


if __name__ == "__main__":
    result = report()
    Path(__file__).with_name("memory_bound.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "all_1024_rounded_block_counts"},
            indent=2,
        )
    )
