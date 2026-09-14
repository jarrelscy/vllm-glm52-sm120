#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create compact, secret-free four-way benchmark aggregates from raw reports."""

import argparse
import json
import pathlib
import statistics


def summarize(reports):
    groups = []
    for report in reports:
        for target in sorted({r["target_prompt_tokens"] for r in report["runs"]}):
            rr = [r for r in report["runs"] if r["target_prompt_tokens"] == target]
            spec = [r.get("speculative", {}) for r in rr]
            rates = [r["decode_tps"] for r in rr if r.get("decode_tps") is not None]
            drafts = sum(s.get("draft_request_steps", 0) for s in spec)
            accepted = sum(s.get("accepted_draft_tokens", 0) for s in spec)
            proposed = sum(s.get("proposed_draft_tokens", 0) for s in spec)
            isoff = "mtp_off" in report["label"]
            isolated = all(
                s.get("isolated_counters_match_request", False) for s in spec
            )
            decode = sum(s.get("server_decode_s") or 0 for s in spec)
            postfirst = sum(r.get("post_first_emission_tokens") or 0 for r in rr)
            step = (
                1000 * decode / drafts
                if isolated and drafts
                else (
                    1000 * decode / postfirst
                    if isolated and isoff and postfirst
                    else None
                )
            )
            emitted = 1 + accepted / drafts if drafts else (1 if isoff else None)
            groups.append(
                {
                    "label": report["label"],
                    "prompt_target": target,
                    "prompt_usage_counts": sorted(
                        {r["usage"]["prompt_tokens"] for r in rr}
                    ),
                    "measured_runs": len(rr),
                    "completion_tokens": sum(r["completion_tokens"] for r in rr),
                    "decode_tps_mean": statistics.mean(rates) if rates else None,
                    "decode_tps_min": min(rates) if rates else None,
                    "decode_tps_max": max(rates) if rates else None,
                    "ttft_ms_mean": statistics.mean(
                        r["ttft_s"] * 1000 for r in rr if r.get("ttft_s") is not None
                    ),
                    "draft_steps_total": drafts,
                    "accepted_draft_tokens_total": accepted,
                    "proposed_draft_tokens_total": proposed,
                    "accepted_per_draft_step": accepted / drafts if drafts else None,
                    "draft_acceptance_fraction": accepted / proposed
                    if proposed
                    else None,
                    "emitted_per_step": emitted,
                    "step_ms_proxy": step,
                    "step_proxy_method": "server decode/drafts"
                    if drafts
                    else (
                        "server decode/post-first-emission-tokens;MTP off"
                        if isoff
                        else None
                    ),
                    "counter_isolation_all_runs": isolated,
                    "ignore_eos": report.get("ignore_eos"),
                    "max_tokens": report.get("max_tokens"),
                    "seed": report.get("seed"),
                    "warmup_runs_per_length": report.get("warmup_runs_per_length"),
                    "prompt_hashes": sorted(
                        {r.get("prompt_sha256") for r in rr if r.get("prompt_sha256")}
                    ),
                }
            )
    for g in groups:
        if "new" in g["label"] and "mtp_on" in g["label"]:
            candidates = [
                r
                for r in groups
                if "original" in r["label"]
                and "mtp_on" in r["label"]
                and r["prompt_target"] == g["prompt_target"]
                and r["counter_isolation_all_runs"]
            ]
            if (
                len(candidates) == 1
                and g["step_ms_proxy"]
                and g["counter_isolation_all_runs"]
            ):
                original = candidates[0]
                matched = all(
                    original[k] == g[k]
                    for k in [
                        "ignore_eos",
                        "max_tokens",
                        "seed",
                        "warmup_runs_per_length",
                        "prompt_hashes",
                    ]
                )
                if matched:
                    g["normalized_to_original_emitted_per_step_tps_estimate"] = (
                        original["emitted_per_step"] * 1000 / g["step_ms_proxy"]
                    )
                    g["normalization_reference"] = original["label"]
                else:
                    g["normalization_error"] = (
                        "Prompt/settings differ from original MTP-on reference."
                    )
    return {
        "groups": groups,
        "notes": [
            (
                "Measured decodeTPS is arithmetic mean of per-request"
                " token-ID-verified rates, excluding first "
                "emission batch."
            ),
            (
                "MTP step duration is pooled server request decode time"
                " / pooled draft request steps, not GPU kernel "
                "latency."
            ),
            (
                "MTP off assumes one forward per token after "
                "first emission, and uses "
                "server decode/post-first tokens."
            ),
            (
                "NormalizedTPS is a counter-derived estimate at "
                "original pooled 1+accepted/drafts; it is not a "
                "measured forced-acceptance rerun."
            ),
            (
                "Throughput on repeated pangram input can reflect "
                "nearly 100% MTP acceptance; do not generalize to "
                "realistic workload quality."
            ),
        ],
    }


def markdown(result):
    s = (
        "# Four-way serving results\n\n| Profile | Prompt target"
        " | Runs | DecodeTPS mean | TTFTms mean | "
        "Emitted/step | Stepms proxy | Draft acceptance | "
        "NewTPS normalized estimate |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )

    def v(n):
        return "—" if n is None else f"{n:.3f}"

    for g in result["groups"]:
        normalized = g.get("normalized_to_original_emitted_per_step_tps_estimate")
        s += (
            f"| {g['label']} | {g['prompt_target']} | {g['measured_runs']} | "
            f"{v(g['decode_tps_mean'])} | {v(g['ttft_ms_mean'])} | "
            f"{v(g['emitted_per_step'])} | {v(g['step_ms_proxy'])} | "
            f"{v(g['draft_acceptance_fraction'])} | {v(normalized)} |\n"
        )
    return s + "\n" + "\n\n".join(result["notes"]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    result = summarize([json.loads(p.read_text()) for p in a.reports])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    a.output.with_suffix(".md").write_text(markdown(result))
    print(a.output)


if __name__ == "__main__":
    main()
