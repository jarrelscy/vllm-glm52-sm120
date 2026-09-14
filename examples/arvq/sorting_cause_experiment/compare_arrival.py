# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare independent HTTP arrival with one multi-prompt completion request."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "throughput_experiment"))
import bench_concurrency as concurrent_bench  # noqa: E402
import bench_serving as b  # noqa: E402
from steady_metrics import steady_decode_summary  # noqa: E402

MODEL = "jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid"
CONTAINER = "homeassistant-vllm-glm5.3-hybrid-1m-1"


def split_choices(events, elapsed, count, max_tokens):
    """Final usage is aggregate; token IDs determine each choice's exact share."""
    usage = next((e["usage"] for _, e in reversed(events) if e.get("usage")), None)
    if not usage or usage["completion_tokens"] != count * max_tokens:
        raise ValueError("Missing or unexpected aggregate token usage")
    per_choice = [[] for _ in range(count)]
    totals = [0] * count
    for when, event in events:
        if event.get("error"):
            raise ValueError("Server returned an error")
        for choice in event.get("choices", []):
            index = choice["index"]
            if not 0 <= index < count:
                raise ValueError("Unexpected choice index")
            totals[index] += len(choice.get("token_ids") or [])
            per_choice[index].append((when, {"choices": [dict(choice, index=0)]}))
    if totals != [max_tokens] * count or sum(totals) != usage["completion_tokens"]:
        raise ValueError("Per-choice token IDs disagree with aggregate usage")
    rows = []
    for index, choice_events in enumerate(per_choice):
        # Per-choice counts are reconstructed from IDs and verified against the
        # authoritative aggregate usage and requested ignore-eos output length.
        choice_events.append((elapsed, {"usage": {"completion_tokens": totals[index]}}))
        row = b.stream_result(choice_events, elapsed)
        row.update(
            stream=index,
            batch_start_offset_s=0.0,
            batch_end_offset_s=elapsed,
            sse_events=choice_events,
        )
        rows.append(row)
    return rows, usage


def batched_request(base, prompt, args):
    body = {
        "model": MODEL,
        "prompt": [prompt] * 8,
        "n": 1,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": args.seed,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/completions", data=json.dumps(body).encode(), headers=b.headers()
    )
    start = time.perf_counter()
    events = []
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            events.append((time.perf_counter() - start, json.loads(payload)))
    elapsed = time.perf_counter() - start
    rows, usage = split_choices(events, elapsed, 8, args.max_tokens)
    return rows, {"aggregate_usage": usage, "raw_sse_events": events}


def output_details(rows):
    hashes, starts, texts = [], [], []
    for row in rows:
        ids = [
            token
            for _, event in row["sse_events"]
            for choice in event.get("choices", [])
            for token in (choice.get("token_ids") or [])
        ]
        hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        starts.append(row["batch_start_offset_s"] + row["ttft_s"])
        texts.append(
            "".join(
                choice.get("text", "")
                for _, event in row["sse_events"]
                for choice in event.get("choices", [])
            )
        )
    return {
        "output_sha256_per_stream": hashes,
        "distinct_output_sequences": len(set(hashes)),
        "first_emission_spread_s": max(starts) - min(starts),
        "first_emission_offset_s": starts,
        "output_texts": texts,
        "steady_decode": steady_decode_summary(rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--output", type=pathlib.Path, default=HERE / "arrival.json")
    p.add_argument("--endpoint", default="http://localhost:8001/v1")
    args = p.parse_args()
    args.max_tokens, args.seed, args.timeout = 512, 173, 900
    args.ignore_eos, args.clocks = True, False
    info = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER]))[0]
    env = dict(x.split("=", 1) for x in info["Config"]["Env"])
    expected = {
        "MAXLEN": "1048576",
        "MAX_NUM_SEQS": "8",
        "ENABLE_LMCACHE": "1",
        "VLLM_NVFP4_P4_MAX_TOKENS": "16",
        "PARALLEL": "tp4-1m-mtp",
    }
    if any(env.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Refusing mismatched stable profile")
    os.environ["OPENAI_API_KEY"] = env.get("VLLM_API_KEY", "")
    base = args.endpoint.rstrip("/")
    b.fetch(base.removesuffix("/v1") + "/health")
    prompt, (actual, method) = b.build_prompt(base, MODEL, 128)
    if actual != 136:
        raise RuntimeError(f"Expected 136 input tokens, got {actual}")
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": info["Config"]["Image"],
        "image_id": info["Image"],
        "safe_environment": {k: env.get(k) for k in expected},
        "prompt": prompt,
        "prompt_tokens": actual,
        "tokenizer_method": method,
        "method": "Four alternating pairs with balanced pair order. Independent "
        "8 HTTP requests versus one prompt-array request containing "
        "8 identical strings, n=1. No warmup/configuration changes. "
        "Same 136 input/512 output tokens, temperature0, seed173.",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pair in range(args.pairs):
        order = (
            ["independent", "batched"] if pair % 2 == 0 else ["batched", "independent"]
        )
        for arrival in order:
            label = f"arrival_{pair}_{arrival}"
            clock_path = args.output.parent / (label + "_thermal.csv")
            before = b.get_metrics(base.removesuffix("/v1") + "/metrics")
            with clock_path.open("w") as clock_file:
                clock = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,clocks.current.sm,"
                        "clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,"
                        "clocks_event_reasons.sw_thermal_slowdown,"
                        "clocks_event_reasons.hw_thermal_slowdown,"
                        "clocks_event_reasons.sw_power_cap",
                        "--format=csv,noheader",
                        "--loop-ms=200",
                    ],
                    stdout=clock_file,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    if arrival == "independent":
                        rows, summary = concurrent_bench.run_batch(
                            base, MODEL, prompt, 8, args, clock_path
                        )
                        extra = {"batch_summary": summary}
                    else:
                        rows, extra = batched_request(base, prompt, args)
                finally:
                    clock.terminate()
                    clock.wait()
            for _ in range(50):
                after = b.get_metrics(base.removesuffix("/v1") + "/metrics")
                delta = b.delta_metrics(before, after)
                count = b.metric_sum(delta, "vllm:request_decode_time_seconds_count")
                if count >= 8:
                    break
                time.sleep(0.1)
            generated = b.metric_sum(delta, "vllm:generation_tokens")
            proposed = b.metric_sum(delta, "vllm:spec_decode_num_draft_tokens")
            accepted = b.metric_sum(delta, "vllm:spec_decode_num_accepted_tokens")
            result = {
                "pair": pair,
                "arrival": arrival,
                "rows": rows,
                **extra,
                **output_details(rows),
                "metrics_before": before,
                "metrics_after": after,
                "metrics_delta": delta,
                "server_request_count": count,
                "server_output_tokens": generated,
                "draft_proposed": proposed,
                "draft_accepted": accepted,
                "acceptance_fraction": accepted / proposed if proposed else None,
                "clock_file": str(clock_path),
            }
            report["runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if count != 8 or generated != 4096:
                raise RuntimeError(
                    "Batch was not isolated: expected8requests/4096outputs"
                )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in [
                            "pair",
                            "arrival",
                            "steady_decode",
                            "first_emission_spread_s",
                            "distinct_output_sequences",
                            "acceptance_fraction",
                        ]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
