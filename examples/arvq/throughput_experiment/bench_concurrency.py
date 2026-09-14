# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent serving throughput from actual token usage and overlapping wall span."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

from steady_metrics import steady_decode_summary

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bench_serving as b  # noqa: E402


def document_prompt(base, model, target):
    def make(n):
        intro = (
            "Review the following synthetic operations report. Summarize the major "
            "incidents, distinguish symptoms from likely causes, propose a "
            "prioritized investigation plan, and cite record IDs.\n\n"
        )
        services = ["checkout", "catalog", "billing", "search", "shipping", "accounts"]
        records = []
        for i in range(n):
            observation = (
                "a cache refresh followed by database connection contention"
                if i % 4 == 0
                else "normal upstream responses with occasional queue delays"
            )
            records.append(
                f"Record {i + 1:03d}: The {services[i % 6]} service handled "
                f"{1200 + i * 37} requests during observation window {i + 1}. "
                f"Median latency was {42 + i % 13 * 7} milliseconds and the error "
                f"count was {i % 11}. Operators observed {observation}. "
                f"The deployment revision was r{200 + i // 7}. No data loss was "
                "reported. The team recorded request traces, compared the previous "
                "window, and deferred configuration changes until the source of "
                "the delay could be checked.\n"
            )
        return intro + "\n".join(records)

    n = max(1, target // 95)
    prompt = make(n)
    actual, _ = b.tokenize(base, model, prompt)
    n = max(1, round(n * target / actual))
    prompt = make(n)
    return prompt, b.tokenize(base, model, prompt)


def batch_summary(rows):
    start = min(r["batch_start_offset_s"] for r in rows)
    end = max(r["batch_end_offset_s"] for r in rows)
    total = sum(r["completion_tokens"] for r in rows)
    return {
        "concurrency": len(rows),
        "steady_decode": steady_decode_summary(rows),
        "completed_output_tokens": total,
        "wall_span_s": end - start,
        "aggregate_output_tps": total / (end - start),
        "ttft_ms_per_stream": [r["ttft_s"] * 1000 for r in rows],
        "request_total_s_per_stream": [r["total_s"] for r in rows],
        "per_request_decode_tps": [r["decode_tps"] for r in rows],
        "dispatch_skew_ms": 1000
        * (max(r["batch_start_offset_s"] for r in rows) - start),
        "all_token_ids_match_usage": all(r["token_ids_match_usage"] for r in rows),
    }


def run_batch(base, model, prompt, c, args, clock_path):
    barrier = threading.Barrier(c + 1)
    origin = time.perf_counter()
    clock = None
    clockfile = None
    if args.clocks:
        clockfile = clock_path.open("w")
        clock = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu",
                "--format=csv,noheader",
                "--loop-ms=200",
            ],
            stdout=clockfile,
            stderr=subprocess.DEVNULL,
        )

    def worker(i):
        barrier.wait()
        started = time.perf_counter()
        result, events = b.completion(base, model, prompt, args)
        ended = time.perf_counter()
        result.update(
            stream=i,
            batch_start_offset_s=started - origin,
            batch_end_offset_s=ended - origin,
            sse_events=events,
        )
        return result

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            fs = [pool.submit(worker, i) for i in range(c)]
            barrier.wait()
            rows = [f.result() for f in fs]
    finally:
        if clock:
            clock.terminate()
            clock.wait()
            clockfile.close()
    return rows, batch_summary(rows)


def markdown(report):
    s = (
        f"# {report['label']}\n\nAggregate throughput is total completed output "
        "tokens divided by the concurrent batch wall span, including prefill and "
        "final stream completion. It is not the sum of per-request decode rates. "
        "Each stream uses the same prompt; prefix reuse and identical text are "
        "part of this controlled workload.\n\n| Prompt target | Concurrency | Run "
        "| Output tokens | Batch seconds | Aggregate tokens/s | TTFT ms per stream "
        "| Emitted/draft step |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n"
    )
    for r in report["runs"]:
        sp = r["speculative"]
        em = sp["emitted_per_draft_request_step"]
        ttft = ", ".join(f"{x:.1f}" for x in r["ttft_ms_per_stream"])
        s += (
            f"| {r['target_prompt_tokens']} | {r['concurrency']} | {r['run']} "
            f"| {r['completed_output_tokens']} | {r['wall_span_s']:.3f} "
            f"| {r['aggregate_output_tps']:.3f} | {ttft} "
            f"| {em if em is not None else '—'} |\n"
        )
    return (
        s + "\nSpeculative counters count request-level draft steps, not scheduler "
        "iterations. Dividing summed request decode time by these counters does "
        "not measure a concurrent GPU batch-step latency. Clock CSVs cover each "
        "request batch and include prefill.\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--prompt-kind", choices=["pangram", "document"], default="pangram")
    ap.add_argument("--prompt-lengths", nargs="+", type=int, default=[128])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=173)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--clocks", action="store_true")
    ap.add_argument(
        "--expect-parallel", choices=["tp4-1m", "tp4-1m-mtp"], required=True
    )
    a = ap.parse_args()
    if min(a.concurrency) < 1 or max(a.concurrency) > 8 or a.runs < 1:
        ap.error("Use concurrency1..8 and at least one measured run.")
    container = json.loads(
        subprocess.check_output(
            ["docker", "inspect", "homeassistant-vllm-glm5.3-hybrid-1m-1"]
        )
    )[0]
    env = dict(s.partition("=")[::2] for s in container["Config"]["Env"])
    if (
        env.get("PARALLEL") != a.expect_parallel
        or env.get("VLLM_ENABLE_NVFP4_P4_O_PROJ") != "1"
    ):
        raise RuntimeError("Wrong runtime profile or dense flag; refusing benchmark.")
    if max(a.concurrency) > int(env.get("MAX_NUM_SEQS", "4")):
        raise RuntimeError("Requested concurrency exceeds active max_num_seqs")
    os.environ["OPENAI_API_KEY"] = env.get("VLLM_API_KEY", "")
    base = a.endpoint.rstrip("/")
    b.fetch(base.removesuffix("/v1") + "/health")
    model = "jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid"
    metricurl = base.removesuffix("/v1") + "/metrics"
    report = {
        "label": a.label,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": container["Config"]["Image"],
        "image_id": container["Image"],
        "container_id": container["Id"],
        "compilation_config": next(
            (
                json.loads(container["Config"]["Cmd"][i + 1])
                for i, arg in enumerate(container["Config"]["Cmd"][:-1])
                if arg == "--compilation-config"
            ),
            None,
        ),
        "parallel": env["PARALLEL"],
        "max_model_len": int(env["MAXLEN"]),
        "max_num_seqs": int(env["MAX_NUM_SEQS"]),
        "model": model,
        "dense_p4": True,
        "max_tokens": a.max_tokens,
        "seed": a.seed,
        "ignore_eos": a.ignore_eos,
        "warmup_runs_per_case": a.warmup_runs,
        "prompt_kind": a.prompt_kind,
        "prompt_policy": (
            "Identical prompt for concurrent streams; pangram uses single-stream "
            "reference generator, document uses deterministic synthetic operations "
            "records plus an early per-concurrency discriminator to prevent "
            "cross-case prefix-cache reuse."
        ),
        "safe_environment": {
            k: v
            for k, v in env.items()
            if k
            in [
                "PARALLEL",
                "VLLM_ENABLE_NVFP4_P4_O_PROJ",
                "VLLM_NVFP4_P4_MAX_TOKENS",
                "MAX_NUM_SEQS",
                "ENABLE_LMCACHE",
                "CUDAGRAPH_MODE",
                "MAXLEN",
                "UTIL",
                "VLLM_NVFP4_P4_PAIRED",
                "LMCACHE_LOCAL_CPU",
                "LMCACHE_MAX_LOCAL_CPU_SIZE",
                "LMCACHE_LOCAL_DISK",
                "LMCACHE_MAX_LOCAL_DISK_SIZE",
                "LMCACHE_CHUNK_SIZE",
                "LMCACHE_USE_GPU_CONNECTOR_V3",
                "LMCACHE_PRE_CACHING_HASH_ALGORITHM",
                "PYTHONHASHSEED",
            ]
        },
        "runs": [],
        "warmups": [],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for target in a.prompt_lengths:
        prompt, (actual, method) = (
            document_prompt(base, model, target)
            if a.prompt_kind == "document"
            else b.build_prompt(base, model, target)
        )
        base_prompt = prompt
        for c in a.concurrency:
            if a.prompt_kind == "document":
                prompt = f"Report batch {c}.\n" + base_prompt
                actual, method = b.tokenize(base, model, prompt)
            for run in range(-a.warmup_runs, a.runs):
                before = b.get_metrics(metricurl)
                clock = a.output.with_name(
                    f"{a.label}_p{target}_c{c}_r{run}_clocks.csv"
                )
                rows, summary = run_batch(base, model, prompt, c, a, clock)
                deadline = time.monotonic() + 5
                while True:
                    after = b.get_metrics(metricurl)
                    delta = b.delta_metrics(before, after)
                    if (
                        b.metric_sum(delta, "vllm:request_decode_time_seconds_count")
                        >= c
                        and b.metric_sum(delta, "vllm:generation_tokens")
                        >= summary["completed_output_tokens"]
                    ):
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                draft = b.metric_sum(delta, "vllm:spec_decode_num_drafts")
                accepted = b.metric_sum(delta, "vllm:spec_decode_num_accepted_tokens")
                proposed = b.metric_sum(delta, "vllm:spec_decode_num_draft_tokens")
                count = b.metric_sum(delta, "vllm:request_decode_time_seconds_count")
                gen = b.metric_sum(delta, "vllm:generation_tokens")
                summary.update(
                    run=run,
                    target_prompt_tokens=target,
                    tokenizer_prompt_tokens=actual,
                    tokenizer_method=method,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    requests=rows,
                    metrics_before=before,
                    metrics_after=after,
                    metrics_delta=delta,
                    clock_file=str(clock) if a.clocks else None,
                    speculative={
                        "draft_request_steps": draft,
                        "accepted_tokens": accepted,
                        "proposed_tokens": proposed,
                        "acceptance_fraction": accepted / proposed
                        if proposed
                        else None,
                        "emitted_per_draft_request_step": 1 + accepted / draft
                        if draft
                        else None,
                        "summed_server_request_decode_s": b.metric_sum(
                            delta, "vllm:request_decode_time_seconds_sum"
                        ),
                        "isolation_matches_batch": count == c
                        and gen == summary["completed_output_tokens"],
                    },
                )
                report["warmups" if run < 0 else "runs"].append(summary)
                a.output.write_text(json.dumps(report, indent=2) + "\n")
                a.output.with_suffix(".md").write_text(markdown(report))
                print(
                    json.dumps(
                        {
                            k: summary[k]
                            for k in [
                                "target_prompt_tokens",
                                "concurrency",
                                "run",
                                "completed_output_tokens",
                                "wall_span_s",
                                "aggregate_output_tps",
                                "ttft_ms_per_stream",
                                "speculative",
                            ]
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
