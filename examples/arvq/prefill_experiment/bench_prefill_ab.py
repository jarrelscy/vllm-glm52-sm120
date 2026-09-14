# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache, identical-token prefill A/B; no model restart or profiler."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bench_serving as b

CONTAINER = "homeassistant-vllm-glm5.3-hybrid-1m-1"
GATE = "/dev/shm/vllm_arvq_grouped_prefill_on"
MODEL = "jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid"


def set_mode(container, enabled, gate=GATE):
    subprocess.run(
        [
            "docker",
            "exec",
            container,
            "touch" if enabled else "rm",
            *([] if enabled else ["-f"]),
            gate,
        ],
        check=True,
    )
    present = (
        subprocess.run(["docker", "exec", container, "test", "-e", gate]).returncode
        == 0
    )
    if present != enabled:
        raise RuntimeError("Grouped-prefill gate did not match requested mode")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", default=CONTAINER)
    p.add_argument(
        "--gate",
        choices=[
            GATE,
            "/dev/shm/vllm_arvq_compact_prefill_on",
            "/dev/shm/vllm_arvq_sort_native_prefill_on",
        ],
        default=GATE,
    )
    p.add_argument(
        "--toggle-env",
        choices=[
            "VLLM_ARVQ_GROUPED_PREFILL",
            "VLLM_ARVQ_COMPACT_PREFILL",
            "VLLM_ARVQ_SORT_NATIVE_PREFILL",
        ],
        default="VLLM_ARVQ_GROUPED_PREFILL",
    )
    p.add_argument("--base", default="http://localhost:8001")
    p.add_argument(
        "--capture",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent
        / "results/prefill/grouped_prefill_capture.json",
    )
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--lengths", nargs="+", type=int, default=[4096])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--arm-boot-only",
        action="store_true",
        help="Enable gate immediately after compose up, before memory profiling.",
    )
    a = p.parse_args()
    if a.arm_boot_only:
        set_mode(a.container, True, a.gate)
        print("Grouped-prefill boot memory gate enabled.", flush=True)
        return
    info = json.loads(subprocess.check_output(["docker", "inspect", a.container]))[0]
    env = dict(x.split("=", 1) for x in info["Config"]["Env"])
    if env.get(a.toggle_env) != "toggle":
        raise RuntimeError("Expected selected toggle environment to equal toggle")
    if env.get("PARALLEL") != "tp4-1m-mtp":
        raise RuntimeError("Expected MTP enabled for controlled comparison")
    os.environ["OPENAI_API_KEY"] = env.get("VLLM_API_KEY", "")
    b.fetch(a.base + "/health")
    original = json.loads(a.capture.read_text())["prompt_tokens"]
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": info["Config"]["Image"],
        "image_id": info["Image"],
        "toggle_env": a.toggle_env,
        "gate_file": a.gate,
        "config": {
            k: env.get(k)
            for k in (
                "PARALLEL",
                "VLLM_ARVQ_GROUPED_PREFILL",
                "VLLM_NVFP4_P4_MAX_TOKENS",
                "VLLM_NVFP4_P4_PAIRED",
                "VLLM_ARVQ_COMPACT_PREFILL",
            )
        },
        "method": (
            "Configured warmups, then measured pairs in alternating order. "
            "Prefix cache reset before every request. One output token, same "
            "body with fresh actual UUID prefix to prevent LMCache reuse. "
            "Server prefill time is primary; HTTP wall latency is secondary."
        ),
        "rows": [],
    }
    try:
        b.fetch(a.base + "/reset_prefix_cache", {}, timeout=120)
        cache_method = "reset_prefix_cache"
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        cache_method = "unique_cache_salt"
    result["cache_method"] = cache_method
    result["method"] = result["method"].replace(
        "Prefix cache reset before every request.",
        "Cold cache forced before every request using " + cache_method + ".",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for length in a.lengths:
            if length <= 0:
                raise ValueError("Prompt length must be positive")
            tokens = (original * ((length + len(original) - 1) // len(original)))[
                :length
            ]
            prompt_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            for warmup, count in ((True, a.warmup), (False, a.runs)):
                for repetition in range(count):
                    for enabled in (
                        (False, True) if repetition % 2 == 0 else (True, False)
                    ):
                        set_mode(a.container, enabled, a.gate)
                        if cache_method == "reset_prefix_cache":
                            b.fetch(a.base + "/reset_prefix_cache", {}, timeout=120)
                        salt = str(uuid.uuid4())
                        prefix = json.loads(
                            b.fetch(
                                a.base + "/tokenize",
                                {
                                    "model": MODEL,
                                    "prompt": "Unique cold benchmark " + salt + ". ",
                                },
                            )
                        )["tokens"]
                        request_tokens = (prefix + tokens)[:length]
                        request_hash = hashlib.sha256(
                            json.dumps(request_tokens).encode()
                        ).hexdigest()
                        metrics_before = b.get_metrics(a.base + "/metrics")
                        start = time.perf_counter()
                        response = json.loads(
                            b.fetch(
                                a.base + "/v1/completions",
                                {
                                    "model": MODEL,
                                    "prompt": request_tokens,
                                    "max_tokens": 1,
                                    "temperature": 0,
                                    "seed": 173,
                                    "ignore_eos": True,
                                    "cache_salt": salt,
                                },
                                timeout=900,
                            )
                        )
                        elapsed = time.perf_counter() - start
                        usage = response.get("usage", {})
                        for _ in range(50):
                            metrics_after = b.get_metrics(a.base + "/metrics")
                            metrics_delta = b.delta_metrics(
                                metrics_before, metrics_after
                            )
                            prefill_count = b.metric_sum(
                                metrics_delta, "vllm:request_prefill_time_seconds_count"
                            )
                            if prefill_count >= 1:
                                break
                            time.sleep(0.1)
                        prefill_s = b.metric_sum(
                            metrics_delta, "vllm:request_prefill_time_seconds_sum"
                        )
                        gpu_hits = b.metric_sum(metrics_delta, "vllm:prefix_cache_hits")
                        external_hits = b.metric_sum(
                            metrics_delta, "vllm:external_prefix_cache_hits"
                        )
                        hit_metrics_present = all(
                            b.metric_present(metrics_after, name)
                            for name in (
                                "vllm:prefix_cache_hits",
                                "vllm:external_prefix_cache_hits",
                            )
                        )
                        isolated_cold = (
                            prefill_count == 1
                            and prefill_s > 0
                            and hit_metrics_present
                            and gpu_hits == 0
                            and external_hits == 0
                        )
                        row = {
                            "grouped": enabled,
                            "warmup": warmup,
                            "repetition": repetition,
                            "prompt_tokens": length,
                            "base_prompt_sha256": prompt_hash,
                            "prompt_sha256": request_hash,
                            "actual_prompt_token_ids": request_tokens,
                            "cache_salt": salt,
                            "request_wall_s": elapsed,
                            "http_input_tokens_per_s": length / elapsed,
                            "prefill_time_s": prefill_s,
                            "prefill_count_delta": prefill_count,
                            "gpu_cache_hit_delta": gpu_hits,
                            "external_cache_hit_delta": external_hits,
                            "isolated_cold_prefill": isolated_cold,
                            "prefill_tokens_per_s": length / prefill_s
                            if isolated_cold
                            else None,
                            "metrics_before": metrics_before,
                            "metrics_after": metrics_after,
                            "usage": usage,
                            "response": response,
                        }
                        result["rows"].append(row)
                        save()
                        if not isolated_cold:
                            raise RuntimeError(
                                "Prefill measurement is not isolated and cache cold; "
                                "raw metrics saved"
                            )
                        if (
                            usage.get("prompt_tokens") != length
                            or usage.get("completion_tokens") != 1
                        ):
                            raise RuntimeError(
                                "Actual usage differs from requested prefill workload"
                            )
                        print(
                            json.dumps(
                                {
                                    k: v
                                    for k, v in row.items()
                                    if k
                                    not in (
                                        "response",
                                        "metrics_before",
                                        "metrics_after",
                                        "actual_prompt_token_ids",
                                    )
                                }
                            ),
                            flush=True,
                        )
        result["primary_metric"] = (
            "Actual prompt tokens / request_prefill_time_seconds_sum delta; "
            "count delta 1 and zero GPU/external hits required. "
            "HTTP latency is secondary."
        )
        result["summary"] = []
        for length in a.lengths:
            means = {}
            for enabled in (False, True):
                values = [
                    r["prefill_time_s"]
                    for r in result["rows"]
                    if not r["warmup"]
                    and r["grouped"] == enabled
                    and r["prompt_tokens"] == length
                ]
                means[enabled] = statistics.mean(values)
                result["summary"].append(
                    {
                        "prompt_tokens": length,
                        "grouped": enabled,
                        "mean_prefill_s": means[enabled],
                        "individual_prefill_s": values,
                    }
                )
            result["summary"].append(
                {"prompt_tokens": length, "grouped_speedup": means[False] / means[True]}
            )
    finally:
        set_mode(a.container, True, a.gate)
        result["final_gate"] = "on"
        save()


if __name__ == "__main__":
    main()
