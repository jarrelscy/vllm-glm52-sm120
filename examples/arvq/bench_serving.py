#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential usage-counted streaming benchmark, with MTP counter deltas.
No GPU/server mutations. Auth only via OPENAI_API_KEY; never serialized.
"""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import time
import urllib.parse
import urllib.request

import regex as re

FILLER = "The quick brown fox jumps over the lazy dog. "
PREFIXES = (
    "vllm:request_prefill_time_seconds_",
    "vllm:prefix_cache_",
    "vllm:external_prefix_cache_",
    "vllm:spec_decode_",
    "vllm:generation_tokens",
    "vllm:request_decode_time_seconds_",
    "vllm:request_success",
)


def headers():
    h = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def fetch(url, body=None, timeout=30):
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(), headers=headers()
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parsed_metrics(raw):
    out = {}
    for line in raw.splitlines():
        if not line.startswith(PREFIXES):
            continue
        m = re.match(r"^([^\s{]+)(\{.*\})?\s+([-+0-9.eE]+)(?:\s+.*)?$", line)
        if m and not m[1].endswith("_created"):
            out[m[1] + (m[2] or "")] = float(m[3])
    return out


def metric_sum(snapshot, name):
    return sum(
        v for k, v in snapshot.items() if k.split("{", 1)[0] in (name, name + "_total")
    )


def metric_present(snapshot, name):
    return any(k.split("{", 1)[0] in (name, name + "_total") for k in snapshot)


def delta_metrics(before, after):
    out = {k: after.get(k, 0) - before.get(k, 0) for k in before.keys() | after.keys()}
    if any(v < 0 for v in out.values()):
        raise ValueError(
            "Metrics counters decreased or disappeared; run cannot be normalized."
        )
    return out


def stream_result(events, total_s):
    """events are (seconds since request, parsed SSE object). Usage is authoritative.
    token_ids delta lengths locate exact first/last token batches;
    SSE chunks are never tokens.
    """
    usage = None
    token_events = []
    text_events = []
    finish = []
    ids_count = 0
    all_ids = True
    for when, event in events:
        if event.get("error"):
            raise ValueError("Server returned a streaming error.")
        if event.get("usage") is not None:
            usage = event["usage"]
        for c in event.get("choices", []):
            if c.get("index", 0) != 0:
                raise ValueError("Only one completion per request is supported.")
            ids = c.get("token_ids")
            if ids is not None:
                if ids:
                    token_events.append((when, len(ids)))
                    ids_count += len(ids)
            elif c.get("text"):
                all_ids = False
            if c.get("text"):
                text_events.append(when)
            if c.get("finish_reason"):
                finish.append(c["finish_reason"])
    if usage is None or "completion_tokens" not in usage:
        raise ValueError(
            "Missing final streamed usage.completion_tokens; "
            "refusing chunk-based fallback."
        )
    total = int(usage["completion_tokens"])
    ttft = (
        token_events[0][0]
        if token_events
        else (text_events[0] if text_events else None)
    )
    verified_ids = all_ids and bool(token_events) and ids_count == total
    first_n = token_events[0][1] if verified_ids else None
    last = (
        token_events[-1][0]
        if verified_ids
        else (text_events[-1] if text_events else None)
    )
    decode_s = last - ttft if last is not None and ttft is not None else None
    decode_tokens = total - first_n if first_n is not None else None
    return {
        "usage": usage,
        "completion_tokens": total,
        "total_s": total_s,
        "ttft_s": ttft,
        "decode_wall_s": decode_s,
        "first_emission_token_count": first_n,
        "post_first_emission_tokens": decode_tokens,
        "decode_tps": decode_tokens / decode_s
        if decode_tokens is not None and decode_s and decode_s > 0
        else None,
        "completion_tokens_over_decode_wall_tps": total / decode_s
        if decode_s and decode_s > 0
        else None,
        "end_to_end_tps": total / total_s if total_s else None,
        "streamed_token_ids_count": ids_count,
        "token_ids_match_usage": verified_ids,
        "text_event_count_diagnostic_only": len(text_events),
        "finish_reasons": finish,
        "token_event_timeline": token_events,
    }


def speculative_summary(delta, request):
    drafts = metric_sum(delta, "vllm:spec_decode_num_drafts")
    accepted = metric_sum(delta, "vllm:spec_decode_num_accepted_tokens")
    proposed = metric_sum(delta, "vllm:spec_decode_num_draft_tokens")
    server_decode = (
        metric_sum(delta, "vllm:request_decode_time_seconds_sum")
        if metric_present(delta, "vllm:request_decode_time_seconds_sum")
        else None
    )
    server_count = metric_sum(delta, "vllm:request_decode_time_seconds_count")
    gen = (
        metric_sum(delta, "vllm:generation_tokens")
        if metric_present(delta, "vllm:generation_tokens")
        else None
    )
    isolated = server_count == 1 and gen == request["completion_tokens"]
    return {
        "draft_request_steps": drafts,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "accepted_per_draft_step": accepted / drafts if drafts else None,
        "expected_emitted_per_draft_step": 1 + accepted / drafts if drafts else None,
        "actual_completion_tokens_per_draft_step": request["completion_tokens"] / drafts
        if drafts
        else None,
        "draft_acceptance_fraction": accepted / proposed if proposed else None,
        "server_decode_s": server_decode,
        "server_completed_request_count": server_count,
        "generation_token_delta": gen,
        "isolated_counters_match_request": isolated,
        "server_decode_ms_per_draft_step_proxy": server_decode * 1000 / drafts
        if server_decode is not None and drafts and isolated
        else None,
        "client_decode_ms_per_draft_step_proxy": request["decode_wall_s"]
        * 1000
        / drafts
        if request["decode_wall_s"] is not None and drafts and isolated
        else None,
    }


def tokenize(base, model, text):
    for url in (base.removesuffix("/v1") + "/tokenize", base + "/tokenize"):
        try:
            d = json.loads(fetch(url, {"model": model, "prompt": text}))
            return d.get("count", len(d.get("tokens", []))), url
        except Exception:
            pass
    return len(text.split()), "word_estimate"


def build_prompt(base, model, target):
    n = target // 8 + 1
    prompt = "Continue the following text:\n\n" + FILLER * n
    actual, method = tokenize(base, model, prompt)
    if actual > 0 and abs(actual - target) > target * 0.1:
        prompt = "Continue the following text:\n\n" + FILLER * (
            int(n * target / actual) + 1
        )
    return prompt, tokenize(base, model, prompt)


def get_metrics(url):
    return parsed_metrics(fetch(url).decode())


def completion(base, model, prompt, args):
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
    }
    req = urllib.request.Request(
        base + "/completions", data=json.dumps(body).encode(), headers=headers()
    )
    events = []
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            events.append((time.perf_counter() - start, json.loads(payload)))
    total = time.perf_counter() - start
    return stream_result(events, total), events


def normalized(new, reference):
    rows = [
        r
        for r in reference["runs"]
        if r["target_prompt_tokens"] == new["target_prompt_tokens"]
        and r.get("speculative", {}).get("isolated_counters_match_request")
        and r["speculative"].get("expected_emitted_per_draft_step")
    ]
    step = new.get("speculative", {}).get("server_decode_ms_per_draft_step_proxy")
    if not rows or not step:
        return None
    target = statistics.mean(
        r["speculative"]["expected_emitted_per_draft_step"] for r in rows
    )
    return {
        "reference_label": reference["label"],
        "reference_mean_expected_emitted_per_draft_step": target,
        "new_server_decode_ms_per_draft_step_proxy": step,
        "normalized_tps_estimate": target / (step / 1000),
        "method": (
            "Original mean(1+accepted/drafts) divided by new "
            "server-decode-time/draft-count. Counter-based "
            "estimate; not measured matched-acceptance "
            "throughput."
        ),
    }


def markdown(report):
    s = (
        f"# {report['label']}\n\n"
        "Token counts come from final streamed usage, never text-chunk counts. "
        "DecodeTPS excludes the first token batch when returned token IDs match "
        "usage. Counters are server-wide; normalization requires exactly one "
        "matching request and no counter reset.\n\n"
        "| Prompt target | Run | Actual output | TTFTms | DecodeTPS | "
        "Emitted/draft | Serverms/draft proxy | NormalizedTPS estimate |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )

    def v(x):
        return "—" if x is None else f"{x:.3f}"

    for r in report["runs"]:
        sp = r.get("speculative", {})
        norm = r.get("normalized_to_reference") or {}
        ttft = r["ttft_s"] * 1000 if r["ttft_s"] is not None else None
        s += (
            f"| {r['target_prompt_tokens']} | {r['run']} | "
            f"{r['completion_tokens']} | {v(ttft)} | {v(r['decode_tps'])} | "
            f"{v(sp.get('expected_emitted_per_draft_step'))} | "
            f"{v(sp.get('server_decode_ms_per_draft_step_proxy'))} | "
            f"{v(norm.get('normalized_tps_estimate'))} |\n"
        )
    return s + (
        "\n`1+accepted/drafts` is the fork’s expected emitted "
        "length per speculative request-step. Stop "
        "truncation, prefill output and non-speculative steps"
        " can make it differ from actual "
        "completion_tokens/drafts. Server decode time divided"
        " by draft count is a request-level latency proxy, "
        "not GPU kernel latency. Normalization changes "
        "acceptance arithmetically; it does not measure a "
        "rerun at forced acceptance.\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--metrics-url")
    ap.add_argument("--model")
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--prompt-lengths", nargs="+", type=int, default=[16, 128, 1024, 4096, 16384]
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--metrics-settle", type=float, default=5)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=173)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--normalize-reference", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.runs < 1 or args.max_tokens < 1 or min(args.prompt_lengths) < 1:
        ap.error("runs, token count and prompt lengths must be positive")
    base = args.endpoint.rstrip("/")
    metricurl = args.metrics_url or base.removesuffix("/v1") + "/metrics"
    model = args.model or json.loads(fetch(base + "/models"))["data"][0]["id"]
    reference = (
        json.loads(args.normalize_reference.read_text())
        if args.normalize_reference
        else None
    )
    report = {
        "label": args.label,
        "model": model,
        "endpoint": base,
        "metrics_url": metricurl,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
        "runs": [],
        "warmups": [],
        "warmup_runs_per_length": args.warmup_runs,
        "metric_source": (
            "local vllm/v1/spec_decode/metrics.py and vllm/v1/metrics/loggers.py"
        ),
        "counting": (
            "usage.completion_tokens; return_token_ids used to "
            "remove first emission batch from decode-rate "
            "numerator"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in args.prompt_lengths:
        prompt, (actual, method) = build_prompt(base, model, target)
        for run in range(-args.warmup_runs, args.runs):
            before = get_metrics(metricurl)
            result, events = completion(base, model, prompt, args)
            deadline = time.monotonic() + args.metrics_settle
            while True:
                after = get_metrics(metricurl)
                delta = delta_metrics(before, after)
                if (
                    metric_sum(delta, "vllm:generation_tokens")
                    >= result["completion_tokens"]
                    and metric_sum(delta, "vllm:request_decode_time_seconds_count") >= 1
                ):
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            result.update(
                target_prompt_tokens=target,
                tokenizer_prompt_tokens=actual,
                tokenizer_method=method,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                run=run,
                metrics_before=before,
                metrics_after=after,
                metrics_delta=delta,
                sse_events=events,
            )
            result["speculative"] = speculative_summary(delta, result)
            if reference:
                result["normalized_to_reference"] = normalized(result, reference)
            report["warmups" if run < 0 else "runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            args.output.with_suffix(".md").write_text(markdown(report))
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "target_prompt_tokens",
                            "run",
                            "completion_tokens",
                            "ttft_s",
                            "decode_tps",
                            "speculative",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
