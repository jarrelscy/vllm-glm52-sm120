#!/usr/bin/env python3
"""Send one long-context request to a running vllm-serve instance.

Invoked by run_longctx_memfit_test.sh AFTER the server for a given
--max-model-len has already reported ready. Its only job is to exercise a
real request whose prompt approaches the target context length, so the
"activations + CUDA graph capture + MTP overhead + concurrency" slack
(estimated at ~3.2-5.9 GiB/GPU, never empirically tested per the task
background) gets probed by an actual forward pass, not just inferred from
the engine's startup KV-cache-reservation log lines.

Uses only the stdlib (urllib) deliberately -- no assumption about which
extra packages happen to be installed in the venv beyond vllm itself.

This script makes HTTP calls to a server that is presumed to already be
running and already holding the GPU lease; it does not itself acquire any
lease or touch CUDA/torch.

Exit code 0 = probe request succeeded (got a non-empty completion).
Exit code 1 = probe request failed (HTTP error, timeout, empty output).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def tokenize(base_url: str, model: str, text: str, timeout: float) -> int:
    """Use the server's own /tokenize endpoint so token counts reflect this
    model's real tokenizer, not a guessed words-per-token ratio."""
    out = http_post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": text},
        timeout=timeout,
    )
    # vLLM's /tokenize returns {"tokens": [...], "count": N, ...}
    if "count" in out:
        return int(out["count"])
    if "tokens" in out:
        return len(out["tokens"])
    raise RuntimeError(f"unexpected /tokenize response shape: {out!r}")


def build_prompt_of_length(
    base_url: str, model: str, target_tokens: int, timeout: float
) -> str:
    """Grow a filler prompt until its real tokenized length is within a few
    percent of target_tokens, using the server's own tokenizer to calibrate
    (word-count heuristics vary too much across tokenizers to trust blind)."""
    filler_unit = "the quick brown fox jumps over the lazy dog . "
    # Calibrate words-per-token ratio from one probe chunk.
    calib_words = 2000
    calib_text = filler_unit * (calib_words // len(filler_unit.split()) + 1)
    calib_word_count = len(calib_text.split())
    calib_tokens = tokenize(base_url, model, calib_text, timeout)
    if calib_tokens <= 0:
        raise RuntimeError("tokenizer calibration returned 0 tokens")
    words_per_token = calib_word_count / calib_tokens

    est_words = int(target_tokens * words_per_token)
    words = filler_unit.split()
    reps = max(1, est_words // len(words) + 1)
    text = (filler_unit * reps)

    # One correction pass: measure real tokens, scale the repeat count.
    actual_tokens = tokenize(base_url, model, text, timeout)
    if actual_tokens > 0 and actual_tokens != target_tokens:
        scale = target_tokens / actual_tokens
        reps = max(1, int(reps * scale))
        text = filler_unit * reps
        actual_tokens = tokenize(base_url, model, text, timeout)

    print(
        f"built probe prompt: target={target_tokens} tokens, "
        f"actual={actual_tokens} tokens (reps={reps})"
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument(
        "--target-context-tokens",
        type=int,
        required=True,
        help="the max_model_len this step is testing (100% of it)",
    )
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.9,
        help="build the prompt to this fraction of target-context-tokens, "
        "leaving room for --max-tokens of output plus tokenizer-estimate slop",
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for the completion call itself (long-context "
        "prefill on a 512K request can be slow, especially first-call)",
    )
    ap.add_argument("--tokenize-timeout", type=float, default=120.0)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    model = args.served_model_name
    target = int(args.target_context_tokens * args.fraction)

    t0 = time.time()
    try:
        prompt = build_prompt_of_length(base_url, model, target, args.tokenize_timeout)
    except Exception as e:  # noqa: BLE001 - report and fail the probe cleanly
        print(f"PROBE_FAIL: could not build calibrated prompt: {e!r}")
        return 1

    print(f"sending completion request (max_tokens={args.max_tokens}) ...")
    try:
        out = http_post_json(
            f"{base_url}/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
            timeout=args.request_timeout,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"PROBE_FAIL: HTTP {e.code}: {body[:2000]}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"PROBE_FAIL: request error: {e!r}")
        return 1

    elapsed = time.time() - t0
    try:
        text = out["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        print(f"PROBE_FAIL: unexpected response shape: {out!r}")
        return 1

    if not text or not text.strip():
        print(f"PROBE_FAIL: empty completion text, elapsed={elapsed:.1f}s, raw={out!r}")
        return 1

    print(f"PROBE_PASS: elapsed={elapsed:.1f}s output={text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
