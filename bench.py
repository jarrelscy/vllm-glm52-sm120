#!/usr/bin/env python3
"""
GLM-5.2 hybrid decode/prefill benchmark — reproduces the config-matrix numbers.

Measures DECODE tok/s the CORRECT way: NON-STREAMED, with a prefix-cache-warmed
second run, using the server's true ``usage.completion_tokens``.

WHY NON-STREAMED: under speculative decoding (MTP / DSpark) vLLM bundles multiple
accepted tokens into a SINGLE streamed SSE delta. Counting SSE chunks therefore
UNDERCOUNTS throughput by ~the acceptance factor (we measured 104 deltas for 300
real tokens). Never benchmark spec decode by counting stream chunks — use
usage.completion_tokens from a non-streamed response, as this script does.

USAGE:
    # 1) launch ONE server config (see BENCHMARKING.md), e.g.:
    #    PARALLEL=pp4-mtp   MODEL_DIR=/data/huggingface/glm52-models/1m bash serve_mtp.sh
    # 2) run the benchmark against it:
    python bench.py                       # all 3 workloads, short context
    python bench.py --workload code       # one workload
    python bench.py --ctx 200000 --docfile book.txt   # long-context decode+prefill

ENV:
    VLLM_HOST   default http://localhost:8001
    VLLM_API_KEY  optional bearer token (if the server was launched with --api-key)
    MODEL       served-model-name, default "glm-5.2"

NOTE on acceptance length: the OpenAI response does not carry per-request spec
acceptance. Read it from the server log lines ("Speculative metrics: ... mean
acceptance length") or the /metrics endpoint. This script reports throughput
(decode tok/s + prefill tok/s), which is what the matrix cells record.
"""
import argparse, json, os, time, urllib.request

HOST  = os.environ.get("VLLM_HOST", "http://localhost:8001")
KEY   = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("MODEL", "glm-5.2")

# --- EXACT prompts used to produce the published benchmark matrix -------------
PROMPTS = {
    # EASY / low-entropy -> near-perfect spec acceptance
    "count": "Write out the integers from 1 to 500, one per line. Output only the numbers.",
    # CODE / medium-entropy
    "code":  ("Implement in Python, as separate functions each with a full docstring "
              "and inline comments: bubble_sort, insertion_sort, selection_sort, "
              "merge_sort, and quick_sort. Output only the code."),
    # COMPLEX / high-entropy reasoning-style prose
    "tcp":   "Write a detailed 400-word explanation of how TCP congestion control works.",
}

def _chat(content, max_tokens, ctx_doc=None):
    msg = (ctx_doc + "\n\n---\n" + content) if ctx_doc else content
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": msg}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(HOST + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    u = d["usage"]
    txt = (d["choices"][0].get("message") or {}).get("content", "")
    return u["completion_tokens"], u["prompt_tokens"], dt, txt

def bench(workload, ngen, ctx_doc):
    prompt = PROMPTS[workload]
    # run 1: warms the prefix cache; its time ~= prefill for this prompt
    _, pt, t_prefill, _ = _chat(prompt, 8, ctx_doc)
    # run 2: prefill is cached -> total time ~= pure decode
    n, _, t, txt = _chat(prompt, ngen, ctx_doc)
    prefill_tps = pt / t_prefill if t_prefill > 0 else 0.0
    print(f"[{workload:5}] prompt_tokens={pt:>7}  "
          f"PREFILL~{prefill_tps:7.1f} tok/s  "
          f"DECODE {n/t:6.2f} tok/s  ({n} tok / {t:.2f}s, cached prefill)")
    print(f"        coherent: {repr((txt or '')[:100])}")
    return n / t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=list(PROMPTS) + ["all"], default="all")
    ap.add_argument("--ngen", type=int, default=240, help="tokens generated in the timed run")
    ap.add_argument("--ctx", type=int, default=0, help="approx context tokens to prepend (long-ctx test)")
    ap.add_argument("--docfile", default=None,
                    help="UTF-8 text file for long-ctx context (a public-domain book works well; "
                         "for realistic spec acceptance at depth use coherent prose, not filler)")
    a = ap.parse_args()

    ctx_doc = None
    if a.ctx > 0:
        if a.docfile:
            raw = open(a.docfile, encoding="utf-8", errors="ignore").read()
            ctx_doc = raw[: int(a.ctx * 4.0)]            # ~4 chars/token
        else:
            ctx_doc = ("The quick brown fox jumps over the lazy dog. " * max(1, a.ctx // 8))[: int(a.ctx * 4.0)]
        src = f"docfile {a.docfile}" if a.docfile else "SYNTHETIC filler (fine for base decode; use --docfile for realistic spec acceptance)"
        print(f"# long-context: ~{a.ctx} tokens prepended ({src})")

    print(f"# host={HOST} model={MODEL}")
    for w in (list(PROMPTS) if a.workload == "all" else [a.workload]):
        bench(w, a.ngen, ctx_doc)

if __name__ == "__main__":
    main()
