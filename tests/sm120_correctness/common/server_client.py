# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only client for the live GLM-5.2 server on :8001.

Promoted from the scratchpad harnesses (glmbench.py / refcap.py /
needle.py / oracle2.py).  No third-party dependencies so the server and
canary tiers can run on a bare host.  The API key is read from the
environment or the deployment .env file — never hardcoded.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

BASE = os.environ.get("GLM_BASE_URL", "http://localhost:8001/v1")
ENV_FILE = os.environ.get("GLM_ENV_FILE", "/home/jarrelscy/homeassistant/.env")
MODEL = os.environ.get("GLM_MODEL", "glm-5.2")


def get_key() -> str:
    k = os.environ.get("VLLM_API_KEY")
    if k:
        return k
    try:
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("VLLM_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    raise RuntimeError(
        "no API key: set VLLM_API_KEY or GLM_ENV_FILE (a file containing "
        "VLLM_API_KEY=...)")


def _request(path: str, payload: dict | None = None, timeout: int = 7200):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + get_key()})
    return urllib.request.urlopen(req, timeout=timeout)


def post_json(path: str, payload: dict, timeout: int = 7200) -> dict:
    return json.loads(_request(path, payload, timeout).read())


def get_json(path: str, timeout: int = 60) -> dict:
    return json.loads(_request(path, None, timeout).read())


def chat(prompt: str, max_tokens: int, temperature: float = 0.0,
         timeout: int = 7200, **extra) -> dict:
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature,
               "stream": False}
    payload.update(extra)
    return post_json("/chat/completions", payload, timeout)


def completions(payload: dict, timeout: int = 7200) -> dict:
    payload = {"model": MODEL, **payload}
    return post_json("/completions", payload, timeout)


def server_alive() -> bool:
    try:
        get_json("/models", timeout=10)
        return True
    except (urllib.error.URLError, OSError, RuntimeError):
        return False


def max_model_len() -> int | None:
    try:
        d = get_json("/models", timeout=30)
        return int(d["data"][0].get("max_model_len"))
    except Exception:
        return None


def metrics() -> dict[str, float]:
    """Scrape the Prometheus /metrics endpoint (spec-decode counters)."""
    url = BASE.rsplit("/v1", 1)[0] + "/metrics"
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + get_key()})
    txt = urllib.request.urlopen(req, timeout=30).read().decode()
    out: dict[str, float] = {}
    for key in ("vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:spec_decode_num_drafts_total"):
        m = re.findall(r"^%s(?:\{[^}]*\})?\s+([0-9.e+]+)" % re.escape(key),
                       txt, re.M)
        out[key] = sum(float(x) for x in m) if m else None
    # per-position acceptance, if exported
    per_pos = re.findall(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos"
        r"(?:_total)?\{[^}]*position=\"(\d+)\"[^}]*\}\s+([0-9.e+]+)",
        txt, re.M)
    if per_pos:
        acc = {}
        for pos, val in per_pos:
            acc[int(pos)] = acc.get(int(pos), 0.0) + float(val)
        out["per_pos_accepted"] = acc
    return out


def make_prompt(approx_tokens: int, salt: str = "") -> str:
    """Deterministic filler prompt (~20 chars/token sentence repeats)."""
    s = (f"{salt} The quick brown fox jumps over the lazy dog near the river "
         "while counting inventory items and recording measurements. ")
    return s * (int(approx_tokens / 20) + 1)


NEEDLE_SECRET = "The vault passcode is ZEPHYR-7741-QUARTZ."


def needle_probe(ctx_tokens: int, max_tokens: int = 200,
                 timeout: int = 7200) -> tuple[bool, str, int]:
    """Needle-in-a-haystack at ~ctx_tokens. Returns (pass, answer, ptoks)."""
    filler = make_prompt(ctx_tokens)
    mid = len(filler) // 2
    prompt = (filler[:mid] + " " + NEEDLE_SECRET + " " + filler[mid:] +
              "\n\nWhat is the vault passcode? Answer with only the passcode.")
    obj = chat(prompt, max_tokens, temperature=0, timeout=timeout)
    ans = obj["choices"][0]["message"]["content"] or ""
    return ("ZEPHYR-7741-QUARTZ" in ans, ans,
            obj["usage"]["prompt_tokens"])


def parse_answer_letter(text: str | None):
    """simple-evals canonical: first 'Answer: <LETTER>' match."""
    if not text:
        return None
    m = re.search(r"Answer:\s*\**([ABCD])", text)
    return m.group(1) if m else None


def extract_letter_second_call(response_text: str):
    """Letter-extraction retry via a minimal-effort second model call."""
    tail = response_text[-6000:]
    body = {
        "messages": [{"role": "user", "content":
                      "The following is a response to a multiple choice "
                      "question. Extract just the final answer letter. Reply "
                      "with only a single letter, one of A, B, C, or D.\n\n"
                      + tail}],
        "temperature": 0, "top_p": 1.0, "max_tokens": 2000,
        "chat_template_kwargs": {"reasoning_effort": "minimal"}}
    try:
        d = post_json("/chat/completions", {"model": MODEL, **body})
        c = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"([ABCD])", c)
        return m.group(1) if m else None
    except Exception:
        return None


def timed(fn, *a, **kw):
    t0 = time.time()
    r = fn(*a, **kw)
    return r, time.time() - t0
