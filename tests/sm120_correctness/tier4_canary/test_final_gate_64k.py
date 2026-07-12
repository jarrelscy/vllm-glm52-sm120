# SPDX-License-Identifier: Apache-2.0
"""TIER 4 — THE FINAL GATE (64K).  MANDATORY PROMOTION GATE.

EVERY future prefill or decode change must pass this test before it is
promoted.  No exceptions.  (README + REGISTRY.md list it first.)

A fixed, deterministic ~64K-token prompt (common/final_gate_prompt.py:
long multi-part technical document + facts embedded at 10%/50%/90%
depth + a final multi-step derivation that synthesizes all of them) is
run at temp 0 against the live server.  Because byte-identical greedy
output is UNACHIEVABLE on this multi-GPU server (NCCL run-to-run
nondeterminism — see memory notes), the golden is an ENVELOPE captured
over N=5 runs of the KNOWN-GOOD shipped config (make_goldens.sh):

  (a) teacher-forced prompt_logprobs over the full 64K prompt
      -> global mean NLL band + per-256-token-chunk mean envelope.
      The prefill computes every one of the ~64K positions, so this
      validates prefill-path changes bit-deep; decode changes surface
      in the same numbers via the shared forward.
  (b) generation depth range,
  (c) the extracted facts/answer,
and all 5 raw runs are stored (goldens/final_gate_64k_runs.json.gz).

Candidate gates:
  (a) mean NLL within the golden band; every 256-token chunk mean
      inside its envelope (+margin) — catches LOCALIZED corruption
      (e.g. one bad prefill chunk) that a global mean would dilute;
  (b) generated answer contains the serial, the product and FINAL: 95;
  (c) depth within [golden_min * 0.6, golden_max * 1.5] — catches
      silent collapse AND runaway;
  (d) finish_reason == stop.

Runtime: ~20-30 min per candidate; golden capture ~1-2 h.
"""

import gzip
import json
import pathlib
import statistics
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common import final_gate_prompt as FG  # noqa: E402
from common.pytest_shim import module_main, pytest  # noqa: E402

GOLDEN = _SUITE / "goldens" / "final_gate_64k.json"
RAW_RUNS = _SUITE / "goldens" / "final_gate_64k_runs.json.gz"

CHUNK = 256
CHUNK_MARGIN = 0.05     # logprob units around the observed chunk envelope
DEPTH_LO_FRAC = 0.6
DEPTH_HI_FRAC = 1.5
GEN_MAX_TOKENS = 30000


def _prompt() -> str:
    return FG.build_prompt(64000)


def teacher_forced_pass(prompt: str):
    """Returns (per-position logprobs list, prompt_tokens)."""
    from common.server_client import MODEL, post_json
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "echo": True, "logprobs": 0,
               "prompt_logprobs": 0}
    obj = post_json("/completions", payload, timeout=7200)
    ch = obj["choices"][0]
    raw = ch.get("prompt_logprobs")
    vals = []
    if raw is not None:
        for tok in raw:
            if tok is None:
                continue
            if isinstance(tok, dict):
                entries = list(tok.values())
                actual = [e for e in entries
                          if isinstance(e, dict) and e.get("rank") is not None]
                pick = (actual or entries)[0]
                vals.append(float(pick["logprob"]
                                  if isinstance(pick, dict) else pick))
            else:
                vals.append(float(tok))
    elif ch.get("logprobs") and ch["logprobs"].get("token_logprobs"):
        vals = [v for v in ch["logprobs"]["token_logprobs"] if v is not None]
    if not vals:
        raise RuntimeError("no prompt logprobs returned")
    return vals, obj["usage"]["prompt_tokens"]


def generation_pass(prompt: str):
    from common.server_client import chat
    obj = chat(prompt, GEN_MAX_TOKENS, temperature=0, timeout=7200)
    ch = obj["choices"][0]
    content = (ch["message"].get("content") or "")
    reasoning = (ch["message"].get("reasoning_content") or "")
    return {
        "ctoks": obj["usage"]["completion_tokens"],
        "finish": ch.get("finish_reason"),
        "fails": FG.check_answer(content or reasoning),
        "tail": (content or reasoning)[-400:],
    }


def _chunk_means(vals):
    return [statistics.fmean(vals[i:i + CHUNK])
            for i in range(0, len(vals) - CHUNK + 1, CHUNK)]


def _load_golden():
    if not GOLDEN.exists():
        pytest.skip(f"golden missing ({GOLDEN}) — run make_goldens.sh "
                    "--final-gate against the KNOWN-GOOD shipped config "
                    "(tp4-1m-mtp) FIRST; the final gate is meaningless "
                    "without a trusted baseline")
    with open(GOLDEN) as f:
        return json.load(f)


def test_final_gate_teacher_forced_nll():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    g = _load_golden()
    vals, ptoks = teacher_forced_pass(_prompt())
    assert ptoks == g["prompt_tokens"], \
        (f"prompt tokenizes to {ptoks} tokens vs golden "
         f"{g['prompt_tokens']} — tokenizer/template drift; regenerate "
         "goldens deliberately if intended")

    mean_nll = -statistics.fmean(vals)
    band = max(0.01, 6 * g["mean_nll_std"])
    assert abs(mean_nll - g["mean_nll"]) <= band, \
        (f"FINAL GATE: mean NLL {mean_nll:.5f} outside golden "
         f"{g['mean_nll']:.5f} ± {band:.5f} — numeric corruption in the "
         "forward pass. DO NOT PROMOTE.")

    chunks = _chunk_means(vals)
    n = min(len(chunks), len(g["chunk_mean_min"]))
    bad = []
    for i in range(n):
        lo = g["chunk_mean_min"][i] - CHUNK_MARGIN
        hi = g["chunk_mean_max"][i] + CHUNK_MARGIN
        if not (lo <= chunks[i] <= hi):
            bad.append((i, chunks[i], lo, hi))
    assert not bad, \
        (f"FINAL GATE: {len(bad)}/{n} 256-token chunks outside the golden "
         f"envelope; first: chunk {bad[0][0]} mean {bad[0][1]:.4f} not in "
         f"[{bad[0][2]:.4f}, {bad[0][3]:.4f}] — LOCALIZED prefill "
         "corruption. DO NOT PROMOTE.")
    print(f"[final-gate/NLL] mean {mean_nll:.5f} (golden "
          f"{g['mean_nll']:.5f}±{band:.5f}), all {n} chunks in envelope")


def test_final_gate_generation():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    g = _load_golden()
    r = generation_pass(_prompt())
    print(f"[final-gate/gen] ctoks={r['ctoks']} finish={r['finish']}")
    lo = int(g["gen_depth_min"] * DEPTH_LO_FRAC)
    hi = int(g["gen_depth_max"] * DEPTH_HI_FRAC)
    problems = list(r["fails"])
    if r["finish"] != "stop":
        problems.append(f"finish_reason={r['finish']} (expected stop)")
    if not (lo <= r["ctoks"] <= hi):
        problems.append(
            f"depth {r['ctoks']} outside [{lo}, {hi}] "
            f"(golden range {g['gen_depth_min']}-{g['gen_depth_max']}) — "
            "silent collapse or runaway")
    assert not problems, \
        ("FINAL GATE FAILED — DO NOT PROMOTE THIS CHANGE:\n  " +
         "\n  ".join(problems) + f"\n  tail: {r['tail']!r}")
    print("[final-gate/gen] answer + depth + finish all PASS")


def capture_golden(runs: int = 5):
    """make_goldens.sh --final-gate: N-run envelope on a TRUSTED config."""
    from common.server_client import server_alive
    assert server_alive(), "no live server"
    prompt = _prompt()
    tf_runs, gen_runs = [], []
    ptoks = None
    for i in range(runs):
        vals, ptoks = teacher_forced_pass(prompt)
        tf_runs.append(vals)
        gen = generation_pass(prompt)
        assert not gen["fails"] and gen["finish"] == "stop", \
            (f"run {i + 1}: the KNOWN-GOOD config failed the gate itself "
             f"({gen['fails'] or gen['finish']}) — refusing to capture a "
             "golden from a broken baseline")
        gen_runs.append(gen)
        print(f"  run {i + 1}/{runs}: ptoks={ptoks} "
              f"meanNLL={-statistics.fmean(vals):.5f} "
              f"gen_ctoks={gen['ctoks']}")
    n = min(len(v) for v in tf_runs)
    tf_runs = [v[:n] for v in tf_runs]
    chunk_runs = [_chunk_means(v) for v in tf_runs]
    nc = min(len(c) for c in chunk_runs)
    means = [-statistics.fmean(v) for v in tf_runs]
    golden = {
        "prompt_tokens": ptoks,
        "runs": runs,
        "chunk": CHUNK,
        "mean_nll": statistics.fmean(means),
        "mean_nll_std": statistics.pstdev(means),
        "chunk_mean_min": [min(c[i] for c in chunk_runs) for i in range(nc)],
        "chunk_mean_max": [max(c[i] for c in chunk_runs) for i in range(nc)],
        "gen_depth_min": min(r["ctoks"] for r in gen_runs),
        "gen_depth_max": max(r["ctoks"] for r in gen_runs),
    }
    with open(GOLDEN, "w") as f:
        json.dump(golden, f)
    with gzip.open(RAW_RUNS, "wt") as f:
        json.dump({"teacher_forced": tf_runs, "generation": gen_runs}, f)
    print(f"golden written: {GOLDEN} (+ raw runs {RAW_RUNS})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-golden":
        capture_golden(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        module_main(globals())
