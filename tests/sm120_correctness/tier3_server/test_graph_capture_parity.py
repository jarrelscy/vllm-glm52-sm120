# SPDX-License-Identifier: Apache-2.0
"""TIER 3 — eager (CUDAGRAPH=0) vs graphed generation parity.

Guards idea 9 (any cudagraph capture change): the same temp-0 probe must
produce comparable DEPTH and a correct answer under both execution modes.
Byte equality is NOT gated (NCCL-level divergence is expected and
documented); the gates are depth similarity + coherence + answer parity
— exactly the axes on which a bad graph capture (stale buffer, wrong
shape bucket, dangling address) diverges.

Two-phase workflow (a single server can only run one mode at a time):

  1. Against the GRAPHED server (default boot):
        python3 test_graph_capture_parity.py --probe graphed
  2. Reboot the server with CUDAGRAPH=0 (entrypoint env), then:
        python3 test_graph_capture_parity.py --probe eager
  3. The pytest gate (or --compare) compares the two saved probes.

Probe files live in goldens/graph_parity/ and are keyed by config.
"""

import json
import pathlib
import re
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

OUT_DIR = _SUITE / "goldens" / "graph_parity"

# temp-0 probe: a short multi-step reasoning question with a known
# answer + a mini-needle, so both depth and correctness are measurable.
PROBE_QUESTION = (
    "A warehouse robot starts with 128 crates. It delivers half of its "
    "crates to dock A, then picks up 37 more, then delivers exactly 25 "
    "to dock B. Work through this step by step, then give the final "
    "number of crates as a line 'FINAL: <n>'."
)
PROBE_ANSWER = "76"
MAX_TOKENS = 4000

DEPTH_RATIO_LO, DEPTH_RATIO_HI = 0.60, 1.67


def run_probe(label: str) -> dict:
    from common.server_client import chat, server_alive
    assert server_alive(), "no live server on :8001"
    obj = chat(PROBE_QUESTION, MAX_TOKENS, temperature=0)
    ch = obj["choices"][0]
    content = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    m = re.search(r"FINAL:\s*\**\s*(\d+)", content) or \
        re.search(r"FINAL:\s*\**\s*(\d+)", reasoning)
    rec = {
        "label": label,
        "completion_tokens": obj["usage"]["completion_tokens"],
        "finish_reason": ch.get("finish_reason"),
        "final_answer": m.group(1) if m else None,
        "content_tail": content[-400:],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"probe_{label}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[probe {label}] ctoks={rec['completion_tokens']} "
          f"finish={rec['finish_reason']} answer={rec['final_answer']} "
          f"-> {path}")
    return rec


def _load(label: str) -> dict:
    path = OUT_DIR / f"probe_{label}.json"
    if not path.exists():
        pytest.skip(
            f"missing probe file {path}; two-phase workflow: run "
            f"`python3 {pathlib.Path(__file__).name} --probe graphed` "
            "against the graphed server, reboot with CUDAGRAPH=0, run "
            "`--probe eager`, then this gate")
    with open(path) as f:
        return json.load(f)


def test_graph_capture_parity():
    g = _load("graphed")
    e = _load("eager")
    for rec in (g, e):
        assert rec["finish_reason"] == "stop", \
            (f"{rec['label']}: finish_reason={rec['finish_reason']} "
             "(hit token cap or aborted — incoherent generation)")
        assert rec["final_answer"] == PROBE_ANSWER, \
            (f"{rec['label']}: wrong answer {rec['final_answer']!r} "
             f"(expected {PROBE_ANSWER}); tail: {rec['content_tail']!r}")
    ratio = g["completion_tokens"] / max(e["completion_tokens"], 1)
    assert DEPTH_RATIO_LO <= ratio <= DEPTH_RATIO_HI, \
        (f"graphed/eager depth ratio {ratio:.2f} outside "
         f"[{DEPTH_RATIO_LO}, {DEPTH_RATIO_HI}] "
         f"(graphed={g['completion_tokens']}, eager={e['completion_tokens']})"
         " — capture-dependent behavior divergence")
    print(f"[parity] depth ratio {ratio:.2f}, both answered "
          f"{PROBE_ANSWER}, finish=stop")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--probe":
        run_probe(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        test_graph_capture_parity()
        print("PASS test_graph_capture_parity")
    else:
        module_main(globals())
