# SPDX-License-Identifier: Apache-2.0
"""TIER 4 — the reasoning-depth CANARY (the SM100-class detector).

THE SINGLE MOST IMPORTANT TEST IN THIS SUITE.  A FAIL HERE = STOP THE
LINE.  Do not rationalize it, do not re-run until it passes, do not
ship the change: on B200 the ONLY observable signal of the fp8_ds_mla
kernel corruption was exactly this — deep chains of thought silently
collapsing to <= ~9.5k tokens with finish=stop and wrong answers, while
every smoke test looked fine.

Three GPQA-diamond questions with known-good deep reasoning depth are
run at temp 0, reasoning_effort=max, max_tokens=40000.  Gates (from
goldens/canary_reference.json, re-measurable by make_goldens.sh):

  * EVERY question reaches >= 60% of its reference depth,
  * EVERY question answers correctly (canonical parse, second-call
    letter extraction as fallback),
  * NO question finishes below 8000 tokens with finish_reason=stop.

~30-45 min on the graphed tp4-1m-mtp config (sequential; set
GLM_CANARY_CONC to parallelize at your own comparability risk).

GPQA data: the question TEXT is never committed (the fork remote may be
public; GPQA asks not to publish plaintext).  The CSV is read from
GPQA_CSV or data/gpqa_diamond.csv (git-ignored); gold letters are
derived by the same seeded shuffle as the original harness and
sanity-checked against the committed golds at runtime.
"""

import csv
import json
import os
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

GOLDEN = _SUITE / "goldens" / "canary_reference.json"
DEFAULT_CSV_PATHS = [
    os.environ.get("GPQA_CSV", ""),
    str(_SUITE / "data" / "gpqa_diamond.csv"),
    "/tmp/claude-1000/-home-jarrelscy-homeassistant/"
    "ca20f55c-6e14-4440-b072-adef0dae71ed/scratchpad/gpqa_diamond.csv",
]

PROMPT_TMPL = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")

STOP_THE_LINE = r"""
############################################################
#  CANARY FAILED — STOP THE LINE.                          #
#  This is the SM100 silent-collapse signature. Whatever   #
#  change is on this server MUST NOT be promoted. Revert   #
#  and bisect before any further optimization work.        #
############################################################
"""


def _find_csv() -> str:
    for p in DEFAULT_CSV_PATHS:
        if p and os.path.exists(p):
            return p
    pytest.skip("gpqa_diamond.csv not found: set GPQA_CSV or copy it to "
                f"{_SUITE / 'data'}/ (git-ignored)")


def build_questions() -> dict:
    """Same seeded shuffle as the original oracle harness (Random(0))."""
    rows = list(csv.DictReader(open(_find_csv())))
    rng = random.Random(0)
    qs = {}
    for i, r in enumerate(rows):
        opts = [r["Correct Answer"].strip(), r["Incorrect Answer 1"].strip(),
                r["Incorrect Answer 2"].strip(),
                r["Incorrect Answer 3"].strip()]
        perm = rng.sample(range(4), 4)
        choices = [opts[j] for j in perm]
        gold = "ABCD"[perm.index(0)]
        qs[f"gpqa-{i}"] = {"question": r["Question"].strip(),
                           "choices": choices, "gold": gold}
    return qs


def _load_golden() -> dict:
    with open(GOLDEN) as f:
        return json.load(f)


def _ask(qid: str, q: dict, max_tokens: int) -> dict:
    from common.server_client import (
        chat,
        extract_letter_second_call,
        parse_answer_letter,
    )
    prompt = PROMPT_TMPL.format(q=q["question"], a=q["choices"][0],
                                b=q["choices"][1], c=q["choices"][2],
                                d=q["choices"][3])
    obj = chat(prompt, max_tokens, temperature=0, top_p=1.0,
               chat_template_kwargs={"reasoning_effort": "max"})
    ch = obj["choices"][0]
    usage = obj.get("usage", {})
    content = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    ans = parse_answer_letter(content) or parse_answer_letter(reasoning)
    parse = "canonical"
    if ans is None:
        ans = extract_letter_second_call(content or reasoning)
        parse = "second_call" if ans else "null"
    return {"id": qid, "finish": ch.get("finish_reason"),
            "ctoks": usage.get("completion_tokens"), "ans": ans,
            "parse": parse}


def test_reasoning_canary():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    golden = _load_golden()
    qs = build_questions()

    # gold sanity: committed golds must match the seeded shuffle
    for qid, spec in golden["questions"].items():
        assert qs[qid]["gold"] == spec["gold"], \
            (f"GOLD SANITY FAIL for {qid}: computed {qs[qid]['gold']} vs "
             f"committed {spec['gold']} — CSV or shuffle drift, refusing "
             "to run the canary against wrong golds")

    max_tokens = golden["gates"]["max_tokens"]
    conc = int(os.environ.get("GLM_CANARY_CONC", "1"))
    ids = list(golden["questions"])

    if conc > 1:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            results = list(ex.map(
                lambda qid: _ask(qid, qs[qid], max_tokens), ids))
    else:
        results = [_ask(qid, qs[qid], max_tokens) for qid in ids]

    frac = golden["gates"]["depth_frac"]
    stop_floor = golden["gates"]["min_stop_tokens"]
    failures = []
    for r in results:
        spec = golden["questions"][r["id"]]
        ref = spec["ref_depth"]
        need = int(ref * frac)
        ok_depth = (r["ctoks"] or 0) >= need
        ok_ans = r["ans"] == spec["gold"]
        collapsed = (r["finish"] == "stop" and (r["ctoks"] or 0) < stop_floor)
        line = (f"  {r['id']}: ctoks={r['ctoks']} (ref~{ref}, need>={need}) "
                f"ans={r['ans']} gold={spec['gold']} finish={r['finish']} "
                f"parse={r['parse']}")
        print(line)
        if not ok_depth:
            failures.append(f"{r['id']}: depth {r['ctoks']} < {need} "
                            f"(60% of reference {ref}) — REASONING COLLAPSE")
        if collapsed:
            failures.append(f"{r['id']}: finished 'stop' at only "
                            f"{r['ctoks']} tokens (< {stop_floor}) — "
                            "the exact SM100 signature")
        if not ok_ans:
            failures.append(f"{r['id']}: WRONG ANSWER {r['ans']} "
                            f"(gold {spec['gold']})")
    if failures:
        print(STOP_THE_LINE)
        raise AssertionError("reasoning canary FAILED:\n" +
                             "\n".join(failures))
    print("[canary] all questions: depth OK, answers correct — PASS")


def capture_golden(out_path: pathlib.Path = GOLDEN):
    """make_goldens.sh: re-measure reference depths on a KNOWN-GOOD
    config (keeps committed golds; only updates ref_depth from a live
    run — run this ONLY when the current server is trusted)."""
    golden = _load_golden()
    qs = build_questions()
    for qid in golden["questions"]:
        r = _ask(qid, qs[qid], golden["gates"]["max_tokens"])
        assert r["ans"] == golden["questions"][qid]["gold"], \
            (f"{qid}: known-good config answered {r['ans']} != gold — "
             "refusing to record this as a reference depth")
        golden["questions"][qid]["ref_depth"] = r["ctoks"]
        print(f"  {qid}: ref_depth <- {r['ctoks']}")
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"golden written: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-golden":
        capture_golden()
    else:
        module_main(globals())
