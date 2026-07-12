# SPDX-License-Identifier: Apache-2.0
"""TIER 3 — MTP acceptance within the documented run-to-run envelope.

Guards the VERIFY path (ideas 1, 2, 5, 6, 9): the MTP verify forward
uses the same MoE gemv as the target model — corruption there shows up
as a DROP in draft acceptance long before it is visible in text.
Baseline (memory notes, tp4-1m-mtp ns=3): per-position p0 ~ 0.86,
overall acceptance run-to-run 0.76-0.82 on the SAME server; gates below
add margin and live in goldens/acceptance_envelope.json (regenerable by
make_goldens.sh against a known-good config).

Bit-exact greedy equality is UNACHIEVABLE on this multi-GPU server
(NCCL reduction-order nondeterminism) — envelope gating is the correct
server-level standard; kernel-level bit-exactness is Tier 1's job.
"""

import json
import pathlib
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

GOLDEN = _SUITE / "goldens" / "acceptance_envelope.json"


def _load_envelope():
    with open(GOLDEN) as f:
        return json.load(f)


def _run_probe():
    from common.server_client import chat, make_prompt, metrics
    tail = ("\nNow count from 1 to 200, printing one integer per line, "
            "nothing else.")
    essay = ("\nWrite a long, detailed technical essay on distributed "
             "systems.")
    chat(make_prompt(2000) + tail, 8)          # warm
    before = metrics()
    if before.get("vllm:spec_decode_num_draft_tokens_total") is None:
        pytest.skip("no spec-decode metrics — server runs without MTP "
                    "(acceptance gate not applicable to this config)")
    chat(make_prompt(2000) + tail, 250)
    chat(make_prompt(2000) + essay, 400)
    after = metrics()
    dd = (after["vllm:spec_decode_num_draft_tokens_total"] -
          before["vllm:spec_decode_num_draft_tokens_total"])
    da = (after["vllm:spec_decode_num_accepted_tokens_total"] -
          before["vllm:spec_decode_num_accepted_tokens_total"])
    dn = (after["vllm:spec_decode_num_drafts_total"] -
          before["vllm:spec_decode_num_drafts_total"])
    per_pos = None
    if "per_pos_accepted" in after and "per_pos_accepted" in before:
        per_pos = {
            p: after["per_pos_accepted"][p] -
            before["per_pos_accepted"].get(p, 0.0)
            for p in after["per_pos_accepted"]
        }
    return dd, da, dn, per_pos


def test_acceptance_in_envelope():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    env = _load_envelope()
    dd, da, dn, per_pos = _run_probe()
    assert dd > 200, f"probe produced too few draft tokens ({dd})"
    rate = da / dd
    acc_per_draft = da / dn if dn else float("nan")
    lo, hi = env["overall"]
    print(f"[accept] rate={rate:.4f} (envelope [{lo}, {hi}]), "
          f"acc/draft={acc_per_draft:.3f}, drafts={dn:.0f}")
    assert lo <= rate <= hi, \
        (f"MTP acceptance {rate:.4f} outside the documented envelope "
         f"[{lo}, {hi}] — verify-path corruption (or a real distribution "
         "shift: re-baseline ONLY after Tier-1 passes bit-exact and the "
         "canary passes).")

    if per_pos and 0 in per_pos and dn > 0:
        p0 = per_pos[0] / dn
        p0lo, p0hi = env["p0"]
        print(f"[accept] p0={p0:.4f} (envelope [{p0lo}, {p0hi}])")
        assert p0lo <= p0 <= p0hi, \
            f"per-position p0 acceptance {p0:.4f} outside [{p0lo}, {p0hi}]"
    else:
        print("[accept] per-position metrics not exported; overall gate only")


def capture_golden(runs: int = 3, out_path: pathlib.Path = GOLDEN):
    """make_goldens.sh: measure the envelope on the known-good config.

    Writes overall/p0 bands = observed min/max widened by 0.04 (the
    documented run-to-run spread is ~0.06 wide; 0.04 margin keeps the
    gate meaningful while absorbing server restarts)."""
    from common.server_client import server_alive
    assert server_alive(), "no live server"
    rates, p0s = [], []
    for r in range(runs):
        dd, da, dn, per_pos = _run_probe()
        rates.append(da / dd)
        if per_pos and 0 in per_pos and dn:
            p0s.append(per_pos[0] / dn)
        print(f"  run {r + 1}/{runs}: rate={rates[-1]:.4f}")
    golden = {
        "overall": [round(min(rates) - 0.04, 4), round(max(rates) + 0.04, 4)],
        "p0": ([round(min(p0s) - 0.04, 4), round(max(p0s) + 0.04, 4)]
               if p0s else [0.78, 0.94]),
        "raw_rates": rates,
        "raw_p0": p0s,
        "note": "measured by make_goldens.sh on the shipped config; "
                "documented memory envelope was 0.76-0.82 overall, p0~0.86",
    }
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"golden written: {out_path} -> {golden['overall']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-golden":
        capture_golden(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        module_main(globals())
