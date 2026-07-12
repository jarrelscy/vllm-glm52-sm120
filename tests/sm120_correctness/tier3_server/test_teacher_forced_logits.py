# SPDX-License-Identifier: Apache-2.0
"""TIER 3 — teacher-forced prompt logprobs vs a stored golden envelope.

Catches NUMERIC CORRUPTION anywhere in the forward pass without needing
determinism: a fixed short prompt is teacher-forced (prompt_logprobs)
and each position's logprob must fall inside the envelope observed over
N=5 baseline runs of the CURRENT shipped config (captured by
make_goldens.sh — the envelope IS the measured run-to-run NCCL noise,
per the memory notes byte-identical greedy output is UNACHIEVABLE on
this multi-GPU server).

Guards: ideas 5, 6, 7 (prefill dequant rewrites + inductor fusion
passes), 8, 9 — anything that touches forward numerics.

Gate:
  * >= 99.5% of positions inside [env_min - margin, env_max + margin]
  * mean NLL within (golden mean +- max(0.02, 6 * golden std))
  * no position deviates more than ABS_CEIL from the golden mean track
"""

import json
import pathlib
import statistics
import sys

_SUITE = pathlib.Path(__file__).resolve().parents[1]
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

from common.pytest_shim import module_main, pytest  # noqa: E402

GOLDEN = _SUITE / "goldens" / "teacher_forced_logits.json"

# Fixed probe prompt: mixed prose + arithmetic + code-ish tokens so many
# layers/experts participate; short enough to be cheap (~200 tokens).
PROBE_PROMPT = (
    "Below is a short technical note.\n\n"
    "A distributed key-value store replicates each shard three times. "
    "If a shard holds 4096 keys and the cluster has 12 shards, the total "
    "number of stored key copies is 4096 * 12 * 3 = 147456. The "
    "coordinator batches writes in groups of 64, so a full resync issues "
    "147456 / 64 = 2304 batches. Consensus uses two-phase commit with "
    "a 250 ms timeout and exponential backoff (base 50 ms, factor 2).\n\n"
    "def resync(shards, copies=3, batch=64):\n"
    "    total = sum(s.keys * copies for s in shards)\n"
    "    return (total + batch - 1) // batch\n\n"
    "Question: how many batches does a full resync issue?")

MARGIN = 0.15      # logprob units added around the observed envelope
ABS_CEIL = 1.5     # no single position may drift this far from the mean
FRAC_INSIDE = 0.995


def collect_prompt_logprobs():
    """One teacher-forced pass; returns list[float] logprobs per position."""
    from common.server_client import MODEL, post_json
    payload = {
        "model": MODEL,
        "prompt": PROBE_PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "echo": True,
        "logprobs": 0,
        # vLLM extension: return logprobs of the prompt tokens themselves
        "prompt_logprobs": 0,
    }
    obj = post_json("/completions", payload, timeout=600)
    ch = obj["choices"][0]
    vals = None
    if ch.get("prompt_logprobs") is not None:
        vals = []
        for tok in ch["prompt_logprobs"]:
            if tok is None:
                continue  # first token has no logprob
            if isinstance(tok, dict):
                # {token_id: {"logprob": ...}} — take the sampled entry
                # (rank field marks the actual token; fall back to max)
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
        raise RuntimeError(
            "server returned no prompt logprobs — enable prompt_logprobs "
            "support or check the vLLM OpenAI-compat version")
    return vals


def _load_golden():
    if not GOLDEN.exists():
        pytest.skip(f"golden missing ({GOLDEN}) — run make_goldens.sh "
                    "--server against the KNOWN-GOOD shipped config first")
    with open(GOLDEN) as f:
        return json.load(f)


def test_teacher_forced_logits_within_envelope():
    from common.server_client import server_alive
    if not server_alive():
        pytest.skip("no live server on :8001")
    g = _load_golden()
    vals = collect_prompt_logprobs()
    n = min(len(vals), len(g["env_min"]))
    assert n > 50, f"probe too short ({n} positions) — tokenizer changed?"
    if len(vals) != len(g["env_min"]):
        print(f"[warn] position count {len(vals)} != golden "
              f"{len(g['env_min'])} (tokenizer drift?); comparing first {n}")

    inside = 0
    worst = (0.0, -1)
    for i in range(n):
        lo = g["env_min"][i] - MARGIN
        hi = g["env_max"][i] + MARGIN
        if lo <= vals[i] <= hi:
            inside += 1
        dev = abs(vals[i] - g["mean_track"][i])
        if dev > worst[0]:
            worst = (dev, i)
    frac = inside / n
    assert frac >= FRAC_INSIDE, \
        (f"only {frac:.4f} of positions inside the golden envelope "
         f"(need >= {FRAC_INSIDE}); worst deviation {worst[0]:.3f} at "
         f"position {worst[1]} — numeric corruption in the forward pass")
    assert worst[0] <= ABS_CEIL, \
        (f"position {worst[1]} deviates {worst[0]:.3f} from the golden "
         f"mean track (ceil {ABS_CEIL})")

    mean_nll = -statistics.fmean(vals[:n])
    lo = g["mean_nll"] - max(0.02, 6 * g["mean_nll_std"])
    hi = g["mean_nll"] + max(0.02, 6 * g["mean_nll_std"])
    assert lo <= mean_nll <= hi, \
        (f"mean NLL {mean_nll:.4f} outside golden band [{lo:.4f}, {hi:.4f}] "
         f"(golden {g['mean_nll']:.4f} ± {g['mean_nll_std']:.4f})")
    print(f"[ok] {n} positions, {frac:.4f} inside envelope, mean NLL "
          f"{mean_nll:.4f} (golden {g['mean_nll']:.4f})")


def capture_golden(runs: int = 5, out_path: pathlib.Path = GOLDEN):
    """Called by make_goldens.sh: capture the N-run envelope."""
    all_runs = []
    for r in range(runs):
        vals = collect_prompt_logprobs()
        all_runs.append(vals)
        print(f"  run {r + 1}/{runs}: {len(vals)} positions, "
              f"mean logprob {statistics.fmean(vals):.4f}")
    n = min(len(v) for v in all_runs)
    all_runs = [v[:n] for v in all_runs]
    env_min = [min(v[i] for v in all_runs) for i in range(n)]
    env_max = [max(v[i] for v in all_runs) for i in range(n)]
    mean_track = [statistics.fmean([v[i] for v in all_runs])
                  for i in range(n)]
    means = [-statistics.fmean(v) for v in all_runs]
    golden = {
        "prompt_sha": __import__("hashlib").sha256(
            PROBE_PROMPT.encode()).hexdigest(),
        "runs": runs,
        "env_min": env_min,
        "env_max": env_max,
        "mean_track": mean_track,
        "mean_nll": statistics.fmean(means),
        "mean_nll_std": statistics.pstdev(means),
        "raw_runs": all_runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(golden, f)
    print(f"golden written: {out_path} ({n} positions, mean NLL "
          f"{golden['mean_nll']:.4f} ± {golden['mean_nll_std']:.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-golden":
        capture_golden(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        module_main(globals())
