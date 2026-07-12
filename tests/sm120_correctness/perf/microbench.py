# SPDX-License-Identifier: Apache-2.0
"""NON-GATING perf harness: kernel-only timing of the hybrid MoE gemv.

Promoted from the scratchpad microbench.py (grouped-gemv investigation).
Use it to time a variant AFTER it passes the Tier-1 bit-exact gates —
never instead of them.  Run inside the glm52-sm120 container:

  source /opt/vllm/.venv/bin/activate
  cd tests/sm120_correctness
  PYTHONPATH=.:../.. python perf/microbench.py [--cflags -DNVFP4_LUT256=1]

Prints per-shape us/launch for the shipped build and (optionally) a
variant build side by side on the REAL verify shapes
(w13 [1024, 6144] s2n=2, w2 [6144, 512] s2n=1, S=32, ~15 unique experts).
"""

import argparse
import pathlib
import sys
import time

_SUITE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import kernels as K  # noqa: E402


def make_verify_case(shape_key: str, uniq: int, seed: int = 0):
    """Production-like verify routing: S=32 slots over `uniq` experts."""
    p = K.SHAPES[shape_key]
    g = np.random.default_rng(seed)
    case = K.make_case(shape_key, "mixed", S=32, books=1, seed=seed)
    # reroute slots to only `uniq` distinct experts (verify redundancy)
    active_a = np.flatnonzero(case["aqlm_ids"] >= 0)
    if active_a.size:
        pool = g.choice(case["codes"].shape[0],
                        size=min(uniq, case["codes"].shape[0]), replace=False)
        case["aqlm_ids"][active_a] = g.choice(pool, size=active_a.size)
    print(f"  case {shape_key}: M={p['M']} K={p['K']} S=32 uniq~{uniq}")
    return case


def timeit(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cflags", nargs="*", default=None,
                    help="build a variant ext with these extra cuda cflags")
    ap.add_argument("--uniq", type=int, default=15)
    args = ap.parse_args()

    idx = K.pick_gpu()
    if idx is None:
        sys.exit("no GPU with enough free memory (box rules)")
    torch.cuda.set_device(idx)
    dev = f"cuda:{idx}"

    base = K.load_ext()
    var = K.load_ext(cflags=tuple(args.cflags)) if args.cflags else None

    for shape in ("w13_real", "w2_real"):
        case = make_verify_case(shape, args.uniq)
        tc = K.case_to_torch(case, dev)

        def run(ext):
            return lambda: ext.hybrid_moe_gemv(
                tc["x"], tc["codes"], tc["codebooks"], tc["scales"],
                tc["aqlm_ids"], tc["packed"], tc["bscale"], tc["scale2"],
                tc["nv_ids"])

        t_base = timeit(run(base))
        line = f"  shipped: {t_base:8.1f} us"
        if var is not None:
            same = torch.equal(run(base)(), run(var)())
            t_var = timeit(run(var))
            line += (f"   variant: {t_var:8.1f} us  speedup "
                     f"{t_base / t_var:5.2f}x  bitexact={same}")
        print(line)


if __name__ == "__main__":
    main()
