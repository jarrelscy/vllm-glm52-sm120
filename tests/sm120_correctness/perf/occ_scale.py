# SPDX-License-Identifier: Apache-2.0
"""NON-GATING perf harness: slot-count (occupancy) scaling of the fused
hybrid MoE gemv.

Promoted from the scratchpad occ_scale.py (Round-4 saturation analysis:
w2 t/slot flat ~3.6us for S>=8; w13 minimized exactly at S=32 — the GPU
is throughput-saturated at the operating point, NOT occupancy-bound).
Re-run after kernel changes to see whether the saturation knee moved.

  PYTHONPATH=.:../.. python perf/occ_scale.py
"""

import pathlib
import sys
import time

_SUITE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUITE))

import torch  # noqa: E402

from common import kernels as K  # noqa: E402


def timeit(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6


def main():
    idx = K.pick_gpu()
    if idx is None:
        sys.exit("no GPU with enough free memory (box rules)")
    torch.cuda.set_device(idx)
    dev = f"cuda:{idx}"
    ext = K.load_ext()

    for shape in ("w13_real", "w2_real"):
        print(f"== {shape} ({K.SHAPES[shape]}) ==")
        print(f"{'S':>4} {'us/launch':>10} {'us/slot':>9}")
        for s in (1, 4, 8, 16, 32, 64):
            case = K.make_case(shape, "mixed", S=s, books=1, seed=s)
            tc = K.case_to_torch(case, dev)
            t = timeit(lambda: ext.hybrid_moe_gemv(
                tc["x"], tc["codes"], tc["codebooks"], tc["scales"],
                tc["aqlm_ids"], tc["packed"], tc["bscale"], tc["scale2"],
                tc["nv_ids"]))
            print(f"{s:>4} {t:>10.1f} {t / s:>9.2f}")


if __name__ == "__main__":
    main()
