#!/usr/bin/env python3
"""V4 (AQLM_GEMV_PIPELINE=1) data-race stress harness.

The V4 HybridMatVecMoEV4 kernel is deterministic by construction (fixed
launch config, no atomics, no cross-block communication), so running it
repeatedly on FIXED inputs must produce bit-identical outputs every time.
This harness hammers it while 2-3 side streams generate heavy memory /
SM traffic (large D2D copies + matmuls) to reproduce the memory-latency
jitter of live serving, and optionally under torch.cuda.CUDAGraph replay.

Any bitwise mismatch across iterations = the race.

Usage (inside the prod image, GPU 0, <1 GiB):
  python3 kbench/stress_v4_race.py --iters 300 --dtype bf16 --rowdiv 8
  python3 kbench/stress_v4_race.py --graph --iters 300
  python3 kbench/stress_v4_race.py --sanitize   # 1 iter, tiny, no traffic

Env: KB_SRC / KB_NAME / KB_CFLAGS as in bench_gemv.py.
"""
import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".torchext-race"))

import torch  # noqa: E402

from bench_gemv import (  # noqa: E402
    H, I, TOPK, build, set_pipeline, set_v3,
)

# Small expert counts: the race is intra-kernel, expert count only scales
# weight bytes. Keeps total allocations well under 1 GiB next to prod.
# --prod-weights switches to the real per-layer expert counts (~800 MiB)
# for realistic L2 thrash / gather+cp.async latency.
N_HOT, N_COLD = 12, 24


def make_weights_small(dev, seed=0):
    global N_HOT, N_COLD
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r16(*s):
        return (torch.randn(*s, generator=g) / 8).to(torch.float16).to(dev)

    w = {}
    w["w13_codes"] = torch.randint(-32768, 32767, (N_COLD, 1, 2 * I, H // 8),
                                   dtype=torch.int16, generator=g).to(dev)
    w["w13_cbs"] = r16(1, 65536, 8)
    w["w13_scales"] = r16(N_COLD, 2 * I)
    w["w13_packed"] = torch.randint(0, 255, (N_HOT, 2 * I, H // 2),
                                    dtype=torch.uint8, generator=g).to(dev)
    w["w13_bscale"] = torch.randint(100, 126, (N_HOT, 2 * I, H // 16),
                                    dtype=torch.uint8, generator=g).to(dev)
    w["w13_scale2"] = torch.rand(N_HOT, 2, generator=g).float().to(dev)
    w["w2_codes"] = torch.randint(-32768, 32767, (N_COLD, 1, H, I // 8),
                                  dtype=torch.int16, generator=g).to(dev)
    w["w2_cbs"] = r16(1, 65536, 8)
    w["w2_scales"] = r16(N_COLD, H)
    w["w2_packed"] = torch.randint(0, 255, (N_HOT, H, I // 2),
                                   dtype=torch.uint8, generator=g).to(dev)
    w["w2_bscale"] = torch.randint(100, 126, (N_HOT, H, I // 16),
                                   dtype=torch.uint8, generator=g).to(dev)
    w["w2_scale2"] = torch.rand(N_HOT, 1, generator=g).float().to(dev)
    return w


def make_slots_small(tokens, hot_frac, dev, seed=1):
    g = torch.Generator().manual_seed(seed)
    S = tokens * TOPK
    hot = torch.rand(S, generator=g) < hot_frac
    a_ids = torch.where(hot, torch.tensor(-1),
                        torch.randint(0, N_COLD, (S,), generator=g)).int()
    n_ids = torch.where(hot, torch.randint(0, N_HOT, (S,), generator=g),
                        torch.tensor(-1)).int()
    # a couple of masked slots (both < 0), as MTP verify produces
    a_ids[S - 1] = -1
    n_ids[S - 1] = -1
    return a_ids.to(dev), n_ids.to(dev)


class Traffic:
    """Side-stream memory + SM traffic generator (~jitter of live serving)."""

    def __init__(self, dev, n_streams=3, mb=48):
        self.streams = [torch.cuda.Stream(device=dev)
                        for _ in range(n_streams)]
        n = mb * 1024 * 1024 // 2
        self.bufs = [(torch.randn(n, dtype=torch.float16, device=dev),
                      torch.empty(n, dtype=torch.float16, device=dev))
                     for _ in self.streams]
        self.mm = [torch.randn(1536, 1536, dtype=torch.float16, device=dev)
                   for _ in self.streams]

    def pump(self, depth=8):
        for s, (a, b), m in zip(self.streams, self.bufs, self.mm):
            with torch.cuda.stream(s):
                for _ in range(depth):
                    b.copy_(a, non_blocking=True)
                    torch.mm(m, m)

    def drain(self):
        for s in self.streams:
            s.synchronize()


def run_case(ext, w, a_ids, n_ids, key, x, row_div, iters, traffic, graph,
             label):
    args = (w[f"{key}_codes"], w[f"{key}_cbs"], w[f"{key}_scales"], a_ids,
            w[f"{key}_packed"], w[f"{key}_bscale"], w[f"{key}_scale2"],
            n_ids)

    def call():
        if row_div > 1:
            return ext.hybrid_moe_gemv(x, *args, row_div)
        return ext.hybrid_moe_gemv(x, *args)

    if graph:
        # warm up allocator + kernel on a side stream, then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = g_out = call()
        run = g.replay
        fetch = lambda: g_out  # noqa: E731
    else:
        run, fetch = call, None

    torch.cuda.synchronize()
    if traffic:
        traffic.pump()
    r = run()
    ref = (fetch() if fetch else r).clone()
    torch.cuda.synchronize()

    distinct = [ref]
    bad_iters = 0
    if iters >= 2000:
        # fast path: accumulate mismatch counts on-GPU, sync rarely
        nbad_acc = torch.zeros((), dtype=torch.int64, device=x.device)
        refi = ref.view(torch.int16)
        for it in range(iters):
            if traffic and it % 8 == 0:
                traffic.pump()
            r = run()
            cur = fetch() if fetch else r
            nbad_acc += (cur.view(torch.int16) != refi).any().long()
        if traffic:
            traffic.drain()
        torch.cuda.synchronize()
        bad_iters = int(nbad_acc)
        tag = "RACE" if bad_iters else "ok  "
        print(f"  [{tag}] {label}: {bad_iters}/{iters} divergent iters "
              f"(fast mode)")
        return bad_iters
    for it in range(iters):
        if traffic and it % 4 == 0:
            traffic.pump()
        r = run()
        cur = fetch() if fetch else r
        if not torch.equal(cur.view(torch.int16), ref.view(torch.int16)):
            bad_iters += 1
            if not any(torch.equal(cur.view(torch.int16), d.view(torch.int16))
                       for d in distinct):
                distinct.append(cur.clone())
                nbad = int((cur.view(torch.int16)
                            != ref.view(torch.int16)).sum())
                md = (cur.float() - ref.float()).abs().max().item()
                if len(distinct) <= 4:
                    idx = (cur.view(torch.int16)
                           != ref.view(torch.int16)).nonzero()[:4]
                    print(f"    iter {it}: NEW divergent output "
                          f"({nbad} el differ, maxabs {md:.4g}, "
                          f"first at {idx.flatten().tolist()[:8]})")
    if traffic:
        traffic.drain()
    torch.cuda.synchronize()
    tag = "RACE" if bad_iters else "ok  "
    print(f"  [{tag}] {label}: {bad_iters}/{iters} divergent iters, "
          f"{len(distinct)} distinct outputs")
    return bad_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--hot-frac", type=float, default=0.6)
    ap.add_argument("--shape", default="both", choices=["w13", "w2", "both"])
    ap.add_argument("--dtype", default="both",
                    choices=["fp16", "bf16", "both"])
    ap.add_argument("--rowdiv", default="both", choices=["1", "8", "both"])
    ap.add_argument("--graph", action="store_true",
                    help="run under CUDAGraph capture+replay")
    ap.add_argument("--no-traffic", action="store_true")
    ap.add_argument("--sanitize", action="store_true",
                    help="tiny single-config run for compute-sanitizer")
    ap.add_argument("--v2", action="store_true",
                    help="stress the V2/V3 (PIPE=0) path instead")
    ap.add_argument("--prod-weights", action="store_true",
                    help="real per-layer expert counts (~800 MiB) for "
                         "realistic L2 thrash")
    args = ap.parse_args()
    if args.prod_weights:
        global N_HOT, N_COLD
        N_HOT, N_COLD = 61, 195

    dev = "cuda:0"
    torch.cuda.set_device(dev)
    src = os.environ.get(
        "KB_SRC", str(HERE.parent / "csrc/quantization/aqlm_moe/aqlm_moe_v2.cu"))
    cflags = [f for f in os.environ.get("KB_CFLAGS", "").split(",") if f]
    if os.environ.get("KB_LINEINFO", "0") not in ("", "0"):
        cflags.append("-lineinfo")  # NB: can perturb ptxas scheduling;
        # default OFF so the tested SASS matches the prod -O3 build.
    name = "kb_race_" + os.environ.get("KB_NAME", "cur")
    print(f"building {src} ...")
    ext = build(src, name, cflags)

    set_v3(False)
    set_pipeline(not args.v2)

    w = make_weights_small(dev)
    a_ids, n_ids = make_slots_small(args.tokens, args.hot_frac, dev)
    g = torch.Generator().manual_seed(3)
    S = args.tokens * TOPK
    x13c = (torch.randn(args.tokens, H, generator=g) / 8).to(torch.float16).to(dev)
    x13e = x13c.repeat_interleave(TOPK, dim=0)
    x2 = (torch.randn(S, I, generator=g) / 8).to(torch.float16).to(dev)

    traffic = None if (args.no_traffic or args.sanitize) else Traffic(dev)
    alloc = torch.cuda.memory_allocated(dev) / (1 << 20)
    print(f"allocated {alloc:.0f} MiB on {dev}  pipeline="
          f"{os.environ['AQLM_GEMV_PIPELINE']} graph={args.graph} "
          f"traffic={traffic is not None}")

    if args.sanitize:
        # one launch per (shape x dtype x rowdiv) for racecheck/initcheck
        for key, xin, rd in (("w13", x13e, 1), ("w13", x13c, TOPK),
                             ("w2", x2, 1)):
            for dt in (torch.float16, torch.bfloat16):
                run_case(ext, w, a_ids, n_ids, key, xin.to(dt), rd, 1, None,
                         False, f"{key} {dt} rowdiv={rd}")
        return

    shapes = ["w13", "w2"] if args.shape == "both" else [args.shape]
    dtypes = ({"fp16": [torch.float16], "bf16": [torch.bfloat16]}
              .get(args.dtype, [torch.float16, torch.bfloat16]))
    rowdivs = {"1": [1], "8": [TOPK]}.get(args.rowdiv, [1, TOPK])

    total_bad = 0
    for key in shapes:
        for dt in dtypes:
            for rd in rowdivs:
                if key == "w2" and rd > 1:
                    continue  # prod w2 is always row_div=1
                xin = (x13c if rd > 1 else (x13e if key == "w13" else x2))
                total_bad += run_case(
                    ext, w, a_ids, n_ids, key, xin.to(dt), rd, args.iters,
                    traffic, args.graph,
                    f"{key} {str(dt)[6:]} rowdiv={rd}"
                    f"{' graph' if args.graph else ''}")
    print(f"TOTAL divergent iters: {total_bad}")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
