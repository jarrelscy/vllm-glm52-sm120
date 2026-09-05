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


def _garbage_(t: torch.Tensor, seed: int) -> torch.Tensor:
    """Fill a tensor in place with adversarial raw bit patterns.

    Mix of uniform random int bits (hits NaN/Inf/denormal fp16/bf16 payloads)
    and all-ones (0xFFFF = fp16 -NaN), so any kernel read of a byte the launch
    does not own turns into a poisoned value that would visibly perturb (or
    NaN-poison) accumulated math.
    """
    g = torch.Generator().manual_seed(seed)
    i16 = t.view(-1).view(torch.int16)
    if seed % 3 == 0:
        i16.fill_(-1)  # 0xFFFF everywhere
    else:
        i16.copy_(torch.randint(-32768, 32767, (i16.numel(),),
                                dtype=torch.int16, generator=g))
    return t


class Poisoner:
    """Randomized allocator-pool poison between launches.

    The serving-side failure mode fixed-input stress cannot catch: a kernel
    reading bytes whose CONTENT is whatever the previous request left behind
    (tail-of-row padding, torch.empty outputs, freed intermediates). Between
    probe launches we (a) run a randomized PREDECESSOR launch of a different
    logical shape through the same op chain, and (b) allocate/garbage/free a
    spread of blocks sized like the probe's own intermediates, so the probe's
    torch.empty allocations land on memory full of varying garbage. The probe
    itself runs on bitwise-fixed logical inputs; any output flip = a read of
    memory the launch does not own.
    """

    def __init__(self, ext, w, dev, seed=1234):
        self.ext, self.w, self.dev = ext, w, dev
        self.it = 0
        self.seed = seed

    def _pool_poison(self, sizes_bytes):
        held = []
        for i, nb in enumerate(sizes_bytes):
            t = torch.empty(nb // 2, dtype=torch.float16, device=self.dev)
            _garbage_(t, self.seed + self.it * 131 + i)
            held.append(t)
        del held  # freed poisoned blocks -> pool

    def predecessor(self):
        """Random-shape random-content launch through the full decode chain."""
        self.it += 1
        g = torch.Generator().manual_seed(self.seed + self.it)
        tokens = int(torch.randint(1, 9, (1,), generator=g))
        S = tokens * TOPK
        hot = torch.rand(S, generator=g) < float(torch.rand(1, generator=g))
        a_ids = torch.where(hot, torch.tensor(-1),
                            torch.randint(0, N_COLD, (S,), generator=g)).int()
        n_ids = torch.where(hot, torch.randint(0, N_HOT, (S,), generator=g),
                            torch.tensor(-1)).int()
        # a couple of masked slots, like MTP verify
        a_ids[S - 1] = -1
        n_ids[S - 1] = -1
        a_ids, n_ids = a_ids.to(self.dev), n_ids.to(self.dev)
        x = torch.empty(tokens, H, dtype=torch.float16, device=self.dev)
        _garbage_(x, self.seed + self.it * 7 + 1)
        # keep magnitudes finite-ish half the time; leave raw NaN garbage the
        # other half (NaN inputs must also stay confined to the predecessor)
        if self.it % 2:
            x.copy_((torch.randn(tokens, H,
                                 generator=g) / 8).to(torch.float16))
        w = self.w
        h13 = self.ext.hybrid_moe_gemv(
            x, w["w13_codes"], w["w13_cbs"], w["w13_scales"],
            a_ids, w["w13_packed"], w["w13_bscale"], w["w13_scale2"],
            n_ids, TOPK)
        hact = self.ext.silu_mul(h13)
        out = self.ext.hybrid_moe_gemv(
            hact, w["w2_codes"], w["w2_cbs"], w["w2_scales"], a_ids,
            w["w2_packed"], w["w2_bscale"], w["w2_scale2"], n_ids)
        wts = torch.rand(S, generator=g).float().to(self.dev)
        self.ext.moe_combine(out, wts, TOPK, 2)
        # poison blocks shaped like the probe's own intermediates
        self._pool_poison([1 << 16, 1 << 18, 1 << 20, 3 << 20, 1 << 22])


def run_case_poison(ext, w, a_ids, n_ids, key, x, row_div, iters, poisoner,
                    label, tail=False, pad_tokens=0):
    """Fixed logical inputs; randomized predecessor + pool poison between
    launches; bit-compare probe outputs. tail=True runs the full
    gemv->silu->gemv->combine chain (covers AQLM_FUSED_SILU /
    AQLM_FUSED_COMBINE / AQLM_GEMV_BF16IN composition), else one gemv.

    pad_tokens > 0 appends garbage rows (re-poisoned each iter) after the
    fixed logical rows, mimicking CUDA-graph padded decode batches; only the
    logical rows are compared (checks cross-row bleed).
    """
    S_logical = a_ids.shape[0]
    g = torch.Generator().manual_seed(77)
    pad_S = pad_tokens * TOPK

    def build_inputs(it):
        if pad_S == 0:
            return x, a_ids, n_ids, None
        xt = x if row_div > 1 else x
        rows = xt.shape[0]
        pad_rows = pad_tokens if row_div > 1 else pad_S
        xp = torch.empty(rows + pad_rows, xt.shape[1], dtype=xt.dtype,
                         device=xt.device)
        xp[:rows] = xt
        _garbage_(xp[rows:], 900 + it)
        gp = torch.Generator().manual_seed(it)
        ap = torch.randint(0, N_COLD, (pad_S,), generator=gp).int().to(x.device)
        np_ = torch.full((pad_S,), -1, dtype=torch.int32, device=x.device)
        # mix formats in the pad rows too
        sel = torch.rand(pad_S, generator=gp) < 0.5
        np_[sel.to(x.device)] = torch.randint(
            0, N_HOT, (int(sel.sum()),), generator=gp).int().to(x.device)
        ap[sel.to(x.device)] = -1
        return (xp, torch.cat([a_ids, ap]), torch.cat([n_ids, np_]), None)

    def call(xi, ai, ni):
        args13 = (w[f"{key}_codes"], w[f"{key}_cbs"], w[f"{key}_scales"], ai,
                  w[f"{key}_packed"], w[f"{key}_bscale"], w[f"{key}_scale2"],
                  ni)
        if not tail:
            if row_div > 1:
                return ext.hybrid_moe_gemv(xi, *args13, row_div)
            return ext.hybrid_moe_gemv(xi, *args13)
        h13 = (ext.hybrid_moe_gemv(xi, *args13, row_div) if row_div > 1
               else ext.hybrid_moe_gemv(xi, *args13))
        hact = ext.silu_mul(h13)
        out = ext.hybrid_moe_gemv(
            hact, w["w2_codes"], w["w2_cbs"], w["w2_scales"], ai,
            w["w2_packed"], w["w2_bscale"], w["w2_scale2"], ni)
        return ext.moe_combine(out, wts_fixed, TOPK, 2)

    wts_fixed = torch.rand(a_ids.shape[0] + pad_S,
                           generator=g).float().to(x.device)

    xi, ai, ni = build_inputs(0)[:3]
    ref = call(xi, ai, ni)[: (S_logical if not tail else S_logical // TOPK)]
    ref = ref.clone()
    torch.cuda.synchronize()

    bad_iters = 0
    first = None
    for it in range(iters):
        poisoner.predecessor()
        xi, ai, ni = build_inputs(it + 1)[:3]
        cur = call(xi, ai, ni)[
            : (S_logical if not tail else S_logical // TOPK)]
        if not torch.equal(cur.view(torch.int16), ref.view(torch.int16)):
            bad_iters += 1
            if first is None:
                d = (cur.view(torch.int16) != ref.view(torch.int16))
                first = (it, int(d.sum()),
                         d.nonzero()[:4].flatten().tolist())
    torch.cuda.synchronize()
    tag = "RACE" if bad_iters else "ok  "
    extra = f" first={first}" if first else ""
    print(f"  [{tag}] {label}: {bad_iters}/{iters} divergent iters "
          f"(poison){extra}")
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
    ap.add_argument("--poison", action="store_true",
                    help="randomized-poison variant: randomized predecessor "
                         "launches + garbage-filled allocator pool between "
                         "fixed-input probe launches (catches reads of "
                         "memory whose content depends on prior requests)")
    ap.add_argument("--pad-tokens", type=int, default=4,
                    help="poison mode: garbage token rows appended after the "
                         "fixed logical rows (CUDA-graph padded batches)")
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
    if args.poison:
        poisoner = Poisoner(ext, w, dev)
        for key in shapes:
            for dt in dtypes:
                for rd in rowdivs:
                    if key == "w2" and rd > 1:
                        continue  # prod w2 is always row_div=1
                    xin = (x13c if rd > 1 else (x13e if key == "w13" else x2))
                    for pad in (0, args.pad_tokens):
                        total_bad += run_case_poison(
                            ext, w, a_ids, n_ids, key, xin.to(dt), rd,
                            args.iters, poisoner,
                            f"{key} {str(dt)[6:]} rowdiv={rd} pad={pad}",
                            pad_tokens=pad)
        # full decode tail chain: gemv -> silu_mul -> gemv -> moe_combine
        # (AQLM_FUSED_SILU / AQLM_FUSED_COMBINE / BF16IN / ROWMAP composed)
        for dt in dtypes:
            for rd in rowdivs:
                xin = (x13c if rd > 1 else x13e).to(dt)
                for pad in (0, args.pad_tokens):
                    total_bad += run_case_poison(
                        ext, w, a_ids, n_ids, "w13", xin, rd, args.iters,
                        poisoner,
                        f"tail-chain {str(dt)[6:]} rowdiv={rd} pad={pad}",
                        tail=True, pad_tokens=pad)
        print(f"TOTAL divergent iters (poison): {total_bad}")
        sys.exit(1 if total_bad else 0)
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
