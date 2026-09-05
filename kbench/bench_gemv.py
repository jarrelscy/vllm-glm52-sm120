#!/usr/bin/env python3
"""Standalone decode-gemv microbench for DECODE-K (real tp4-1m-mtp shapes).

Replicates the production fused hybrid_moe_gemv calls per MoE layer at
MTP-verify batch (T tokens x top_k=8 slots):
  w13: [M=1024=2*512, K=6144], w13 books=1, mix hot(NVFP4)/cold(AQLM)
  w2 : [M=6144, K=512],  w2c books=1, same slot mix

Env:
  KB_CFLAGS   extra -D flags, comma separated (e.g. "-DAQLM_MLP=4")
  KB_SRC      kernel .cu (default: this worktree's aqlm_moe_v2.cu)
  KB_NAME     extension name suffix
Args: --tokens N (default 4 = 1+ns3 verify) --hot-frac F --iters --compile-only
      --shape w13|w2|both --mix prod|aqlm|nv --check (vs shipped kernel)
      --pipeline (time the AQLM_GEMV_PIPELINE=1 V4 path; with --check, also
                  validates it against an fp32 dequant reference at the same
                  tolerance as the env-off kernel)
      --bf16in  (tail-fusion: validate bf16-activation gemv input against the
                 fp16-cast path — must be BIT-exact on V2, V3 and V4, incl an
                 all-65536-bf16-bit-pattern activation sweep — and time it)
      --combine (tail-fusion: validate moe_combine against the eager
                 .float()*w -> sum(dim=1) -> .to(dtype) tail — bit-exact in
                 all three output modes — and time both)
      --rowmap  (tail-fusion-2: validate hybrid_moe_gemv row_div (in-kernel
                 slot -> slot//top_k activation row mapping) against the
                 x.repeat_interleave(top_k) expansion it replaces — must be
                 BIT-exact on V2, V3 (dedup+lane-rows) and V4, fp16 + bf16,
                 both shapes, T=1..512 — and time gemv + the saved kernel)
      --silu    (tail-fusion-2: validate the fused silu_mul kernel against
                 torch's eager F.silu(x[:, :m]) * x[:, m:] pair — bit-exact
                 incl all-65536-bit-pattern gate and up sweeps, -0.0,
                 denormals, inf/nan — and time both)
      --compose (tail-fusion-2: full decode-tail emulation with V4 pipeline
                 + bf16 input + fused combine: {repeat_interleave + eager
                 silu} vs {rowmap + fused silu} must produce bit-identical
                 final combined output)
"""
import argparse, os, pathlib, sys

import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
H, I, TOPK = 6144, 512, 8            # TP4 shard: moe_inter 2048/4
N_HOT, N_COLD = 61, 195              # modal production layer mix


def build(src, name, cflags):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return load(name=name, sources=[str(src)],
                extra_cuda_cflags=["-O3", *cflags], verbose=False)


def make_weights(dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def r16(*s):
        return (torch.randn(*s, generator=g) / 8).to(torch.float16).to(dev)
    w = {}
    # w13: AQLM cold [nB,1,2I,H/8] + NVFP4 hot [nA,2I,H/2]
    w["w13_codes"] = torch.randint(-32768, 32767, (N_COLD, 1, 2 * I, H // 8),
                                   dtype=torch.int16, generator=g).to(dev)
    w["w13_cbs"] = r16(1, 65536, 8)
    w["w13_scales"] = r16(N_COLD, 2 * I)
    w["w13_packed"] = torch.randint(0, 255, (N_HOT, 2 * I, H // 2),
                                    dtype=torch.uint8, generator=g).to(dev)
    w["w13_bscale"] = torch.randint(100, 126, (N_HOT, 2 * I, H // 16),
                                    dtype=torch.uint8, generator=g).to(dev)
    w["w13_scale2"] = torch.rand(N_HOT, 2, generator=g).float().to(dev)
    # w2 cold: [nC,1,H,I/8]; hot [nA,H,I/2]
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


def make_slots(tokens, hot_frac, mix, dev, seed=1):
    g = torch.Generator().manual_seed(seed)
    S = tokens * TOPK
    if mix == "aqlm":
        hot = torch.zeros(S, dtype=torch.bool)
    elif mix == "nv":
        hot = torch.ones(S, dtype=torch.bool)
    else:
        hot = torch.rand(S, generator=g) < hot_frac
    a_ids = torch.where(hot, torch.tensor(-1),
                        torch.randint(0, N_COLD, (S,), generator=g)).int()
    n_ids = torch.where(hot, torch.randint(0, N_HOT, (S,), generator=g),
                        torch.tensor(-1)).int()
    if mix == "dup":
        # adversarial for dedup: >4 duplicates of one expert in both formats,
        # plus masked slots (both ids < 0)
        a_ids = torch.tensor([3] * 7 + [-1] * (S - 7)).int()
        n_ids = torch.full((S,), -1, dtype=torch.int32)
        for i in range(7, S):
            if i % 3 == 0:
                n_ids[i] = 5
            elif i % 3 == 1:
                a_ids[i] = (i // 3) % N_COLD
            # else: masked (zero-fill)
    elif mix == "realdup":
        # simulate MTP verify: `tokens` tokens, ~50% adjacent expert overlap
        e = torch.randint(0, N_HOT + N_COLD, (TOPK,), generator=g)
        ids = [e]
        for _ in range(tokens - 1):
            keep = torch.rand(TOPK, generator=g) < 0.5
            nxt = torch.where(
                keep, ids[-1],
                torch.randint(0, N_HOT + N_COLD, (TOPK,), generator=g))
            ids.append(nxt)
        flat = torch.cat(ids)[:S]
        is_hot = flat < N_HOT
        a_ids = torch.where(is_hot, torch.tensor(-1), flat - N_HOT).int()
        n_ids = torch.where(is_hot, flat, torch.tensor(-1)).int()
    return a_ids.to(dev), n_ids.to(dev)


FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def fp32_reference(x, w, a_ids, n_ids, key):
    """fp32 dequant + fp32 matmul reference for one projection.

    Matches the kernel's math (scale applied after the dot) but with all
    arithmetic in fp32, so both the shipped kernel and the V4 path can be
    held to the same tolerance against it.
    """
    dev = x.device
    S, K = x.shape
    M = w[f"{key}_codes"].shape[2]
    cb = w[f"{key}_cbs"][0].float()                       # [65536, 8]
    scales = w[f"{key}_scales"].float()                   # [nC, M]
    scale2 = w[f"{key}_scale2"]                           # [nH, s2n]
    s2n = scale2.shape[1]
    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device=dev)
    xf = x.float()
    out = torch.zeros(S, M, dtype=torch.float32, device=dev)
    rows = (torch.arange(M, device=dev) * s2n) // M
    for s in range(S):
        a, nv = int(a_ids[s]), int(n_ids[s])
        if a >= 0:
            idx = (w[f"{key}_codes"][a, 0].int() & 0xFFFF).long()  # [M, K/8]
            W = cb[idx.reshape(-1)].reshape(M, K)
            out[s] = (W @ xf[s]) * scales[a]
        elif nv >= 0:
            p = w[f"{key}_packed"][nv].int()              # [M, K/2]
            W = torch.stack([lut[p & 0xF], lut[p >> 4]], -1).reshape(M, K)
            bs = w[f"{key}_bscale"][nv].view(torch.float8_e4m3fn).float()
            Wb = (W.reshape(M, K // 16, 16) * bs[:, :, None]).reshape(M, K)
            out[s] = (Wb @ xf[s]) * scale2[nv][rows]
    return out


def set_pipeline(on):
    os.environ["AQLM_GEMV_PIPELINE"] = "1" if on else "0"


def set_v3(on):
    os.environ["GLM_MOE_DEDUP"] = "4" if on else "0"
    os.environ["GLM_MOE_LANE_ROWS"] = "1" if on else "0"


def check_bf16in(ext, w, a_ids, n_ids, x13, x2, iters):
    """bf16 activations must be BIT-identical to feeding x.to(fp16)."""
    dev = x13.device
    failed = False
    cases = [("w13", x13, "w13"), ("w2 ", x2, "w2")]
    # all-65536-bit-pattern sweep (covers every bf16 value incl inf/nan and
    # values that overflow fp16): staging conversion must match torch's cast
    # for every input bit pattern, so the outputs stay bit-identical.
    npat = min(65536, x13.numel())  # full coverage needs >= 11 tokens
    pat13 = torch.zeros(x13.numel(), dtype=torch.int16)
    pat13[:npat] = torch.arange(npat, dtype=torch.int32).to(torch.int16)
    pat13 = pat13.view(torch.bfloat16).reshape(x13.shape).to(dev)
    for path, pre, post in (
        ("V2", lambda: (set_pipeline(False), set_v3(False)), None),
        ("V3", lambda: (set_pipeline(False), set_v3(True)),
         lambda: set_v3(False)),
        ("V4", lambda: (set_pipeline(True), set_v3(False)),
         lambda: set_pipeline(False)),
    ):
        pre()
        for nm, x, k in cases:
            for tag, xb in (("rand", x.to(torch.bfloat16)),
                            ("bitsweep", pat13 if k == "w13" else
                             pat13.reshape(-1)[: x.numel()].reshape(x.shape))):
                args = (w[f"{k}_codes"], w[f"{k}_cbs"], w[f"{k}_scales"],
                        a_ids, w[f"{k}_packed"], w[f"{k}_bscale"],
                        w[f"{k}_scale2"], n_ids)
                got = ext.hybrid_moe_gemv(xb, *args)
                ref = ext.hybrid_moe_gemv(xb.to(torch.float16), *args)
                same = torch.equal(got.view(torch.int16),
                                   ref.view(torch.int16))
                print(f"  BF16IN {path} {nm} [{tag}]: bit-exact={same}")
                failed |= not same
        if post:
            post()
    if failed:
        sys.exit("BF16IN NOT BIT-EXACT")
    # timing: V4 w13 fp16-in vs bf16-in, plus the eliminated cast kernel
    set_pipeline(True)
    xb13 = x13.to(torch.bfloat16)
    args13 = (w["w13_codes"], w["w13_cbs"], w["w13_scales"], a_ids,
              w["w13_packed"], w["w13_bscale"], w["w13_scale2"], n_ids)
    t_fp16 = timed(lambda: ext.hybrid_moe_gemv(x13, *args13), iters)
    t_bf16 = timed(lambda: ext.hybrid_moe_gemv(xb13, *args13), iters)
    t_cast = timed(lambda: xb13.to(torch.float16), iters)
    set_pipeline(False)
    print(f"  BF16IN V4 w13: fp16-in {t_fp16:.2f} us  bf16-in {t_bf16:.2f} us"
          f"  (in-kernel convert {t_bf16 - t_fp16:+.2f})  saved cast kernel "
          f"{t_cast:.2f} us")


def check_combine(ext, tokens, iters, dev):
    """moe_combine must match the eager tail bit-for-bit in every mode."""
    g = torch.Generator().manual_seed(7)
    K, M = TOPK, H
    failed = False
    for T in (1, 2, 3, tokens, 64, 512):
        y = (torch.randn(T * K, M, generator=g) / 4).to(torch.float16).to(dev)
        wts = torch.randn(T, K, generator=g).float().to(dev)  # both signs
        ref32 = (y.view(T, K, M).float() * wts.view(T, K, 1)).sum(dim=1)
        # order probe: torch's reduce is the vec4-strided tree the kernel
        # replicates (acc[k % 4] += p_k, then ((a0+a1)+a2)+a3)
        tmp = y.view(T, K, M).float() * wts.view(T, K, 1)
        a = [tmp[:, i, :] + tmp[:, i + 4, :] for i in range(4)]
        man = ((a[0] + a[1]) + a[2]) + a[3]
        probe = torch.equal(ref32, man)
        wflat = wts.reshape(-1).contiguous()
        for mode, dt in ((0, torch.float32), (1, torch.float16),
                         (2, torch.bfloat16)):
            got = ext.moe_combine(y, wflat, K, mode)
            ref = ref32.to(dt)
            bits = {torch.float32: torch.int32}.get(dt, torch.int16)
            same = torch.equal(got.view(bits), ref.view(bits))
            md = (got.float() - ref.float()).abs().max().item()
            print(f"  COMBINE T={T:3d} mode={dt}: bit-exact={same} "
                  f"maxdiff={md}" + ("" if mode else f"  (order probe "
                                     f"{'PASS' if probe else 'FAIL'})"))
            failed |= not same
    # -0.0 edge: torch's zero-init flips an all-negative-zero sum to +0.0
    yz = torch.full((K, M), -0.0, dtype=torch.float16, device=dev)
    wz = torch.ones(K, dtype=torch.float32, device=dev)
    gz = ext.moe_combine(yz, wz, K, 0)
    rz = (yz.view(1, K, M).float() * wz.view(1, K, 1)).sum(dim=1)
    zsame = torch.equal(gz.view(torch.int32), rz.view(torch.int32))
    print(f"  COMBINE -0.0 edge: bit-exact={zsame}")
    failed |= not zsame
    if failed:
        sys.exit("COMBINE NOT BIT-EXACT vs torch tail")
    T = tokens
    y = (torch.randn(T * K, M, generator=g) / 4).to(torch.float16).to(dev)
    wts = torch.randn(T, K, generator=g).float().to(dev)
    wflat = wts.reshape(-1).contiguous()
    t_k = timed(lambda: ext.moe_combine(y, wflat, K, 2), iters)
    t_e = timed(
        lambda: (y.view(T, K, M).float()
                 * wts.view(T, K, 1)).sum(dim=1).to(torch.bfloat16), iters)
    print(f"  COMBINE kernel {t_k:.2f} us   eager tail {t_e:.2f} us   "
          f"({t_e - t_k:+.2f} us/layer)")


def check_rowmap(ext, w, hot_frac, iters, dev):
    """gemv(x, row_div=top_k) must be BIT-identical to
    gemv(x.repeat_interleave(top_k), row_div=1) on every kernel path."""
    failed = False
    g = torch.Generator().manual_seed(11)
    for path, pre, post in (
        ("V2", lambda: (set_pipeline(False), set_v3(False)), None),
        ("V3", lambda: (set_pipeline(False), set_v3(True)),
         lambda: set_v3(False)),
        ("V4", lambda: (set_pipeline(True), set_v3(False)),
         lambda: set_pipeline(False)),
    ):
        pre()
        for T in (1, 2, 3, 4, 64, 512):
            mixes = ("prod",) if T > 4 else ("prod", "dup", "realdup")
            for mix in mixes:
                a_ids, n_ids = make_slots(T, hot_frac, mix, dev, seed=T)
                for k, K in (("w13", H), ("w2", I)):
                    xt = ((torch.randn(T, K, generator=g) / 8)
                          .to(torch.float16).to(dev))
                    for dt in (torch.float16, torch.bfloat16):
                        xd = xt.to(dt)
                        xr = xd.repeat_interleave(TOPK, dim=0)
                        args = (w[f"{k}_codes"], w[f"{k}_cbs"],
                                w[f"{k}_scales"], a_ids, w[f"{k}_packed"],
                                w[f"{k}_bscale"], w[f"{k}_scale2"], n_ids)
                        got = ext.hybrid_moe_gemv(xd, *args, TOPK)
                        ref = ext.hybrid_moe_gemv(xr, *args)
                        same = torch.equal(got.view(torch.int16),
                                           ref.view(torch.int16))
                        if not same or (T == 4 and mix == "prod"):
                            print(f"  ROWMAP {path} {k:3s} T={T:3d} "
                                  f"mix={mix} {str(dt)[6:]}: "
                                  f"bit-exact={same}")
                        failed |= not same
        if post:
            post()
    if failed:
        sys.exit("ROWMAP NOT BIT-EXACT vs repeat_interleave path")
    print("  ROWMAP all paths/shapes/dtypes/T bit-exact PASS")
    # timing at the prod decode point: saved repeat_interleave kernel plus
    # V4 w13 gemv expanded-input vs rowmap
    set_pipeline(True)
    for T in (1, 4):
        a_ids, n_ids = make_slots(T, hot_frac, "prod", dev, seed=T)
        xt = ((torch.randn(T, H, generator=g) / 8).to(torch.float16).to(dev))
        xr = xt.repeat_interleave(TOPK, dim=0)
        args = (w["w13_codes"], w["w13_cbs"], w["w13_scales"], a_ids,
                w["w13_packed"], w["w13_bscale"], w["w13_scale2"], n_ids)
        t_rep = timed(lambda: xt.repeat_interleave(TOPK, dim=0), iters)
        t_exp = timed(lambda: ext.hybrid_moe_gemv(xr, *args), iters)
        t_map = timed(lambda: ext.hybrid_moe_gemv(xt, *args, TOPK), iters)
        print(f"  ROWMAP V4 w13 T={T}: gemv expanded {t_exp:.2f} us  "
              f"rowmap {t_map:.2f} us ({t_map - t_exp:+.2f})  saved "
              f"repeat_interleave {t_rep:.2f} us  net "
              f"{t_exp + t_rep - t_map:+.2f} us/layer")
    set_pipeline(False)


def _eager_silu_mul(x):
    d = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :d]) * x[..., d:]).contiguous()


def check_silu(ext, tokens, iters, dev):
    """silu_mul must match torch's eager silu + mul pair bit-for-bit."""
    g = torch.Generator().manual_seed(13)
    failed = False

    def cmp(tag, x):
        got = ext.silu_mul(x)
        ref = _eager_silu_mul(x)
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
        n_nan = int((got.isnan() & ref.isnan()).sum())
        print(f"  SILU {tag}: bit-exact={same}"
              + (f" (incl {n_nan} nan outputs)" if n_nan else ""))
        return same

    m = I  # 512: prod w13 output half-width
    # all-65536-bit-pattern gate sweep (covers -0.0, denormals, inf, nan)
    pats = (torch.arange(65536, dtype=torch.int32).to(torch.int16)
            .view(torch.float16).to(dev).reshape(-1, m))
    ones = torch.ones_like(pats)
    rnd = (torch.randn(pats.shape, generator=g) * 2).to(torch.float16).to(dev)
    failed |= not cmp("gate=allbits up=1   ",
                      torch.cat([pats, ones], -1).contiguous())
    failed |= not cmp("gate=allbits up=rand",
                      torch.cat([pats, rnd], -1).contiguous())
    failed |= not cmp("gate=rand up=allbits",
                      torch.cat([rnd, pats], -1).contiguous())
    # -0.0 / denormal edges on both halves
    z = torch.full((4, m), -0.0, dtype=torch.float16, device=dev)
    dn = (torch.randint(1, 1024, (4, m), generator=g).to(torch.int16)
          .view(torch.float16).to(dev))  # fp16 denormals
    for tag, a, b in (("gate=-0.0 up=rand ", z, rnd[:4]),
                      ("gate=rand up=-0.0 ", rnd[:4], z),
                      ("gate=denorm up=den", dn, -dn)):
        failed |= not cmp(tag, torch.cat([a, b], -1).contiguous())
    # prod shapes (vector kernel) + scalar-kernel odd width
    for T in (1, 2, 3, tokens, 64, 512):
        x = ((torch.randn(T * TOPK, 2 * m, generator=g) / 4)
             .to(torch.float16).to(dev))
        failed |= not cmp(f"T={T:3d} m={m}       ", x)
    xo = ((torch.randn(37, 2 * 12, generator=g) / 4)
          .to(torch.float16).to(dev))
    failed |= not cmp("scalar m=12         ", xo)
    if failed:
        sys.exit("SILU NOT BIT-EXACT vs torch eager pair")
    x = ((torch.randn(tokens * TOPK, 2 * m, generator=g) / 4)
         .to(torch.float16).to(dev))
    t_k = timed(lambda: ext.silu_mul(x), iters)
    t_e = timed(lambda: _eager_silu_mul(x), iters)
    print(f"  SILU kernel {t_k:.2f} us   eager silu+mul {t_e:.2f} us   "
          f"({t_e - t_k:+.2f} us/layer)")


def check_compose(ext, w, tokens, hot_frac, iters, dev):
    """Full decode-tail emulation of _apply_gemv (fused n_base==0 path) with
    the existing flags on (V4 pipeline, bf16 activations, fused combine):
    baseline {repeat_interleave + eager silu} vs {rowmap + fused silu} must
    agree bit-for-bit on the final combined [T, H] bf16 output."""
    g = torch.Generator().manual_seed(17)
    set_pipeline(True)
    failed = False
    for T in (1, 2, 4):
        a_ids, n_ids = make_slots(T, hot_frac, "prod", dev, seed=100 + T)
        x = (torch.randn(T, H, generator=g) / 8).to(torch.bfloat16).to(dev)
        wts = torch.randn(T, TOPK, generator=g).float().to(dev)
        wflat = wts.reshape(-1).contiguous()
        a13 = (w["w13_codes"], w["w13_cbs"], w["w13_scales"], a_ids,
               w["w13_packed"], w["w13_bscale"], w["w13_scale2"], n_ids)
        a2 = (w["w2_codes"], w["w2_cbs"], w["w2_scales"], a_ids,
              w["w2_packed"], w["w2_bscale"], w["w2_scale2"], n_ids)

        def tail(rowmap_silu):
            if rowmap_silu:
                h13 = ext.hybrid_moe_gemv(x, *a13, TOPK)
                hact = ext.silu_mul(h13)
            else:
                xr = x.repeat_interleave(TOPK, dim=0)
                h13 = ext.hybrid_moe_gemv(xr, *a13)
                hact = _eager_silu_mul(h13)
            y = ext.hybrid_moe_gemv(hact, *a2)
            return ext.moe_combine(y, wflat, TOPK, 2)  # bf16 out

        ref, got = tail(False), tail(True)
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
        print(f"  COMPOSE T={T} (V4+bf16in+fused-combine): rowmap+silu "
              f"bit-exact={same}")
        failed |= not same
        if T == tokens or T == 4:
            # best-of-3 alternating reps: a single pass right after the
            # bit-checks sees a cold L2/allocator state and can invert the
            # comparison by several us.
            t_off = min(timed(lambda: tail(False), iters) for _ in range(3))
            t_on = min(timed(lambda: tail(True), iters) for _ in range(3))
            print(f"  COMPOSE T={T} tail: flags-off {t_off:.2f} us  "
                  f"flags-on {t_on:.2f} us  ({t_on - t_off:+.2f} us/layer)")
    set_pipeline(False)
    if failed:
        sys.exit("COMPOSE NOT BIT-EXACT")


def timed(fn, iters, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--hot-frac", type=float, default=0.6)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--shape", default="both", choices=["w13", "w2", "both"])
    ap.add_argument("--mix", default="prod",
                    choices=["prod", "aqlm", "nv", "dup", "realdup"])
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="bit-compare vs the shipped kernel (cudagraphs-v2)")
    ap.add_argument("--prof", action="store_true",
                    help="single pass per shape (for ncu -c N capture)")
    ap.add_argument("--pipeline", action="store_true",
                    help="run/validate the AQLM_GEMV_PIPELINE=1 V4 path")
    ap.add_argument("--bf16in", action="store_true",
                    help="validate/time bf16-activation gemv input")
    ap.add_argument("--combine", action="store_true",
                    help="validate/time the fused top-k combine epilogue")
    ap.add_argument("--rowmap", action="store_true",
                    help="validate/time in-kernel slot->token row mapping")
    ap.add_argument("--silu", action="store_true",
                    help="validate/time the fused silu_mul mid-tail kernel")
    ap.add_argument("--compose", action="store_true",
                    help="all-flags-on decode-tail emulation bit-check")
    args = ap.parse_args()
    set_pipeline(False)  # bit-exact V2/V3 default unless --pipeline

    src = os.environ.get(
        "KB_SRC", str(REPO / "csrc/quantization/aqlm_moe/aqlm_moe_v2.cu"))
    cflags = [c for c in os.environ.get("KB_CFLAGS", "").split(",") if c]
    name = "kb_ext" + os.environ.get("KB_NAME", "")
    if cflags:
        name += "_" + "".join(
            ch for c in cflags for ch in c.lower() if ch.isalnum())
    ext = build(src, name, cflags)
    print(f"built {name} from {src} cflags={cflags}")
    if args.compile_only:
        return

    dev = "cuda:0"
    torch.cuda.set_device(dev)
    w = make_weights(dev)
    a_ids, n_ids = make_slots(args.tokens, args.hot_frac, args.mix, dev)
    S = args.tokens * TOPK
    g = torch.Generator().manual_seed(2)
    x13 = ((torch.randn(S, H, generator=g) / 8).to(torch.float16).to(dev))
    x2 = ((torch.randn(S, I, generator=g) / 8).to(torch.float16).to(dev))

    if args.bf16in:
        check_bf16in(ext, w, a_ids, n_ids, x13, x2, args.iters)
    if args.combine:
        check_combine(ext, args.tokens, args.iters, dev)
    if args.rowmap:
        check_rowmap(ext, w, args.hot_frac, args.iters, dev)
    if args.silu:
        check_silu(ext, args.tokens, args.iters, dev)
    if args.compose:
        check_compose(ext, w, args.tokens, args.hot_frac, args.iters, dev)
    if (args.bf16in or args.combine or args.rowmap or args.silu
            or args.compose) and not (args.check or args.prof):
        return

    runs = []
    if args.shape in ("w13", "both"):
        runs.append(("w13", lambda: ext.hybrid_moe_gemv(
            x13, w["w13_codes"], w["w13_cbs"], w["w13_scales"], a_ids,
            w["w13_packed"], w["w13_bscale"], w["w13_scale2"], n_ids)))
    if args.shape in ("w2", "both"):
        runs.append(("w2 ", lambda: ext.hybrid_moe_gemv(
            x2, w["w2_codes"], w["w2_cbs"], w["w2_scales"], a_ids,
            w["w2_packed"], w["w2_bscale"], w["w2_scale2"], n_ids)))

    if args.check:
        ref_src = os.environ.get(
            "KB_REF_SRC",
            "/shipped/csrc/quantization/aqlm_moe/aqlm_moe_v2.cu")
        if not pathlib.Path(ref_src).exists():
            ref_src = ("/home/jarrelscy/glm52/vllm/csrc/quantization/"
                       "aqlm_moe/aqlm_moe_v2.cu")
        ref = build(ref_src, "kb_ref_shipped", [])
        for nm, fn in runs:
            got = fn()  # env-off path: must stay bit-exact vs shipped
            x = x13 if nm.strip() == "w13" else x2
            k = "w13" if nm.strip() == "w13" else "w2"
            r = ref.hybrid_moe_gemv(
                x, w[f"{k}_codes"], w[f"{k}_cbs"], w[f"{k}_scales"], a_ids,
                w[f"{k}_packed"], w[f"{k}_bscale"], w[f"{k}_scale2"], n_ids)
            same = torch.equal(got.view(torch.int16), r.view(torch.int16))
            md = (got.float() - r.float()).abs().max().item()
            print(f"  CHECK {nm}: bit-exact={same} maxdiff={md}")
            if not same:
                sys.exit(f"NOT BIT-EXACT on {nm}")
            if args.pipeline:
                f32 = fp32_reference(x, w, a_ids, n_ids, k)
                # Two-part tolerance, anchored to the CURRENT kernel's own
                # deviation from the fp32 reference:
                #  - RMS error must not degrade (statistical quality gate;
                #    single-element max metrics are dominated by rounding
                #    jitter on the largest / most-cancelled outputs).
                #  - max-abs error bounded by old * 1.25 + 2 fp16 ULP of the
                #    largest output (guards against localized indexing /
                #    format bugs, which produce O(magnitude) errors).
                d_old = got.float() - f32
                set_pipeline(True)
                new = fn()
                set_pipeline(False)
                d_new = new.float() - f32
                rms_old = d_old.pow(2).mean().sqrt().item()
                rms_new = d_new.pow(2).mean().sqrt().item()
                a_old = d_old.abs().max().item()
                a_new = d_new.abs().max().item()
                ulp_max = 2.0 ** (torch.floor(torch.log2(
                    f32.abs().max().clamp(min=2 ** -14))).item() - 10)
                ok = (rms_new <= rms_old * 1.05 + 1e-6 and
                      a_new <= a_old * 1.25 + 2 * ulp_max)
                print(f"  CHECK {nm} V4 vs fp32 ref: "
                      f"rms {rms_old:.6f}->{rms_new:.6f} "
                      f"max {a_old:.4f}->{a_new:.4f} "
                      f"{'PASS' if ok else 'FAIL'}")
                if not ok:
                    sys.exit(f"V4 EXCEEDS TOLERANCE on {nm}")

    set_pipeline(args.pipeline)
    if args.prof:
        for nm, fn in runs:
            fn()
        torch.cuda.synchronize()
        return

    n_aqlm = int((a_ids >= 0).sum())
    print(f"tokens={args.tokens} slots={S} (aqlm={n_aqlm} nv={S - n_aqlm}) "
          f"mix={args.mix} path={'V4-pipeline' if args.pipeline else 'env'}")
    for nm, fn in runs:
        us = timed(fn, args.iters)
        # DRAM bytes: codes for aqlm slots + packed/bscale for nv slots
        if nm.strip() == "w13":
            ab = n_aqlm * (2 * I) * (H // 8) * 2
            nb = (S - n_aqlm) * (2 * I) * (H // 2 + H // 16)
        else:
            ab = n_aqlm * H * (I // 8) * 2
            nb = (S - n_aqlm) * H * (I // 2 + I // 16)
        gbps = (ab + nb) / 1e9 / (us / 1e6)
        print(f"  {nm} {us:9.2f} us   weights {(ab+nb)/1e6:7.2f} MB   "
              f"{gbps:7.1f} GB/s eff")


if __name__ == "__main__":
    main()
