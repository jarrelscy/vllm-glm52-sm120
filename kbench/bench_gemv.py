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
    args = ap.parse_args()

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
            got = fn()
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

    if args.prof:
        for nm, fn in runs:
            fn()
        torch.cuda.synchronize()
        return

    n_aqlm = int((a_ids >= 0).sum())
    print(f"tokens={args.tokens} slots={S} (aqlm={n_aqlm} nv={S - n_aqlm}) "
          f"mix={args.mix}")
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
