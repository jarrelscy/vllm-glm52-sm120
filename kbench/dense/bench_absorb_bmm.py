"""Prototype Triton batched small-M bmm for MLA absorb ops, exact prod strides.

q absorb: x [N=16, B=M, P=192] strides (256, 16*256, 1)  @ W_UK_T [16,192,512] -> out [16,M,512] contig
v up:     x [N=16, B=M, L=512] strides (512, 16*512, 1)  @ W_UV   [16,512,256] -> out.T view [16,M,256] strides (256,16*256,1)
"""
import os
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = "cuda:0"
M = int(os.environ.get("BENCH_M", "4"))
NH = 16
TARGET = 3 * 128 * 1024 * 1024


@triton.jit
def _absorb_bmm(x_ptr, w_ptr, o_ptr,
                sxb, sxm, sob, som,
                B: tl.constexpr, Mtok: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr):
    pid = tl.program_id(0)
    nblocks = tl.cdiv(N, BLOCK_N)
    b = pid // nblocks
    pn = pid % nblocks
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, MP)
    acc = tl.zeros((MP, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xm = tl.load(x_ptr + b * sxb + offs_m[:, None] * sxm + offs_k[None, :],
                     mask=(offs_m[:, None] < Mtok) & (offs_k[None, :] < K), other=0.0)
        wm = tl.load(w_ptr + b * K * N + offs_k[:, None] * N + offs_n[None, :],
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(xm, wm, out_dtype=tl.float32)
    tl.store(o_ptr + b * sob + offs_m[:, None] * som + offs_n[None, :],
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < Mtok) & (offs_n[None, :] < N))


def run(xs, w, o, cfg):
    BLOCK_N, BLOCK_K, warps = cfg
    B, Mtok, K = xs.shape
    N = w.shape[2]
    grid = (B * triton.cdiv(N, BLOCK_N),)
    _absorb_bmm[grid](xs, w, o,
                      xs.stride(0), xs.stride(1), o.stride(0), o.stride(1),
                      B, Mtok, K, N, BLOCK_N, BLOCK_K, 16, num_warps=warps)


def time_call(fn, ws, iters=25):
    n = len(ws)
    for w in ws:
        fn(w)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for i in range(n * iters):
        fn(ws[i % n])
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (n * iters)


for name, P_in, N_out in [("q_absorb", 192, 512), ("v_up", 512, 256)]:
    K = P_in
    N = N_out
    wbytes = NH * K * N * 2
    ncopies = max(2, min(96, -(-TARGET // wbytes)))
    if name == "q_absorb":
        base = torch.randn(M, NH, 256, dtype=torch.bfloat16, device=dev)
        x = base[..., :192].transpose(0, 1)  # [16, M, 192] strided
        out = torch.empty(NH, M, N, dtype=torch.bfloat16, device=dev)
        oV = out
    else:
        base = torch.randn(M, NH, 512, dtype=torch.bfloat16, device=dev)
        x = base.transpose(0, 1)  # [16, M, 512] strided
        obase = torch.empty(M, NH, N, dtype=torch.bfloat16, device=dev)
        oV = obase.transpose(0, 1)  # strided out like prod
    ws = [torch.randn(NH, K, N, dtype=torch.bfloat16, device=dev) for _ in range(ncopies)]

    t_torch = time_call(lambda w: torch.bmm(x, w, out=oV), ws)
    ref = torch.bmm(x.float(), ws[0].float())

    best = (1e9, None)
    for BLOCK_N in (32, 64, 128, 256):
        for BLOCK_K in (32, 64, 128, 256):
            if BLOCK_K > K or BLOCK_N > N:
                continue
            for warps in (2, 4, 8):
                cfg = (BLOCK_N, BLOCK_K, warps)
                try:
                    run(x, ws[0], oV, cfg)
                    torch.cuda.synchronize()
                except Exception:
                    continue
                err = (oV.float() - ref).abs().max().item()
                rel = err / ref.abs().max().item()
                if rel > 2e-2:
                    print(f"  cfg {cfg}: WRONG rel={rel}")
                    continue
                t = time_call(lambda w, c=cfg: run(x, w, oV, c), ws, 15)
                if t < best[0]:
                    best = (t, cfg)
    roof = wbytes / 1468 / 1e3
    print(f"{name}: w={wbytes/1e6:.2f}MB roof={roof:.2f}us | torch.bmm {t_torch:.2f}us | triton {best[0]:.2f}us cfg={best[1]}")
    # ULP check for best cfg
    run(x, ws[0], oV, best[1])
    tref = torch.bmm(x, ws[0], out=torch.empty_like(oV.contiguous()).as_strided(oV.shape, oV.stride()) if not oV.is_contiguous() else torch.empty_like(oV))
    diff = (oV.float() - tref.float()).abs()
    denom = tref.float().abs().clamp_min(1e-6)
    print(f"   vs cublas bf16: max abs {diff.max().item():.3e}, max rel {(diff/denom).max().item():.3e}")
