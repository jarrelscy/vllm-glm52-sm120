"""FMA-style Triton kernels (no tl.dot, no M-padding) for tiny-M dense ops:
- MLA absorb bmms: [16,M,192]x[16,192,512], [16,M,512]x[16,512,256] (strided x/out)
- indexer wk_weights_proj gemv: [M,6144]x[6144,160] with split-K

Numerics: exact bf16 products accumulated in fp32 (same math as cuBLAS bf16
GEMM with fp32 accumulate; reduction order differs).
"""
import os
import functools, sys
print = functools.partial(print, flush=True)
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = "cuda:0"
M = int(os.environ.get("BENCH_M", "4"))
NH = 16
TARGET = 3 * 128 * 1024 * 1024


@triton.jit
def _fma_bmm(x_ptr, w_ptr, o_ptr,
             sxb, sxm, sob, som,
             Mtok: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[b,m,n] = sum_k x[b,m,k] w[b,k,n]; one program per (b, n-block).
    M is tiny: accumulate an (Mtok, BLOCK_N) tile via broadcast FMA."""
    pid = tl.program_id(0)
    nblocks = N // BLOCK_N
    b = pid // nblocks
    pn = pid % nblocks
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, Mtok)
    acc = tl.zeros((Mtok, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xm = tl.load(x_ptr + b * sxb + offs_m[:, None] * sxm + offs_k[None, :]).to(tl.float32)
        wm = tl.load(w_ptr + b * K * N + offs_k[:, None] * N + offs_n[None, :]).to(tl.float32)
        acc += tl.sum(xm[:, None, :] * tl.trans(wm)[None, :, :], axis=2)
    tl.store(o_ptr + b * sob + offs_m[:, None] * som + offs_n[None, :], acc.to(tl.bfloat16))


@triton.jit
def _fma_gemv_splitk(x_ptr, w_ptr, p_ptr,
                     Mtok: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """partial[s,m,n] = sum over k-slice s of x[m,k] w[n,k] (w row-major [N,K])."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, Mtok)
    K_per: tl.constexpr = K // SPLIT_K
    k_start = pid_k * K_per
    acc = tl.zeros((Mtok, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + K_per, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xm = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :]).to(tl.float32)
        wm = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                     mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc += tl.sum(xm[:, None, :] * wm[None, :, :], axis=2)
    tl.store(p_ptr + pid_k * Mtok * N + offs_m[:, None] * N + offs_n[None, :],
             acc, mask=offs_n[None, :] < N)


@triton.jit
def _reduce_bf16(p_ptr, o_ptr, MN: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + s * MN + offs, mask=offs < MN, other=0.0)
    tl.store(o_ptr + offs, acc.to(tl.bfloat16), mask=offs < MN)


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


print(f"M={M}")
# ---- absorb bmms ----
for name, K, N in [("q_absorb", 192, 512), ("v_up", 512, 256)]:
    wbytes = NH * K * N * 2
    ncopies = max(2, min(96, -(-TARGET // wbytes)))
    if name == "q_absorb":
        base = torch.randn(M, NH, 256, dtype=torch.bfloat16, device=dev)
        x = base[..., :192].transpose(0, 1)
        oV = torch.empty(NH, M, N, dtype=torch.bfloat16, device=dev)
    else:
        base = torch.randn(M, NH, 512, dtype=torch.bfloat16, device=dev)
        x = base.transpose(0, 1)
        obase = torch.empty(M, NH, N, dtype=torch.bfloat16, device=dev)
        oV = obase.transpose(0, 1)
    ws = [torch.randn(NH, K, N, dtype=torch.bfloat16, device=dev) for _ in range(ncopies)]
    t_torch = time_call(lambda w: torch.bmm(x, w, out=oV), ws)
    ref = torch.bmm(x.float(), ws[0].float())
    best = (1e9, None)
    for BLOCK_N in (16, 32, 64):
        for BLOCK_K in (16, 32):
            if BLOCK_N > N or BLOCK_K > K or BLOCK_N * BLOCK_K > 1024:
                continue
            for warps in (2, 4):
                cfg = (BLOCK_N, BLOCK_K, warps)
                grid = (NH * (N // BLOCK_N),)
                def call(w, c=cfg, g=grid):
                    _fma_bmm[g](x, w, oV, x.stride(0), x.stride(1), oV.stride(0), oV.stride(1),
                                M, K, N, c[0], c[1], num_warps=c[2])
                try:
                    call(ws[0]); torch.cuda.synchronize()
                except Exception:
                    continue
                rel = ((oV.float() - ref).abs().max() / ref.abs().max()).item()
                if rel > 2e-2:
                    print(f"  {name} cfg {cfg}: WRONG rel={rel}")
                    continue
                t = time_call(call, ws, 15)
                if t < best[0]:
                    best = (t, cfg)
    roof = wbytes / 1468 / 1e3
    print(f"{name}: w={wbytes/1e6:.2f}MB roof={roof:.2f}us | torch.bmm {t_torch:.2f} | fma-triton {best[0]:.2f} cfg={best[1]}")
    del ws
    torch.cuda.empty_cache()

# ---- wk_weights gemv ----
K, N = 6144, 160
wbytes = K * N * 2
ncopies = 96
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
ws = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) for _ in range(ncopies)]
out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
t_torch = time_call(lambda w: torch.nn.functional.linear(x, w, out=None), ws)
ref = torch.nn.functional.linear(x.float(), ws[0].float())
best = (1e9, None)
for BLOCK_N in (16, 32):
    for BLOCK_K in (32, 64):
        for SPLIT_K in (8, 16, 32, 48):
            if K % SPLIT_K or (K // SPLIT_K) % BLOCK_K:
                continue
            cfg = (BLOCK_N, BLOCK_K, SPLIT_K)
            partial = torch.empty(SPLIT_K * M * N, dtype=torch.float32, device=dev)
            grid = (triton.cdiv(N, BLOCK_N), SPLIT_K)
            MN = M * N
            def call(w, c=cfg, p=partial, g=grid):
                _fma_gemv_splitk[g](x, w, p, M, K, N, c[0], c[1], c[2], num_warps=2)
                _reduce_bf16[(triton.cdiv(MN, 256),)](p, out, MN, c[2], 256)
            try:
                call(ws[0]); torch.cuda.synchronize()
            except Exception:
                continue
            rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            if rel > 2e-2:
                print(f"  wk cfg {cfg}: WRONG rel={rel}")
                continue
            t = time_call(call, ws, 15)
            if t < best[0]:
                best = (t, cfg)
print(f"wk_weights_proj: w={wbytes/1e6:.2f}MB roof={wbytes/1468/1e3:.2f}us | torch {t_torch:.2f} | fma-splitk {best[0]:.2f} cfg={best[1]}")
