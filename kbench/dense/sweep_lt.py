"""cuBLASLt heuristic algo sweep + Triton small-M GEMM for underperforming
GLM-5.3 decode dense GEMM shapes. GPU0 only.

Every timed configuration rotates weights across >=384MB of copies to defeat L2.
"""
import ctypes as C
import os
import glob
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.manual_seed(0)
dev = "cuda:0"
M = int(os.environ.get("BENCH_M", "4"))
TARGET = 3 * 128 * 1024 * 1024

# ---------------- cublasLt via ctypes ----------------
lib = None
for cand in (
    glob.glob("/opt/vllm/.venv/lib/python*/site-packages/nvidia/cublas/lib/libcublasLt.so*")
    + glob.glob("/opt/vllm/.venv/lib/python*/site-packages/torch/lib/libcublasLt*.so*")
    + ["libcublasLt.so.13", "libcublasLt.so.12"]
):
    try:
        lib = C.CDLL(cand)
        LIBNAME = cand
        break
    except OSError:
        continue
assert lib is not None

CUDA_R_16BF = 14
CUDA_R_32F = 0
COMPUTE_32F = 68
OP_N, OP_T = 0, 1
DESC_TRANSA, DESC_TRANSB = 3, 4
PREF_MAX_WORKSPACE = 1

class HeurResult(C.Structure):
    _fields_ = [
        ("algo", C.c_uint64 * 8),
        ("workspaceSize", C.c_size_t),
        ("state", C.c_int),
        ("wavesCount", C.c_float),
        ("reserved", C.c_int * 4),
    ]

handle = C.c_void_p()
assert lib.cublasLtCreate(C.byref(handle)) == 0

def make_layout(dtype, rows, cols, ld):
    lo = C.c_void_p()
    st = lib.cublasLtMatrixLayoutCreate(C.byref(lo), C.c_int(dtype), C.c_uint64(rows), C.c_uint64(cols), C.c_int64(ld))
    assert st == 0, st
    return lo

def make_desc():
    d = C.c_void_p()
    st = lib.cublasLtMatmulDescCreate(C.byref(d), C.c_int(COMPUTE_32F), C.c_int(CUDA_R_32F))
    assert st == 0, st
    ta, tb = C.c_int(OP_T), C.c_int(OP_N)
    assert lib.cublasLtMatmulDescSetAttribute(d, C.c_int(DESC_TRANSA), C.byref(ta), C.c_size_t(4)) == 0
    assert lib.cublasLtMatmulDescSetAttribute(d, C.c_int(DESC_TRANSB), C.byref(tb), C.c_size_t(4)) == 0
    return d

WS_BYTES = 64 * 1024 * 1024
workspace = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)

def lt_algos(Mm, K, N, out_dtype=CUDA_R_16BF, n_req=48):
    """heuristic algos for out[M,N] = x[M,K] @ w[N,K]^T (row-major views)."""
    desc = make_desc()
    Adesc = make_layout(CUDA_R_16BF, K, N, K)   # W buffer, transA=T -> N x K
    Bdesc = make_layout(CUDA_R_16BF, K, Mm, K)  # x buffer -> K x M
    Cdesc = make_layout(out_dtype, N, Mm, N)
    pref = C.c_void_p()
    assert lib.cublasLtMatmulPreferenceCreate(C.byref(pref)) == 0
    ws = C.c_size_t(WS_BYTES)
    assert lib.cublasLtMatmulPreferenceSetAttribute(pref, C.c_int(PREF_MAX_WORKSPACE), C.byref(ws), C.c_size_t(8)) == 0
    results = (HeurResult * n_req)()
    nret = C.c_int(0)
    st = lib.cublasLtMatmulAlgoGetHeuristic(handle, desc, Adesc, Bdesc, Cdesc, Cdesc, pref, C.c_int(n_req), results, C.byref(nret))
    assert st == 0, st
    return desc, Adesc, Bdesc, Cdesc, [results[i] for i in range(nret.value) if results[i].state == 0]

alpha = C.c_float(1.0)
beta = C.c_float(0.0)

def lt_matmul(desc, Adesc, Bdesc, Cdesc, w, x, out, algo):
    stream = torch.cuda.current_stream().cuda_stream
    st = lib.cublasLtMatmul(
        handle, desc,
        C.byref(alpha), C.c_void_p(w.data_ptr()), Adesc,
        C.c_void_p(x.data_ptr()), Bdesc,
        C.byref(beta), C.c_void_p(out.data_ptr()), Cdesc,
        C.c_void_p(out.data_ptr()), Cdesc,
        C.byref(algo.algo), C.c_void_p(workspace.data_ptr()), C.c_size_t(algo.workspaceSize if algo.workspaceSize <= WS_BYTES else 0),
        C.c_void_p(stream))
    return st

# ---------------- Triton small-M kernel ----------------
@triton.jit
def _gemv_smallm(x_ptr, w_ptr, out_ptr, M_: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                 locks_ptr, partial_ptr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((M_, BLOCK_N), dtype=tl.float32)
    K_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per
    k_end = tl.minimum(k_start + K_per, K)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < k_end
        xm = tl.load(x_ptr + tl.arange(0, M_)[:, None] * K + offs_k[None, :],
                     mask=kmask[None, :], other=0.0)
        wm = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                     mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)
        acc += tl.dot(xm, tl.trans(wm), out_dtype=tl.float32)
    if SPLIT_K == 1:
        tl.store(out_ptr + tl.arange(0, M_)[:, None] * N + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=offs_n[None, :] < N)
    else:
        tl.store(partial_ptr + pid_k * M_ * N + tl.arange(0, M_)[:, None] * N + offs_n[None, :],
                 acc, mask=offs_n[None, :] < N)

@triton.jit
def _reduce_k(partial_ptr, out_ptr, MN: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(partial_ptr + s * MN + offs, mask=offs < MN, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=offs < MN)


def triton_linear(x, w, out, partial, cfg):
    BLOCK_N, BLOCK_K, SPLIT_K, warps = cfg
    N, K = w.shape
    grid = (triton.cdiv(N, BLOCK_N), SPLIT_K)
    _gemv_smallm[grid](x, w, out, x.shape[0] if x.shape[0] >= 16 else 16, K, N,
                       BLOCK_N, BLOCK_K, SPLIT_K, None, partial, num_warps=warps)


# padded-M variant: tl.dot needs M>=16; pad x to 16 rows
def run_triton(x, w, out, partial, cfg):
    BLOCK_N, BLOCK_K, SPLIT_K, warps = cfg
    N, K = w.shape
    Mp = 16
    grid = (triton.cdiv(N, BLOCK_N), SPLIT_K)
    _gemv_smallm[grid](x, w, out, Mp, K, N, BLOCK_N, BLOCK_K, SPLIT_K, None, partial, num_warps=warps)
    if SPLIT_K > 1:
        MN = Mp * N
        _reduce_k[(triton.cdiv(MN, 4096),)](partial, out, MN, SPLIT_K, 4096)


def time_call(fn, ws, iters_per_copy=30):
    n = len(ws)
    for w in ws:
        fn(w)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for i in range(n * iters_per_copy):
        fn(ws[i % n])
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (n * iters_per_copy)


SHAPES = [
    ("q_b/wq_b", 2048, 4096),
    ("fused_qkv_a", 6144, 2624),
    ("shared_w13", 6144, 1024),
    ("shared_w2", 512, 6144),
    ("wk_weights_proj", 6144, 160),
    ("dense_down", 3072, 6144),
    ("o_proj", 4096, 6144),
]

print(f"lib={LIBNAME}  M={M}")
for name, K, N in SHAPES:
    wbytes = K * N * 2
    ncopies = max(2, min(64, -(-TARGET // wbytes)))
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ws = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) for _ in range(ncopies)]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    ref = F.linear(x.float(), ws[0].float())

    t_def = time_call(lambda w: F.linear(x, w), ws)
    DO_LT = os.environ.get('DO_LT','1')=='1'
    DO_TR = os.environ.get('DO_TR','1')=='1'

    # Lt sweep
    t_lt, best_ai, algos = 1e9, -1, []
    if DO_LT:
        pass
    if DO_LT:
        desc, Adesc, Bdesc, Cdesc, algos = lt_algos(M, K, N)
    best = (1e9, -1)
    for ai, algo in enumerate(algos):
        st = lt_matmul(desc, Adesc, Bdesc, Cdesc, ws[0], x, out, algo)
        torch.cuda.synchronize()
        if st != 0:
            continue
        err = (out.float() - ref).abs().max().item() / max(1e-9, ref.abs().max().item())
        if err > 2e-2:
            print(f"  algo {ai}: WRONG rel_err {err}")
            continue
        t = time_call(lambda w, a=algo: lt_matmul(desc, Adesc, Bdesc, Cdesc, w, x, out, a), ws, 15)
        if t < best[0]:
            best = (t, ai)
    if DO_LT:
        t_lt, best_ai = best

    # Triton sweep
    Mp = 16
    xp = torch.zeros(Mp, K, dtype=torch.bfloat16, device=dev)
    xp[:M] = x
    outp = torch.empty(Mp, N, dtype=torch.bfloat16, device=dev)
    best_tr = (1e9, None)
    for BLOCK_N in ((16, 32, 64, 128) if DO_TR else ()):
        for BLOCK_K in (64, 128, 256, 512):
            for SPLIT_K in (1, 2, 4, 8, 16):
                if N // BLOCK_N * SPLIT_K < 96 and (N // max(1,BLOCK_N)) * SPLIT_K < 1536:
                    pass
                cfg = (BLOCK_N, BLOCK_K, SPLIT_K, 4)
                partial = torch.empty(SPLIT_K * Mp * N, dtype=torch.float32, device=dev) if SPLIT_K > 1 else outp
                try:
                    run_triton(xp, ws[0], outp, partial, cfg)
                    torch.cuda.synchronize()
                except Exception:
                    continue
                err = (outp[:M].float() - ref).abs().max().item() / max(1e-9, ref.abs().max().item())
                if err > 2e-2:
                    continue
                t = time_call(lambda w, c=cfg, p=partial: run_triton(xp, w, outp, p, c), ws, 10)
                if t < best_tr[0]:
                    best_tr = (t, cfg)
    t_tr, cfg_tr = best_tr
    roof = wbytes / 1468 / 1e3
    print(f"{name:18s} K={K:5d} N={N:5d} w={wbytes/1e6:6.2f}MB roof={roof:6.2f}us | torch {t_def:6.2f} | Lt best {t_lt:6.2f} (algo {best_ai}/{len(algos)}) | triton {t_tr:6.2f} cfg={cfg_tr}")
    del ws, x
    torch.cuda.empty_cache()
