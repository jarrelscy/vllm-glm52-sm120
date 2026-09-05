"""Microbench of GLM-5.3 hybrid decode dense GEMM shapes at TP4, M=4 (MTP verify).

Run inside glm52-vision-sm120 image on GPU 0:
docker run --rm --gpus '"device=0"' -v /home/jarrelscy/glm52:/glm52 --entrypoint bash \
  glm52-vision-sm120:latest -c '/opt/vllm/.venv/bin/python /glm52/dense-gemm-work/bench_dense.py'

Weights are rotated across enough copies to defeat the 128MB L2 so each timed
call streams weights from DRAM, matching production (a full forward pass evicts
everything between two uses of the same weight).
"""

import os
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda:0"

M = int(os.environ.get("BENCH_M", "4"))

# name, K, N, dtype, calls/step (approx, from trace mapping), prod avg us (if known)
SHAPES = [
    ("fused_qkv_a", 6144, 2624, torch.bfloat16),
    ("q_b_proj", 2048, 4096, torch.bfloat16),
    ("wk_weights_proj", 6144, 160, torch.bfloat16),
    ("o_proj", 4096, 6144, torch.bfloat16),
    ("router_fp32", 6144, 256, torch.float32),
    ("router_bf16w_fp32out", 6144, 256, torch.bfloat16),  # tier-3 candidate
    ("shared_w13", 6144, 1024, torch.bfloat16),
    ("shared_w2", 512, 6144, torch.bfloat16),
    ("dense_gate_up", 6144, 6144, torch.bfloat16),
    ("dense_down", 3072, 6144, torch.bfloat16),
    ("eh_proj", 12288, 6144, torch.bfloat16),
    ("lm_head", 6144, 38720, torch.bfloat16),
]

L2_BYTES = 128 * 1024 * 1024
TARGET_FOOTPRINT = 3 * L2_BYTES


def measure_peak_bw():
    n = 512 * 1024 * 1024  # 512M elements bf16 = 1GB
    a = torch.empty(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty(n, dtype=torch.bfloat16, device=dev)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    iters = 20
    for _ in range(iters):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 1000 / iters
    bw = 2 * n * 2 / t / 1e9
    del a, b
    torch.cuda.empty_cache()
    return bw


def bench_shape(name, K, N, wdtype, use_out_fp32=False, iters_per_copy=30):
    wbytes = K * N * wdtype.itemsize
    ncopies = max(2, min(48, -(-TARGET_FOOTPRINT // wbytes)))
    xdtype = wdtype if not use_out_fp32 else torch.bfloat16
    if name == "router_fp32":
        x = torch.randn(M, K, dtype=torch.float32, device=dev)
    else:
        x = torch.randn(M, K, dtype=xdtype, device=dev)
    ws = [torch.randn(N, K, dtype=wdtype, device=dev) for _ in range(ncopies)]

    def call(w):
        if use_out_fp32:
            return torch.mm(x, w.t(), out_dtype=torch.float32)
        return F.linear(x, w)

    # warmup
    for w in ws:
        call(w)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    total = ncopies * iters_per_copy
    for i in range(total):
        call(ws[i % ncopies])
    end.record()
    torch.cuda.synchronize()
    t_us = start.elapsed_time(end) * 1000 / total
    gbs = wbytes / (t_us * 1e-6) / 1e9
    # capture kernel name
    kname = "?"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(ncopies):
            call(ws[i % ncopies])
    evs = sorted(prof.key_averages(), key=lambda e: -(e.self_device_time_total or 0))
    knames = [e.key for e in evs if e.self_device_time_total and 'gemm' in e.key.lower() or 'gemv' in e.key.lower() or 'Kernel2' in e.key or 'cutlass' in e.key or 'splitK' in e.key]
    for w in ws:
        del w
    del ws, x
    torch.cuda.empty_cache()
    return t_us, gbs, wbytes, knames[:3]


def main():
    print(f"torch {torch.__version__}, M={M}")
    print(f"allow_bf16_reduced_precision_reduction={torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    peak = measure_peak_bw()
    print(f"D2D copy bandwidth: {peak:.0f} GB/s")
    print()
    print(f"{'name':22s} {'K':>6s} {'N':>6s} {'wMB':>7s} {'us':>8s} {'GB/s':>7s} {'roofline_us':>11s} {'headroom':>8s}")
    rows = []
    for name, K, N, dt in SHAPES:
        use_out_fp32 = name == "router_bf16w_fp32out"
        t_us, gbs, wbytes, knames = bench_shape(name, K, N, dt, use_out_fp32)
        roof = wbytes / peak / 1e3  # us at measured copy BW
        print(f"{name:22s} {K:6d} {N:6d} {wbytes/1e6:7.2f} {t_us:8.2f} {gbs:7.0f} {roof:11.2f} {t_us/roof:8.2f}x")
        rows.append((name, K, N, wbytes, t_us, gbs, roof))
        for k in knames:
            print(f"    kernel: {k[:110]}")

    # bmm absorb shapes (MLA): [16,M,192]x[16,192,512] and [16,M,512]x[16,512,256]
    for bname, (B, Mb, Kb, Nb) in [("bmm_q_absorb", (16, M, 192, 512)), ("bmm_o_absorb", (16, M, 512, 256))]:
        wbytes = B * Kb * Nb * 2
        ncopies = max(2, min(48, -(-TARGET_FOOTPRINT // wbytes)))
        x = torch.randn(B, Mb, Kb, dtype=torch.bfloat16, device=dev)
        ws = [torch.randn(B, Kb, Nb, dtype=torch.bfloat16, device=dev) for _ in range(ncopies)]
        for w in ws:
            torch.bmm(x, w)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        total = ncopies * 30
        for i in range(total):
            torch.bmm(x, ws[i % ncopies])
        e.record(); torch.cuda.synchronize()
        t_us = s.elapsed_time(e) * 1000 / total
        gbs = wbytes / (t_us * 1e-6) / 1e9
        roof = wbytes / peak / 1e3
        print(f"{bname:22s} {Kb:6d} {Nb:6d} {wbytes/1e6:7.2f} {t_us:8.2f} {gbs:7.0f} {roof:11.2f} {t_us/roof:8.2f}x")
        del ws, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
