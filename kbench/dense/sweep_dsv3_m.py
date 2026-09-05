"""dsv3_router_gemm vs fp32 SIMT path vs cublas bf16->fp32, M=1..16, GLM-5 shape."""
import torch
import vllm._custom_ops as ops

torch.manual_seed(0)
dev = "cuda:0"
K, N = 6144, 256
NC = 61  # weight copies to defeat L2 (6.3MB fp32 / 3.1MB bf16 each)

print(f"{'M':>3} {'dsv3_us':>8} {'fp32lin_us':>10} {'cublas_us':>10} {'dsv3_maxerr':>11} {'fp32_maxerr':>11}")
for M in [1, 2, 3, 4, 8, 16]:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    wsb = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) for _ in range(NC)]
    wsf = [w.float() for w in wsb]
    ref = x.double() @ wsb[0].double().t()
    e_dsv3 = e_fp32 = float("nan")
    def t(fn, ws):
        for w in ws:
            fn(w)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for i in range(NC * 40):
            fn(ws[i % NC])
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / (NC * 40)
    try:
        if M <= 16:
            o = ops.dsv3_router_gemm(hidden_states=x, router_weight=wsb[0], output_dtype=torch.float32)
            e_dsv3 = (o.double() - ref).abs().max().item()
            t_dsv3 = t(lambda w: ops.dsv3_router_gemm(hidden_states=x, router_weight=w, output_dtype=torch.float32), wsb)
        else:
            t_dsv3 = float("nan")
    except Exception as ex:
        t_dsv3 = float("nan")
        print("  dsv3 failed at M=", M, ex)
    xf = x.float()
    o2 = torch.nn.functional.linear(xf, wsf[0])
    e_fp32 = (o2.double() - ref).abs().max().item()
    t_fp32 = t(lambda w: torch.nn.functional.linear(xf, w), wsf)
    t_cb = t(lambda w: torch.mm(x, w.t(), out_dtype=torch.float32), wsb)
    print(f"{M:3d} {t_dsv3:8.2f} {t_fp32:10.2f} {t_cb:10.2f} {e_dsv3:11.3e} {e_fp32:11.3e}")
    del wsb, wsf
    torch.cuda.empty_cache()
