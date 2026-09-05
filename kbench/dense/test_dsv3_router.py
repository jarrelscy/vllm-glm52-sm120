import torch, time
torch.manual_seed(0)
dev='cuda:0'
import vllm._custom_ops as ops
M,K,N=4,6144,256
x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
w=torch.randn(N,K,dtype=torch.bfloat16,device=dev)
# reference: fp64 exact
ref64 = (x.double() @ w.double().t())
# current prod path: fp32 weights, fp32 x, F.linear fp32
cur = torch.nn.functional.linear(x.float(), w.float())
try:
    out = ops.dsv3_router_gemm(hidden_states=x, router_weight=w, output_dtype=torch.float32)
    print("dsv3_router_gemm OK, dtype", out.dtype, out.shape)
    d_new = (out.double()-ref64).abs().max().item()
    d_cur = (cur.double()-ref64).abs().max().item()
    rel = (out - cur).abs().max().item()
    print(f"max|dsv3-fp64ref|={d_new:.3e}  max|fp32path-fp64ref|={d_cur:.3e}  max|dsv3-fp32path|={rel:.3e} scale={ref64.abs().max().item():.1f}")
    # top-8 expert agreement over many random draws
    mismatch=0
    for i in range(200):
        x2=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
        o1=ops.dsv3_router_gemm(hidden_states=x2, router_weight=w, output_dtype=torch.float32)
        o2=torch.nn.functional.linear(x2.float(), w.float())
        t1=o1.topk(8,dim=-1).indices.sort(-1).values
        t2=o2.topk(8,dim=-1).indices.sort(-1).values
        mismatch += (t1!=t2).any().item()
    print("top8 mismatches over 200 draws:", mismatch)
    # timing with L2 rotation
    ws=[torch.randn(N,K,dtype=torch.bfloat16,device=dev) for _ in range(61)]
    for wq in ws: ops.dsv3_router_gemm(hidden_states=x, router_weight=wq, output_dtype=torch.float32)
    torch.cuda.synchronize()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True); s.record()
    for i in range(61*40):
        ops.dsv3_router_gemm(hidden_states=x, router_weight=ws[i%61], output_dtype=torch.float32)
    e.record(); torch.cuda.synchronize()
    t=s.elapsed_time(e)*1000/(61*40)
    print(f"dsv3_router_gemm: {t:.2f} us  ({K*N*2/t/1e3:.0f} GB/s)")
except Exception as ex:
    print("dsv3_router_gemm FAILED:", ex)
# tier-3 fallback: cublas bf16 x bf16 -> fp32
o3 = torch.mm(x, w.t(), out_dtype=torch.float32)
print("tier3 max|.-fp64ref|:", (o3.double()-ref64).abs().max().item())
