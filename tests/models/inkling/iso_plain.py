"""Ground-truth probe: does the SM120 FA4 forward compute PLAIN causal attention
correctly (no rel_bias, no paging), vs a trivial torch softmax reference?
This removes any doubt about the rel_bias reference / paging harness.
"""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
L, H, D = 64, 4, 128
scale = 1.0 / D

q = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
k = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
v = torch.randn(L, H, D, device=dev, dtype=DTYPE)
cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, L], dtype=torch.int32, device=dev)

# torch reference: plain causal SDPA
qf, kf, vf = q.float(), k.float(), v.float()
scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)
scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum("hqk,khd->qhd", probs, vf)

def report(name, out):
    out = out.view(L, H, D).float()
    diff = (out - ref).abs()
    bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} "
          f"bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%) nan={torch.isnan(out).any().item()}")

try:
    out = flash_attn_varlen_func(
        q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=L, max_seqlen_k=L, softmax_scale=scale,
        causal=True, window_size=(None, None), num_splits=1, return_lse=False,
    )
    if isinstance(out, tuple): out = out[0]
    report("plain_contig", out)
except Exception as e:
    import traceback; traceback.print_exc()
    print("[plain_contig] EXC:", type(e).__name__, str(e)[:300])
