"""Isolate the VARLEN (rank-3 rel_logits) rel_bias mapping, contiguous (no paging).
Compares zero-bias (=plain) and diagonal-bias against references."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func
from vllm.third_party.tml_fa4.testing import attention_ref

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
L, H, D = 64, 4, 128
REL = 128
scale = 1.0 / D
q = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
k = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
v = torch.randn(L, H, D, device=dev, dtype=DTYPE)
cu = torch.tensor([0, L], dtype=torch.int32, device=dev)

def rep(name, out, ref):
    out = out.view(L, H, D).float(); ref = ref.view(L, H, D).float()
    diff = (out - ref).abs()
    bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%)")

# batched reference for plain and diagonal-bias (b=1)
qb, kb, vb = q[None], k[None], v[None]
ref_plain = attention_ref(qb, kb, vb, causal=True, softmax_scale=scale)
ref_plain = (ref_plain[0] if isinstance(ref_plain, tuple) else ref_plain)[0]

# Case A: varlen, rel_bias = 0 -> plain
relz = torch.zeros(L, H, REL, device=dev, dtype=DTYPE)
rA = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=L, max_seqlen_k=L, softmax_scale=scale, causal=True,
    window_size=(None, None), num_splits=1, return_lse=False, rel_bias=relz)
outA = rA[0] if isinstance(rA, tuple) else rA
rep("varlen relbias=0 vs plain", outA, ref_plain)

# Case B: diagonal bias rel_logits[i,h,0]=C -> bias on i-j==0
C = 3.0
relB = torch.zeros(L, H, REL, device=dev, dtype=DTYPE); relB[..., 0] = C
ab = torch.zeros(1, H, L, L, device=dev, dtype=torch.float32)
idx = torch.arange(L, device=dev); ab[:, :, idx, idx] = C
ref_diag = attention_ref(qb, kb, vb, causal=True, softmax_scale=scale, attn_bias=ab)
ref_diag = (ref_diag[0] if isinstance(ref_diag, tuple) else ref_diag)[0]
rB = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=L, max_seqlen_k=L, softmax_scale=scale, causal=True,
    window_size=(None, None), num_splits=1, return_lse=False, rel_bias=relB)
outB = rB[0] if isinstance(rB, tuple) else rB
rep("varlen diag(C) vs ref", outB, ref_diag)
