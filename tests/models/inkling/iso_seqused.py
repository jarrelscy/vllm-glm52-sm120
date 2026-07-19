"""Discriminator: contiguous K but with seqused_k (not cu_seqlens_k), diag bias.
If this fails like the paged case, the bug is seqused_k handling in the score_mod
(kv_idx / seqlen), not paging per se."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func
from vllm.third_party.tml_fa4.testing import attention_ref

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
L, H, D = 64, 4, 128
REL = 128; scale = 1.0 / D
q = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
k = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
v = torch.randn(L, H, D, device=dev, dtype=DTYPE)
cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
seqused_k = torch.tensor([L], dtype=torch.int32, device=dev)
qb, kb, vb = q[None], k[None], v[None]

def rep(name, out, ref):
    out = out.view(L, H, D).float(); ref = ref.view(L, H, D).float()
    diff = (out - ref).abs(); bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%)")

C = 3.0
relB = torch.zeros(L, H, REL, device=dev, dtype=DTYPE); relB[..., 0] = C
ab = torch.zeros(1, H, L, L, device=dev, dtype=torch.float32); idx = torch.arange(L, device=dev); ab[:, :, idx, idx] = C
ref_diag = attention_ref(qb, kb, vb, causal=True, softmax_scale=scale, attn_bias=ab)
ref_diag = (ref_diag[0] if isinstance(ref_diag, tuple) else ref_diag)[0]
# contiguous k but drive seqlen via seqused_k (no cu_seqlens_k, no page_table)
r = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, seqused_k=seqused_k,
    max_seqlen_q=L, max_seqlen_k=L, softmax_scale=scale, causal=True,
    window_size=(None, None), num_splits=1, return_lse=False, rel_bias=relB)
out = r[0] if isinstance(r, tuple) else r
rep("contig+seqused_k diag(C) vs ref", out, ref_diag)
