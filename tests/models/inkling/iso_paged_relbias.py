"""Isolate PAGED + rel_bias (mirrors the test's paged setup, single seq)."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func
from vllm.third_party.tml_fa4.testing import attention_ref

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
L, H, D = 64, 4, 128
BLK = 16; REL = 128
scale = 1.0 / D
q = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
k_flat = torch.nn.functional.normalize(torch.randn(L, H, D, device=dev).float(), dim=-1).to(DTYPE)
v_flat = torch.randn(L, H, D, device=dev, dtype=DTYPE)
cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)

nblk = (L + BLK - 1) // BLK
num_blocks = nblk + 1
key_cache = torch.zeros(num_blocks, BLK, H, D, device=dev, dtype=DTYPE)
value_cache = torch.zeros(num_blocks, BLK, H, D, device=dev, dtype=DTYPE)
for t in range(L):
    key_cache[1 + t // BLK, t % BLK] = k_flat[t]
    value_cache[1 + t // BLK, t % BLK] = v_flat[t]
block_table = torch.arange(1, 1 + nblk, dtype=torch.int32, device=dev)[None]
cache_seqlens = torch.tensor([L], dtype=torch.int32, device=dev)

qb, kb, vb = q[None], k_flat[None], v_flat[None]
ref_plain = attention_ref(qb, kb, vb, causal=True, softmax_scale=scale)
ref_plain = (ref_plain[0] if isinstance(ref_plain, tuple) else ref_plain)[0]

def paged_call(rel):
    r = flash_attn_varlen_func(q=q, k=key_cache, v=value_cache, cu_seqlens_q=cu_q,
        seqused_k=cache_seqlens, max_seqlen_q=L, page_table=block_table,
        softmax_scale=scale, causal=True, window_size=(None, None),
        num_splits=1, return_lse=False, rel_bias=rel)
    return (r[0] if isinstance(r, tuple) else r)

def rep(name, out, ref):
    out = out.view(L, H, D).float(); ref = ref.view(L, H, D).float()
    diff = (out - ref).abs(); bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%) nan={torch.isnan(out).any().item()}")

# A: paged + zero bias -> plain
rep("paged relbias=0 vs plain", paged_call(torch.zeros(L, H, REL, device=dev, dtype=DTYPE)), ref_plain)
# B: paged + diagonal bias
C = 3.0
relB = torch.zeros(L, H, REL, device=dev, dtype=DTYPE); relB[..., 0] = C
ab = torch.zeros(1, H, L, L, device=dev, dtype=torch.float32); idx = torch.arange(L, device=dev); ab[:, :, idx, idx] = C
ref_diag = attention_ref(qb, kb, vb, causal=True, softmax_scale=scale, attn_bias=ab)
ref_diag = (ref_diag[0] if isinstance(ref_diag, tuple) else ref_diag)[0]
rep("paged diag(C) vs ref", paged_call(relB), ref_diag)
