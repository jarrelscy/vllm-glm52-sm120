"""Reveal the paged path's rel_bias offset->(i-j) mapping. One-hot rel_logits at
offset d0 + tiny scale => output[i] ~= V[j] where the kernel places bias for d0.
Spec: bias at i-j==d0 => dominant j = i-d0 => out[i,0] ~= i-d0."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
L, H, D = 64, 1, 128
BLK = 16; REL = 128
scale = 1e-4
q = torch.randn(L, H, D, device=dev, dtype=DTYPE)
k_flat = torch.randn(L, H, D, device=dev, dtype=DTYPE)
v_flat = torch.zeros(L, H, D, device=dev, dtype=DTYPE)
for j in range(L):
    v_flat[j, 0, 0] = float(j)
cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
nblk = (L + BLK - 1) // BLK
key_cache = torch.zeros(nblk + 1, BLK, H, D, device=dev, dtype=DTYPE)
value_cache = torch.zeros(nblk + 1, BLK, H, D, device=dev, dtype=DTYPE)
for t in range(L):
    key_cache[1 + t // BLK, t % BLK] = k_flat[t]
    value_cache[1 + t // BLK, t % BLK] = v_flat[t]
block_table = torch.arange(1, 1 + nblk, dtype=torch.int32, device=dev)[None]
cache_seqlens = torch.tensor([L], dtype=torch.int32, device=dev)

for d0 in (0, 5, 20):
    rel = torch.zeros(L, H, REL, device=dev, dtype=DTYPE); rel[..., d0] = 50.0
    r = flash_attn_varlen_func(q=q, k=key_cache, v=value_cache, cu_seqlens_q=cu_q,
        seqused_k=cache_seqlens, max_seqlen_q=L, page_table=block_table,
        softmax_scale=scale, causal=True, window_size=(None, None),
        num_splits=1, return_lse=False, rel_bias=rel)
    out = (r[0] if isinstance(r, tuple) else r).float().view(L, H, D)
    got = out[:, 0, 0]
    isr = torch.arange(L, device=dev).float()
    implied = isr - got
    samp = [d0+2, d0+10, 40, 63]
    vals = [(int(i), round(got[i].item(),1), round(implied[i].item(),1)) for i in samp if i < L]
    print(f"d0={d0}: (i, dom_j, i-j)={vals}  spec i-j=={d0}")
