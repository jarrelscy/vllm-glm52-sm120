"""Isolation probe: run the SM120 FA4 rel-attn kernel with CONTIGUOUS kv (no
page_table) vs PAGED kv on identical data, both compared to the pure-torch
reference. Tells us whether the numeric bug is paging-specific or in the
SM120 forward/score_mod itself. Run inside the glm52-sm120 container.
"""
import torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func
from tests.models.inkling.test_fa4_rel_attention import _ref_rel_attn, BLOCK_SIZE, HEAD_DIM

torch.manual_seed(0)
dev = "cuda"
DTYPE = torch.bfloat16

L = 64            # single sequence, q_len = kv_len = 64 (causal full attn)
H = 4             # num_heads
Hkv = 4           # num_kv_heads (g=1)
REL = 128
scale = 1.0 / HEAD_DIM

q = torch.randn(L, H, HEAD_DIM, device=dev, dtype=DTYPE)
q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
# flat contiguous K/V for the sequence
k_flat = torch.randn(L, Hkv, HEAD_DIM, device=dev, dtype=DTYPE)
k_flat = torch.nn.functional.normalize(k_flat.float(), dim=-1).to(DTYPE)
v_flat = torch.randn(L, Hkv, HEAD_DIM, device=dev, dtype=DTYPE)
rel_logits = torch.randn(L, H, REL, device=dev, dtype=DTYPE)

cu_seqlens_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
cu_seqlens_k = torch.tensor([0, L], dtype=torch.int32, device=dev)

# ---- Build a paged view of the SAME data ----
nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
num_blocks = nblk + 1  # block 0 = pad
key_cache = torch.zeros(num_blocks, BLOCK_SIZE, Hkv, HEAD_DIM, device=dev, dtype=DTYPE)
value_cache = torch.zeros(num_blocks, BLOCK_SIZE, Hkv, HEAD_DIM, device=dev, dtype=DTYPE)
for t in range(L):
    b = 1 + t // BLOCK_SIZE
    o = t % BLOCK_SIZE
    key_cache[b, o] = k_flat[t]
    value_cache[b, o] = v_flat[t]
block_table = torch.zeros(1, nblk, dtype=torch.int32, device=dev)
block_table[0] = torch.arange(1, 1 + nblk, dtype=torch.int32, device=dev)
cache_seqlens = torch.tensor([L], dtype=torch.int32, device=dev)

# ---- Reference ----
ref = _ref_rel_attn(
    q, key_cache, value_cache, rel_logits,
    q_lens=[L], kv_lens=[L], block_table=block_table,
    scale=scale, rel_extent=REL, window_left=None,
).float()

def report(name, out):
    out = out.view(L, H, HEAD_DIM).float()
    diff = (out - ref).abs()
    bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} "
          f"bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%)")

# ---- CONTIGUOUS (no page_table) ----
try:
    out_c = flash_attn_varlen_func(
        q=q, k=k_flat, v=v_flat,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=L, max_seqlen_k=L,
        softmax_scale=scale, causal=True, window_size=(None, None),
        num_splits=1, return_lse=False, rel_bias=rel_logits.contiguous(),
    )
    if isinstance(out_c, tuple): out_c = out_c[0]
    report("contiguous", out_c)
except Exception as e:
    print("[contiguous] EXC:", type(e).__name__, str(e)[:200])

# ---- PAGED ----
try:
    out_p = flash_attn_varlen_func(
        q=q, k=key_cache, v=value_cache,
        cu_seqlens_q=cu_seqlens_q, seqused_k=cache_seqlens,
        max_seqlen_q=L, page_table=block_table,
        softmax_scale=scale, causal=True, window_size=(None, None),
        num_splits=1, return_lse=False, rel_bias=rel_logits.contiguous(),
    )
    if isinstance(out_p, tuple): out_p = out_p[0]
    report("paged", out_p)
except Exception as e:
    print("[paged] EXC:", type(e).__name__, str(e)[:200])
