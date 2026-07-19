"""Disambiguate the rel_bias bug: zero-bias must equal plain attention."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_func
from vllm.third_party.tml_fa4.testing import attention_ref

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
b, s, h, d = 1, 128, 4, 128    # s=128 so seqlen >= rel_extent(128)
REL = 128
scale = 1.0 / d
q = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
k = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
v = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)

def rep(name, out, ref):
    out = out.float(); ref = ref.float()
    diff = (out - ref).abs()
    bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} mean={diff.mean().item():.5f} "
          f"bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%) nan={torch.isnan(out).any().item()}")

# plain reference (no bias)
ref_plain = attention_ref(q, k, v, causal=True, softmax_scale=scale)
if isinstance(ref_plain, tuple): ref_plain = ref_plain[0]

# Case A: rel_bias = zeros -> must equal plain attention
relz = torch.zeros(b, s, h, REL, device=dev, dtype=DTYPE)
rA = flash_attn_func(q, k, v, rel_bias=relz, softmax_scale=scale, causal=True, num_splits=1)
outA = rA[0] if isinstance(rA, tuple) else rA
rep("relbias=0 vs plain", outA, ref_plain)

# Case B: rel_bias only at offset 0 (diagonal): rel_logits[i,h,0]=C, else 0.
# Then bias(i,j,h) = C when i-j==0 (j==i), else 0. Reference via attention_ref
# needs the matching attn_bias. Build attn_bias[b,h,i,j] explicitly.
C = 3.0
relB = torch.zeros(b, s, h, REL, device=dev, dtype=DTYPE)
relB[..., 0] = C
# attn_bias for attention_ref is [b, h, s_q, s_k] added to scores
ab = torch.zeros(b, h, s, s, device=dev, dtype=torch.float32)
idx = torch.arange(s, device=dev)
ab[:, :, idx, idx] = C   # i-j==0
ref_diag = attention_ref(q, k, v, causal=True, softmax_scale=scale, attn_bias=ab)
if isinstance(ref_diag, tuple): ref_diag = ref_diag[0]
rB = flash_attn_func(q, k, v, rel_bias=relB, softmax_scale=scale, causal=True, num_splits=1)
outB = rB[0] if isinstance(rB, tuple) else rB
rep("relbias diag(C) vs ref", outB, ref_diag)
