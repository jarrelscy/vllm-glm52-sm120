"""Reveal the kernel's rel_bias offset->(i-j) mapping behaviorally.
One-hot rel_logits at offset d0 + tiny softmax_scale => output[i] ~= V[argmax_j bias]
=> the key j the kernel biases for offset d0. Spec: bias at i-j==d0 => j=i-d0."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_func

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
b, s, h, d = 1, 128, 1, 128
REL = 128
scale = 1e-4  # near-zero so QK contributes ~nothing; bias dominates
q = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
k = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
# make V rows easily identifiable: V[j] = one-hot-ish distinct pattern (use j as scale)
v = torch.zeros(b, s, h, d, device=dev, dtype=DTYPE)
for j in range(s):
    v[0, j, 0, 0] = float(j)   # V[j][0] = j, so output[i][0] reveals which key dominated

for d0 in (0, 1, 5, 20):
    rel = torch.zeros(b, s, h, REL, device=dev, dtype=DTYPE)
    rel[..., d0] = 50.0
    r = flash_attn_func(q, k, v, rel_bias=rel, softmax_scale=scale, causal=True, num_splits=1)
    out = (r[0] if isinstance(r, tuple) else r).float()
    # For query i (>= d0), spec expects dominant key j=i-d0 => out[i,0] ~= i-d0
    got = out[0, :, 0, 0]  # [s]
    # measure, for i in [d0+2 .. s-1], (i - got[i]) which should == d0 if spec-correct
    isr = torch.arange(s, device=dev).float()
    implied_offset = (isr - got)  # i - j_dominant ; expect == d0
    sample_i = [d0+2, d0+10, s//2, s-1]
    vals = [(int(i), round(got[i].item(),1), round(implied_offset[i].item(),1)) for i in sample_i if i < s]
    print(f"d0={d0}: (i, out[i]=~j_dominant, i-j)={vals}  spec expects i-j=={d0}")
