"""Characterize the SM120 plain-attention error structure."""
import torch
from vllm.third_party.tml_fa4 import flash_attn_func
from vllm.third_party.tml_fa4.testing import attention_ref

torch.manual_seed(0)
dev = "cuda"; DTYPE = torch.bfloat16
b, s, h, d = 1, 64, 4, 128
scale = 1.0 / d
q = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
k = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)
v = torch.randn(b, s, h, d, device=dev, dtype=DTYPE)

r = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=1)
out = (r[0] if isinstance(r, tuple) else r).float()[0]   # [s,h,d]
ref = attention_ref(q, k, v, causal=True, softmax_scale=scale)
if isinstance(ref, tuple): ref = ref[0]
ref = ref.float()[0]

bad = (out - ref).abs() > 2e-2 * (ref.abs() + 1)   # [s,h,d]
print("per-head bad%:", [f"{100*bad[:,hh,:].float().mean().item():.0f}" for hh in range(h)])
rowbad = bad.float().mean(dim=(1,2))  # [s]
print("per-row bad% (q rows 0..63):")
print(" ".join(f"{100*rowbad[i].item():.0f}" for i in range(s)))
dimbad = bad.float().mean(dim=(0,1))  # [d]
print("per-dim bad% (first 32 of 128):", " ".join(f"{100*dimbad[i].item():.0f}" for i in range(32)))
print("dim bad% blocks of 16:", [f"{100*dimbad[i*16:(i+1)*16].mean().item():.0f}" for i in range(8)])
print("\nrow0 head0 out[:6]:", out[0,0,:6].tolist())
print("row0 head0 ref[:6]:", ref[0,0,:6].tolist())
print("row63 head0 out[:6]:", out[63,0,:6].tolist())
print("row63 head0 ref[:6]:", ref[63,0,:6].tolist())
# is out maybe correct but permuted across heads?
for hh in range(h):
    dmin = min(((out[:,hh,:]-ref[:,hh2,:]).abs().mean().item(), hh2) for hh2 in range(h))
    print(f"out head {hh} closest ref head {dmin[1]} (mean_abs {dmin[0]:.4f})")
