"""Authors'-convention probe: flash_attn_func (batched) vs the vendored
attention_ref, plain causal, no rel_bias. Eliminates harness doubt."""
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

def rep(name, out, ref):
    out = out.float(); ref = ref.float()
    diff = (out - ref).abs()
    bad = (diff > 2e-2 * (ref.abs() + 1)).sum().item()
    print(f"[{name}] max_abs={diff.max().item():.4f} mean={diff.mean().item():.5f} "
          f"bad={bad}/{ref.numel()} ({100*bad/ref.numel():.1f}%) nan={torch.isnan(out).any().item()}")

for nsp in (1,):
    try:
        r = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=nsp)
        out = r[0] if isinstance(r, tuple) else r
        ref = attention_ref(q, k, v, causal=True, softmax_scale=scale)
        if isinstance(ref, tuple): ref = ref[0]
        rep(f"flash_attn_func ns={nsp}", out, ref)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("EXC:", type(e).__name__, str(e)[:300])
