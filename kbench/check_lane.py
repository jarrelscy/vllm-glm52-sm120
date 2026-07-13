#!/usr/bin/env python3
"""Bit-exactness check: GLM_MOE_LANE_ROWS=1 vs =0 on SM100, identical inputs.
Flags are read per-launch via getenv, so toggle os.environ between calls."""
import os, sys, pathlib, torch
sys.argv = ["x", "--compile-only"]  # reuse bench's build+weights
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_gemv as B

ext = B.build(os.environ.get("KB_SRC", str(B.REPO/"csrc/quantization/aqlm_moe/aqlm_moe_v2.cu")),
              "kb_ext", [])
dev = "cuda:0"; torch.cuda.set_device(dev)
w = B.make_weights(dev)
bad = 0
for tokens in (4, 64, 128):
    for mix in ("prod", "realdup"):
        a_ids, n_ids = B.make_slots(tokens, 0.6, mix, dev)
        S = tokens * B.TOPK
        g = torch.Generator().manual_seed(2)
        x13 = (torch.randn(S, B.H, generator=g)/8).to(torch.float16).to(dev)
        x2  = (torch.randn(S, B.I, generator=g)/8).to(torch.float16).to(dev)
        for nm, x, k in (("w13", x13, "w13"), ("w2", x2, "w2")):
            def call():
                return ext.hybrid_moe_gemv(x, w[f"{k}_codes"], w[f"{k}_cbs"], w[f"{k}_scales"],
                    a_ids, w[f"{k}_packed"], w[f"{k}_bscale"], w[f"{k}_scale2"], n_ids)
            os.environ["GLM_MOE_LANE_ROWS"] = "0"; os.environ["GLM_MOE_DEDUP"] = "0"
            o0 = call().clone()
            os.environ["GLM_MOE_LANE_ROWS"] = "1"
            o1 = call().clone()
            same = torch.equal(o0.view(torch.int16), o1.view(torch.int16))
            md = (o0.float()-o1.float()).abs().max().item()
            print(f"  t={tokens:3d} {mix:7s} {nm:3s}: bitexact(lane1==lane0)={same} maxdiff={md}")
            if not same: bad += 1
print("RESULT:", "ALL BIT-EXACT" if bad == 0 else f"{bad} MISMATCH")
