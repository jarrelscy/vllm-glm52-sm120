# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent arithmetic ablation; native decode vs explicit rounding stages."""

import json
import math

import torch
from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv3_2

from vllm import _custom_ops as ops

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(4)


def emu(q, raw, cpb, wp, bp):
    q = q.float().clone()
    qt = q[:, :512].reshape(-1, 4, 128)
    qs = torch.exp2(
        torch.ceil(torch.log2(qt.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    )
    q[:, :512] = ((qt / qs).to(torch.float8_e4m3fn).float() * qs).flatten(1)
    vf = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = raw[:, 512:528].contiguous().view(torch.float32)
    v = vf * scales.repeat_interleave(128, -1)
    rope = raw[:, 528:].contiguous().view(torch.bfloat16).float()
    scores = (q @ torch.cat((v, rope), -1).T) / 16
    parts = []
    lses = []
    for lo in range(0, len(raw), 64 * cpb):
        hi = min(lo + 64 * cpb, len(raw))
        mx = torch.full((q.shape[0], 1), -torch.inf, device=q.device)
        den = torch.zeros_like(mx)
        acc = torch.zeros((q.shape[0], 512), device=q.device)
        for k in range(lo, hi, 64):
            end = min(k + 64, hi)
            sc = scores[:, k:end]
            nm = torch.maximum(mx, sc.amax(-1, keepdim=True))
            alpha = (mx - nm).exp()
            p = (sc - nm).exp()
            acc *= alpha
            den = den * alpha + p.sum(-1, keepdim=True)
            mx = nm
            for vc in range(4):
                weights = p * scales[k:end, vc][None]
                if wp:
                    ws = weights.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
                    wn = weights / ws
                    high = wn.to(torch.float8_e4m3fn).float()
                    residual = (wn - high).to(torch.float8_e4m3fn).float()
                    weights = (high + residual) * ws
                acc[:, vc * 128 : (vc + 1) * 128] += (
                    weights @ vf[k:end, vc * 128 : (vc + 1) * 128]
                )
        partial = acc / den
        if bp:
            partial = partial.bfloat16().float()
        parts.append(partial)
        lses.append((mx + den.log()).squeeze(-1))
    lse = torch.stack(lses, -1)
    w = lse.softmax(-1)
    result = (torch.stack(parts, 1) * w[:, :, None]).sum(1)
    return result.bfloat16(), torch.logsumexp(lse, -1) / math.log(2)


def error(a, b):
    a, b = a.float(), b.float()
    return dict(
        rel_l2=float((a - b).norm() / b.norm()),
        unequal=int((a != b).sum()),
        max_abs=float((a - b).abs().max()),
    )


results = []
for heads in (16, 64):
    for n in (128, 2048):
        torch.manual_seed(123 + heads + n)
        kv = (
            torch.randn(n, 512, device="cuda")
            * torch.exp(torch.randn(n, 4, device="cuda")).repeat_interleave(128, -1)
        ).bfloat16()
        rope = torch.randn(n, 64, device="cuda").bfloat16()
        raw = torch.zeros(n // 64, 64, 656, device="cuda", dtype=torch.uint8)
        ops.concat_and_cache_mla(
            kv,
            rope,
            raw,
            torch.arange(n, device="cuda"),
            kv_cache_dtype="fp8_ds_mla",
            scale=torch.tensor(1.0, device="cuda"),
        )
        q = (torch.randn(1, heads, 576, device="cuda") * 0.3).bfloat16()
        idx = torch.arange(n, device="cuda", dtype=torch.int32)[None]
        for cpb in sorted(set((1, min(4, n // 64), n // 64))):
            mid = torch.empty(
                1, heads, n // 64, 512, device="cuda", dtype=torch.bfloat16
            )
            ml = torch.empty(1, heads, n // 64, device="cuda")
            out = torch.empty(1, heads, 512, device="cuda", dtype=torch.bfloat16)
            ol = torch.empty(1, heads, device="cuda")
            sparse_mla_sm120_decode_dsv3_2(
                q,
                raw,
                idx,
                mid,
                ml,
                out,
                ol,
                1 / 16,
                model_type=2,
                chunks_per_block=cpb,
            )
            row = dict(heads=heads, n=n, cpb=cpb, arms={})
            for wp, bp in ((False, False), (True, False), (False, True), (True, True)):
                ref, lse = emu(q[0], raw.reshape(-1, 656), cpb, wp, bp)
                row["arms"][f"fp8w={wp},bf16partial={bp}"] = dict(
                    **error(out[0], ref), lse_abs=float((ol[0] - lse).abs().max())
                )
            results.append(row)
            print(json.dumps(row), flush=True)
with open("/work/attention_rounding_results.json", "w") as output:
    output.write(json.dumps(results, indent=2))
