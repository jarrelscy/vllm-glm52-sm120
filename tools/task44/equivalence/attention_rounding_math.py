# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent arithmetic ablation; native decode vs explicit rounding stages."""

import math

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(4)


def emu(q, raw, cpb, wp, bp, source_order=False):
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
    if source_order:
        scores = torch.zeros((q.shape[0], raw.shape[0]), device=q.device)
        for vc in range(4):
            scores += (
                q[:, vc * 128 : (vc + 1) * 128] @ vf[:, vc * 128 : (vc + 1) * 128].T
            ) * scales[:, vc][None]
        for rc in range(0, 64, 16):
            scores += q[:, 512 + rc : 512 + rc + 16] @ rope[:, rc : rc + 16].T
        scores *= 1 / 16
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
    spacing = (
        (
            torch.nextafter(
                b.bfloat16(), torch.full_like(b.bfloat16(), float("inf"))
            ).float()
            - b
        )
        .abs()
        .clamp_min(1e-38)
    )
    ulps = (a - b).abs() / spacing
    return dict(
        rel_l2=float((a - b).norm() / b.norm()),
        unequal=int((a != b).sum()),
        max_abs=float((a - b).abs().max()),
        over_one_ulp=int((ulps > 1.01).sum()),
        max_ulp=float(ulps.max()),
    )
