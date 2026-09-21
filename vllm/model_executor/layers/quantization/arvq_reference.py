# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slow, eager PyTorch references for serialized ARVQ/NVFP4 and sparse attention.

No custom CUDA, FP4 MMA, fused packing, or persistent decoded-weight cache.
Disable TF32 in the diagnostic runner. These functions intentionally synchronize
when dispatching experts and cannot be captured in CUDA graphs.
"""

import torch

# This module is imported only by the diagnostic opt-in paths.
torch.backends.cuda.matmul.allow_tf32 = False


def stable_indexer_topk(logits, starts, ends, output):
    """Reference selection: descending score, low index at ties, canonical order.

    Bounds address columns in each logits row. Returned indices are relative
    to each row's start, matching the native prefill selector. Invalid output
    positions are -1. This diagnostic intentionally synchronizes row bounds.
    """
    output.fill_(-1)
    bounds = zip(starts.flatten().tolist(), ends.flatten().tolist())
    for row, (start, end) in enumerate(bounds):
        start = max(0, min(start, logits.shape[1]))
        end = max(start, min(end, logits.shape[1]))
        count = min(end - start, output.shape[1])
        if count:
            selected = torch.argsort(
                logits[row, start:end], descending=True, stable=True
            )[:count]
            output[row, :count] = selected.sort(descending=True).values


def fp4(codes):
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=codes.device, dtype=torch.float32
    )
    return levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)


def e4m3(codes):
    """Unsigned finite E4M3 block scales; reject invalid serialized values."""
    codes = codes.long()
    if bool(((codes < 0) | (codes >= 127)).any()):
        raise ValueError("Expected unsigned finite E4M3 scales")
    exponent, mantissa = codes >> 3, codes & 7
    return torch.where(
        exponent == 0,
        mantissa.float() * (2.0**-9),
        (1 + mantissa.float() / 8) * torch.pow(2.0, exponent.float() - 7),
    )


def activation_planes(x, planes=4):
    """Independent scalar quantization semantics, including residual x16 steps."""
    if planes not in (1, 4) or x.shape[-1] % 16:
        raise ValueError("Expected one/four planes and K divisible by 16")
    residual = x.to(torch.float16).float().reshape(*x.shape[:-1], -1, 16)
    result = torch.zeros_like(residual)
    thresholds = residual.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    for plane in range(planes):
        maximum = residual.abs().amax(-1, keepdim=True)
        exponent = torch.ceil(torch.log2((maximum / 6).clamp_min(2.0**-20)))
        scale = torch.pow(2.0, exponent.clamp(-6, 8))
        magnitude = (residual.abs() / scale).contiguous()
        # bucketize(right=False) matches the kernel's strict > thresholds.
        code = torch.bucketize(magnitude, thresholds) | ((residual < 0).long() * 8)
        decoded = fp4(code) * scale
        result += decoded * (16.0**-plane)
        residual = (residual - decoded) * 16
    return result.reshape(x.shape)


def cold_rows(
    packed,
    codebooks,
    scales,
    n,
    k,
    expert,
    first=0,
    stop=None,
    selectors=None,
    book_factors=None,
):
    """Decode physical tiled 8+7/8+8 storage into ordinary FP32 matrix rows.

    mcbook16 (4352-entry expert codebooks): selectors [E, N/16, K/64] pick one
    of 16 residual books per tile and book_factors [16] weight its atoms.
    """
    stop = n if stop is None else stop
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
        _expert_codebooks,
    )

    codebooks = _expert_codebooks(codebooks, expert)
    if codebooks.numel() not in (384, 512, 4352) or n % 16 or k % 128:
        raise ValueError("Invalid ARVQ layout")
    mb16 = codebooks.numel() == 4352
    if mb16 != (selectors is not None) or (mb16 and book_factors is None):
        raise ValueError("mcbook16 codebooks require selectors and book_factors")
    bits = 15 if codebooks.numel() == 384 else 16
    device = packed.device
    row = torch.arange(first, stop, device=device)[:, None]
    col = torch.arange(0, k, 8, device=device)[None, :]
    tile = (expert * (n // 16) + row // 16) * (k // 64) + col // 64
    position = ((row % 16) // 8 + 2 * ((col % 64) // 32)) * 32
    position = position + (row % 8) * 4 + (col % 32) // 8
    bit = position * bits
    word = tile * (4 * bits) + bit // 32
    storage = packed.flatten()
    pair = (storage[word].long() & 0xFFFFFFFF) >> (bit % 32)
    if bits == 15:
        # Last crossing uses the documented one-word guard.
        pair |= (storage[word + 1].long() & 0xFFFFFFFF) << (32 - bit % 32)
    cb = codebooks.flatten().long() & 0xFFFFFFFF
    a = cb[pair & 255]
    residual = (pair >> 8) & ((1 << (bits - 8)) - 1)
    if mb16:
        m = selectors[expert, row // 16, col // 64].long()
        b = cb[256 + m * 256 + residual]
        factor = book_factors.float()[m][..., None]
    else:
        b = cb[256 + residual]
        factor = 1.0
    shifts = torch.arange(8, device=device) * 4
    values = fp4((a[..., None] >> shifts) & 15)
    values += fp4((b[..., None] >> shifts) & 15) * factor
    block = scales.reshape(-1, n // 16, k // 128, 16)
    scale = e4m3(block[expert, row // 16, col // 128, row % 16])
    return (values * scale[..., None]).reshape(stop - first, k)


def hot_rows(packed, scales, n, k, expert, first=0, stop=None):
    """Decode native MMA-layout NVFP4, including one scale per 16 columns."""
    stop = n if stop is None else stop
    row = torch.arange(first, stop, device=packed.device)[:, None]
    col = torch.arange(0, k, 8, device=packed.device)[None, :]
    j = (row % 16) // 8 + 2 * ((col % 64) // 32)
    lane = (row % 8) * 4 + (col % 32) // 8
    words = packed.reshape(-1, n // 16, k // 64, 4, 32)
    values = words[expert, row // 16, col // 64, j, lane].long() & 0xFFFFFFFF
    shifts = torch.arange(8, device=packed.device) * 4
    values = fp4((values[..., None] >> shifts) & 15)
    sw = scales.reshape(-1, n, k // 64)[expert, row, col // 64].long()
    scale = e4m3((sw >> (8 * ((col % 64) // 16))) & 255)
    return (values * scale[..., None]).reshape(stop - first, k)


def projection(x, cold_ids, hot_ids, tensors, alpha, n, split=1, hot_parts=1):
    """Drop-in eager projection reference with bounded temporary weight storage."""
    del split
    cw, cb, cs, hw, hs, hg = tensors[:6]
    # 8-wide groups append mcbook16 selectors/book_factors (zero-element
    # placeholders on v4 projections).
    sel = tensors[6] if len(tensors) > 6 and tensors[6].numel() else None
    fac = tensors[7] if sel is not None else None
    k = x.shape[-1]
    activation = activation_planes(x)
    out = torch.zeros((x.shape[0], n), device=x.device, dtype=torch.float32)
    for cold, ids in ((True, cold_ids), (False, hot_ids)):
        valid = ids >= 0
        if not cold:
            valid &= cold_ids < 0
        for expert in ids[valid].unique().tolist():
            slots = torch.where(valid & (ids == expert))[0]
            for first in range(0, n, 1024):
                stop = min(first + 1024, n)
                if cold:
                    weights = cold_rows(
                        cw, cb, cs, n, k, expert, first, stop,
                        selectors=sel, book_factors=fac,
                    )
                    scale = alpha
                else:
                    weights = hot_rows(hw, hs, n, k, expert, first, stop)
                    parts = torch.arange(first, stop, device=x.device) // (
                        n // hot_parts
                    )
                    scale = hg.reshape(-1, hot_parts)[expert, parts].float()
                out[slots, first:stop] = (activation[slots] @ weights.T) * scale
    return out


def sparse_attention(q, keys, values, indices, scale):
    """Explicit FP32 sparse attention; duplicate indices retain multiplicity.

    q: [tokens, heads, qk_dim]; keys/values: globally addressed [slots, dim].
    indices: [tokens, selected_slots], negative entries are empty. Caller owns
    cache decoding, causal selection, and optional kernel-matched Q quantization.
    Returns FP32 output and base-2 LSE; empty rows are zero / negative infinity.
    """
    out = torch.zeros((*q.shape[:2], values.shape[-1]), device=q.device)
    lse2 = torch.full(q.shape[:2], -torch.inf, device=q.device)
    for token in range(q.shape[0]):
        selected = indices[token]
        selected = selected[selected >= 0].long()
        if selected.numel() == 0:
            continue
        scores = (q[token].float() @ keys[selected].float().T) * scale
        out[token] = torch.softmax(scores, dim=-1) @ values[selected].float()
        lse2[token] = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
    return out, lse2


def paged_attention(q, cache, indices, scale, seq_lens=None):
    """Reference for the 656-byte SM120 fp8_ds_mla cache, one query at a time.

    Gather only selected cache entries, avoiding a full-context FP32 KV copy.
    Physical index conversion and, when enabled, DCP combine remain caller-owned.
    """
    if q.shape[-1] != 576 or cache.shape[-1] != 656:
        raise ValueError("Reference supports only 512 latent + 64 rope fp8_ds_mla")
    qf = q.float().clone()
    tiles = qf[..., :512].reshape(*q.shape[:2], 4, 128)
    qs = torch.exp2(
        torch.ceil(torch.log2(tiles.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    )
    qf[..., :512] = ((tiles / qs).to(torch.float8_e4m3fn).float() * qs).flatten(-2)
    raw = cache.view(torch.uint8).reshape(-1, 656)
    out = torch.zeros((*q.shape[:2], 512), device=q.device, dtype=torch.float32)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device, dtype=torch.float32)
    for token in range(q.shape[0]):
        selected = indices[token].flatten()
        if seq_lens is not None:
            selected = selected[: int(seq_lens.flatten()[token])]
        selected = selected[selected >= 0].long()
        if not selected.numel():
            continue
        entries = raw[selected]
        values = entries[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = entries[:, 512:528].contiguous().view(torch.float32)
        values *= scales.repeat_interleave(128, -1)
        rope = entries[:, 528:].contiguous().view(torch.bfloat16).float()
        keys = torch.cat((values, rope), -1)
        local_indices = torch.arange(len(selected), device=q.device)[None]
        row_out, row_lse = sparse_attention(
            qf[token : token + 1], keys, values, local_indices, scale
        )
        out[token], lse[token] = row_out[0], row_lse[0]
    return out.to(q.dtype), lse
