# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton split-KV decode attention for Inkling (rel-bias, paged KV, GQA).

Why this exists (task #124): the FA4 SM120 score-mod kernel costs a flat
~380us/layer at decode -- rel_bias forces 128x128 tiles (96KB of the 99KB
SMEM), pack-GQA reduces the grid to ~num_kv_heads CTAs, and SM120 has no
split-KV, so a decode step uses ~2 CTAs on 188 SMs. At 66 layers that is
~25ms/token, the single largest decode cost.

This kernel handles ONLY the decode shape (one query token per request):

    out[t, h] = softmax(q[t, h] @ K[t, :sk].T * scale + bias) @ V[t, :sk]
    bias[j]   = rel_logits[t, h, dist]  if 0 <= dist < rel_extent else 0
                where dist = (sk - 1) - j
    sliding window (local layers): only j >= sk - 1 - window_left attend.

Layout contracts (matching attention.py's FA4 call):
  q          [T, H, D] bf16, one query token per request
  key_cache  [num_blocks, BLOCK, HKV, D] bf16 (paged)
  value_cache same
  block_table[T, max_blocks] int32
  cache_seqlens [T] int32 (seqused_k)
  rel_logits [T, H, rel_extent] bf16

Split-KV: grid (T * HKV * NSPLITS); each program computes an online-softmax
partial (m, l, acc) over its KV chunk; a small torch epilogue combines the
splits. No host syncs and fixed shapes for a fixed T -> FULL cudagraph safe.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# INKLING_ATTN_FUSED_COMBINE=1 (default off): replace the torch epilogue
# that merges the split-KV partials (max/exp/isnan/sum/div -- ~8 small
# kernels serialized per layer per decode pass) with one Triton kernel.
# Same streaming-softmax merge math; the -inf/empty-split guard is expressed
# as a mask instead of an isnan() fixup (identical values, no NaNs formed);
# split partials are accumulated in fp32 chunks so results differ from the
# torch reduction only by fp32 rounding.
_FUSED_COMBINE = os.environ.get("INKLING_ATTN_FUSED_COMBINE", "0") == "1"


@triton.jit
def _rel_decode_attn_kernel(
    q_ptr,  # [T, H, D] bf16
    k_ptr,  # [NB, BLOCK, HKV, D] bf16
    v_ptr,  # [NB, BLOCK, HKV, D] bf16
    bt_ptr,  # [T, MAXB] int32
    sk_ptr,  # [T] int32
    rel_ptr,  # [T, H, REL_EXT] bf16
    om_ptr,  # [NSPLITS, T, H] fp32 partial running max
    ol_ptr,  # [NSPLITS, T, H] fp32 partial sumexp
    oa_ptr,  # [NSPLITS, T, H, D] fp32 partial weighted V acc
    scale,
    T,
    stride_bt,  # block_table row stride
    # KV cache strides (task #125): _split_kv_cache returns STRIDED views —
    # K and V are packed in the last dim of one buffer, so the token stride
    # is HKV*2D and the head stride 2D, NOT the contiguous HKV*D / D the old
    # flat math assumed. That mismatch read the wrong storage element for
    # every key/value and was the decode-drift coherence regression.
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    HKV: tl.constexpr,
    GQA: tl.constexpr,  # q heads per kv head
    D: tl.constexpr,
    BLOCK: tl.constexpr,  # kv-cache page size
    BLOCK_KV: tl.constexpr,  # keys per inner iteration (multiple of BLOCK)
    REL_EXT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,  # -1 => no window
    NSPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % NSPLITS
    hk = (pid // NSPLITS) % HKV
    t = pid // (NSPLITS * HKV)
    H: tl.constexpr = HKV * GQA

    sk = tl.load(sk_ptr + t)
    lo = 0
    if WINDOW_LEFT >= 0:
        lo = tl.maximum(sk - 1 - WINDOW_LEFT, 0)
    total = sk - lo
    per_split = tl.cdiv(total, NSPLITS)
    start = lo + s * per_split
    end = tl.minimum(start + per_split, sk)

    rows = tl.arange(0, 16)  # GQA padded to 16 for tl.dot
    rmask = rows < GQA
    h = hk * GQA + tl.where(rmask, rows, 0)  # clamp padded rows in-bounds
    d = tl.arange(0, D)

    m_i = tl.full((16,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((16,), dtype=tl.float32)
    acc = tl.zeros((16, D), dtype=tl.float32)

    if start < end:
        qt = tl.load(
            q_ptr + (t * H + h)[:, None] * D + d[None, :],
            mask=rmask[:, None], other=0.0,
        )  # [16, D] bf16
        for kk in range(start, end, BLOCK_KV):
            js = kk + tl.arange(0, BLOCK_KV)
            jmask = js < end
            bid = tl.load(bt_ptr + t * stride_bt + js // BLOCK, mask=jmask,
                          other=0)
            koff = (bid.to(tl.int64) * stride_kb
                    + (js % BLOCK).to(tl.int64) * stride_kt + hk * stride_kh)
            voff = (bid.to(tl.int64) * stride_vb
                    + (js % BLOCK).to(tl.int64) * stride_vt + hk * stride_vh)
            kt = tl.load(k_ptr + koff[:, None] + d[None, :],
                         mask=jmask[:, None], other=0.0)  # [BLOCK_KV, D]
            scores = tl.dot(qt, tl.trans(kt)) * scale  # [16, BLOCK_KV] fp32
            dist = (sk - 1) - js
            bmask = (rmask[:, None] & jmask[None, :]
                     & (dist[None, :] >= 0) & (dist[None, :] < REL_EXT))
            bias = tl.load(
                rel_ptr + (t * H + h)[:, None] * REL_EXT + dist[None, :],
                mask=bmask, other=0.0,
            )
            scores += bias.to(tl.float32)
            scores = tl.where(jmask[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            # guard the shift for rows that have seen no valid key yet
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.where(m_i == float("-inf"), 0.0,
                             tl.exp(m_i - m_safe))
            p = tl.exp(scores - m_safe[:, None])
            p = tl.where(jmask[None, :], p, 0.0)
            vt = tl.load(v_ptr + voff[:, None] + d[None, :],
                         mask=jmask[:, None], other=0.0)  # [BLOCK_KV, D]
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    off = (s * T + t) * H + hk * GQA + rows
    tl.store(om_ptr + off, m_i, mask=rmask)
    tl.store(ol_ptr + off, l_i, mask=rmask)
    tl.store(oa_ptr + off[:, None] * D + d[None, :], acc,
             mask=rmask[:, None])


@triton.jit
def _combine_splits_kernel(
    om_ptr,  # [NSPLITS, T, H] fp32
    ol_ptr,  # [NSPLITS, T, H] fp32
    oa_ptr,  # [NSPLITS, T, H, D] fp32
    out_ptr,  # [T*H, D] target dtype rows (contiguous)
    TH,  # T * H
    NSPLITS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    th = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, NSPLITS)
    m_part = tl.load(om_ptr + s * TH + th)  # [NSPLITS]
    m = tl.max(m_part, axis=0)
    # exp(-inf - -inf) would be NaN only when EVERY split is empty (m ==
    # -inf); masking om == -inf to weight 0 reproduces the torch epilogue's
    # isnan()->0 fixup and the exp(-inf)=0 case in one expression.
    w_all = tl.where(
        m_part == float("-inf"), 0.0, tl.exp(m_part - m)
    )  # [NSPLITS]
    l_part = tl.load(ol_ptr + s * TH + th)
    l_tot = tl.maximum(tl.sum(l_part * w_all, axis=0), 1e-38)

    d = tl.arange(0, D)
    o = tl.zeros((D,), dtype=tl.float32)
    for s0 in range(0, NSPLITS, BLOCK_S):
        sb = s0 + tl.arange(0, BLOCK_S)
        mb = tl.load(om_ptr + sb * TH + th)
        wb = tl.where(mb == float("-inf"), 0.0, tl.exp(mb - m))
        ab = tl.load(
            oa_ptr + (sb * TH + th)[:, None] * D + d[None, :]
        )  # [BLOCK_S, D]
        o += tl.sum(ab * wb[:, None], axis=0)
    o = o / l_tot
    if OUT_BF16:
        tl.store(out_ptr + th * D + d, o.to(tl.bfloat16))
    else:
        tl.store(out_ptr + th * D + d, o)


def triton_rel_decode_attention(
    q: torch.Tensor,  # [T, H, D] bf16
    key_cache: torch.Tensor,  # [NB, BLOCK, HKV, D] bf16
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,  # [T, MAXB] int32
    cache_seqlens: torch.Tensor,  # [T] int32
    rel_logits: torch.Tensor,  # [T, H, REL_EXT] bf16
    softmax_scale: float,
    rel_extent: int,
    window_left: int = -1,  # -1 => full causal
    num_splits: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    T, H, D = q.shape
    NB, BLOCK, HKV, Dk = key_cache.shape
    assert Dk == D and H % HKV == 0
    gqa = H // HKV
    assert gqa <= 16, "GQA row padding is fixed at 16"
    assert rel_logits.shape == (T, H, rel_extent)
    # Strided cache views are the E2E norm (_split_kv_cache packs K/V in the
    # last dim of one buffer); only the innermost dim must be dense.
    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    assert q.is_contiguous()

    dev = q.device
    om = torch.empty(num_splits, T, H, dtype=torch.float32, device=dev)
    ol = torch.empty(num_splits, T, H, dtype=torch.float32, device=dev)
    oa = torch.empty(num_splits, T, H, D, dtype=torch.float32, device=dev)

    block_kv = 128
    assert block_kv % BLOCK == 0
    grid = (T * HKV * num_splits,)
    _rel_decode_attn_kernel[grid](
        q, key_cache, value_cache,
        block_table, cache_seqlens,
        rel_logits.contiguous(),
        om, ol, oa,
        softmax_scale,
        T,
        block_table.stride(0),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        HKV=HKV, GQA=gqa, D=D, BLOCK=BLOCK, BLOCK_KV=block_kv,
        REL_EXT=rel_extent, WINDOW_LEFT=window_left, NSPLITS=num_splits,
        num_warps=4,
    )

    # combine splits: standard streaming-softmax merge.
    if _FUSED_COMBINE and out is not None and out.dtype in (
        torch.bfloat16, torch.float32
    ):
        out_rows = out.view(T * H, D)
        if out_rows.is_contiguous():
            block_s = min(16, num_splits)
            assert num_splits % block_s == 0
            _combine_splits_kernel[(T * H,)](
                om, ol, oa, out_rows,
                T * H,
                NSPLITS=num_splits,
                D=D,
                BLOCK_S=block_s,
                OUT_BF16=out.dtype == torch.bfloat16,
                num_warps=4,
            )
            return out

    # Torch fallback: all fixed-shape ops.
    m = om.max(dim=0).values  # [T, H]
    w = torch.exp(om - m.unsqueeze(0))  # -inf partials -> exp(-inf)=0
    w = torch.where(torch.isnan(w), torch.zeros_like(w), w)  # all-empty guard
    l_tot = (ol * w).sum(dim=0).clamp_min(1e-38)  # [T, H]
    o = (oa * w.unsqueeze(-1)).sum(dim=0) / l_tot.unsqueeze(-1)  # [T, H, D]
    if out is None:
        return o.to(q.dtype)
    out.copy_(o.to(out.dtype).view(out.shape))
    return out
