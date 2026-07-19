# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton varlen prefill attention for Inkling (rel-bias, paged KV, GQA).

Why this exists (task #123): the vendored FA4 SM120 kernel spends 0.76s of a
2.9s 4K prefill (26%) — the rel_bias score-mod forces 128x128 tiles and the
SM120 port leaves most of the machine idle, mirroring the decode pathology
fixed by ops/triton_decode_attention.py. This is the same online-softmax
machinery tiled over query blocks with causal + sliding-window masking.

Semantics (matching FA4's contract in attention.py):

    out[i, h] = softmax(q[i, h] @ K[:kv_i].T * scale + bias) @ V[:kv_i]
    bias[j]   = rel_logits[i, h, dist] if 0 <= dist < rel_extent else 0
                where dist = qpos_i - j
    causal    : key j attends iff dist >= 0
    window    : local layers additionally require dist <= window_left

qpos_i = seq_len_b - qlen_b + (i - qstart_b): the current chunk's K/V are
already written to the paged cache before attention runs (fused_qkvr_prep),
so the kernel scans the cache range [lo, qpos_max] with per-pair masking —
no separate self-attention pass.

Layout contracts:
  q           [NT, H, D] bf16 (varlen-packed query tokens)
  key_cache   [NB, BLOCK, HKV, D] bf16 (paged)
  value_cache same
  block_table [B, MAXB] int32
  cu_seqlens_q[B+1] int32
  seq_lens    [B] int32 (total KV length incl. the current chunk)
  rel_logits  [NT, H, REL_EXT] bf16

Eager-only is fine: prefill never runs under cudagraphs (FULL_DECODE_ONLY).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rel_prefill_attn_kernel(
    q_ptr,  # [NT, H, D] bf16
    k_ptr,  # [NB, BLOCK, HKV, D] bf16
    v_ptr,  # [NB, BLOCK, HKV, D] bf16
    bt_ptr,  # [B, MAXB] int32
    qs_ptr,  # [B+1] int32 cu_seqlens_q
    sk_ptr,  # [B] int32 total kv len per request
    rel_ptr,  # [NT, H, REL_EXT] bf16
    out_ptr,  # [NT, H, D] bf16
    scale,
    stride_bt,
    # KV cache strides (task #125): _split_kv_cache returns STRIDED views —
    # K and V are packed in the last dim of one buffer, so the token stride
    # is HKV*2D and the head stride 2D, NOT the contiguous HKV*D / D the old
    # flat math assumed. That mismatch read the wrong storage element for
    # every key/value and was the first-token-garbage coherence regression.
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    H: tl.constexpr,
    HKV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,  # kv-cache page size
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    REL_EXT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,  # -1 => no window
    MAX_MBLOCKS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    b = pid // MAX_MBLOCKS
    mb = pid % MAX_MBLOCKS
    hk = h // GQA

    qstart = tl.load(qs_ptr + b)
    qlen = tl.load(qs_ptr + b + 1) - qstart
    if mb * BLOCK_M >= qlen:
        return
    sk = tl.load(sk_ptr + b)

    ms = mb * BLOCK_M + tl.arange(0, BLOCK_M)  # in-request query index
    m_ok = ms < qlen
    qtok = qstart + tl.where(m_ok, ms, qlen - 1)  # [BM] global token row
    qpos = sk - qlen + ms  # kv position of each query row (masked rows ok)

    d = tl.arange(0, D)
    qt = tl.load(
        q_ptr + (qtok.to(tl.int64) * H + h)[:, None] * D + d[None, :],
        mask=m_ok[:, None], other=0.0,
    )  # [BM, D] bf16

    # KV scan range for this block: causal upper bound from the last valid
    # row; window lower bound from the first row.
    qpos_lo = sk - qlen + mb * BLOCK_M
    qpos_hi = sk - qlen + tl.minimum(mb * BLOCK_M + BLOCK_M, qlen) - 1
    lo = 0
    if WINDOW_LEFT >= 0:
        lo = tl.maximum(qpos_lo - WINDOW_LEFT, 0)
        # keep page-aligned starts so k/v loads stay coalesced
        lo = (lo // BLOCK) * BLOCK
    hi = qpos_hi + 1

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for kk in tl.range(lo, hi, BLOCK_KV, num_stages=NUM_STAGES):
        js = kk + tl.arange(0, BLOCK_KV)
        jmask = js < hi
        bid = tl.load(bt_ptr + b * stride_bt + js // BLOCK, mask=jmask, other=0)
        koff = (bid.to(tl.int64) * stride_kb
                + (js % BLOCK).to(tl.int64) * stride_kt + hk * stride_kh)
        voff = (bid.to(tl.int64) * stride_vb
                + (js % BLOCK).to(tl.int64) * stride_vt + hk * stride_vh)
        kt = tl.load(k_ptr + koff[:, None] + d[None, :],
                     mask=jmask[:, None], other=0.0)  # [BKV, D]
        scores = tl.dot(qt, tl.trans(kt)) * scale  # [BM, BKV] fp32

        dist = qpos[:, None] - js[None, :]
        allow = jmask[None, :] & (dist >= 0)
        if WINDOW_LEFT >= 0:
            allow = allow & (dist <= WINDOW_LEFT)
        bmask = allow & (dist < REL_EXT) & m_ok[:, None]
        bias = tl.load(
            rel_ptr + (qtok.to(tl.int64) * H + h)[:, None] * REL_EXT + dist,
            mask=bmask, other=0.0,
        )
        scores += bias.to(tl.float32)
        scores = tl.where(allow, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.exp(scores - m_safe[:, None])
        p = tl.where(allow, p, 0.0)
        vt = tl.load(v_ptr + voff[:, None] + d[None, :],
                     mask=jmask[:, None], other=0.0)  # [BKV, D]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # every valid row saw at least its own key (current chunk is in-cache)
    o = acc / tl.maximum(l_i, 1e-38)[:, None]
    tl.store(
        out_ptr + (qtok.to(tl.int64) * H + h)[:, None] * D + d[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=m_ok[:, None],
    )


def triton_rel_prefill_attention(
    q: torch.Tensor,  # [NT, H, D] bf16
    key_cache: torch.Tensor,  # [NB, BLOCK, HKV, D] bf16
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,  # [B, MAXB] int32
    cu_seqlens_q: torch.Tensor,  # [B+1] int32
    cache_seqlens: torch.Tensor,  # [B] int32
    max_seqlen_q: int,
    rel_logits: torch.Tensor,  # [NT, H, REL_EXT] bf16
    softmax_scale: float,
    rel_extent: int,
    window_left: int = -1,
    out: torch.Tensor | None = None,
    block_m: int = 64,
    block_kv: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    NT, H, D = q.shape
    NB, BLOCK, HKV, Dk = key_cache.shape
    assert Dk == D and H % HKV == 0
    B = cache_seqlens.shape[0]
    assert cu_seqlens_q.shape[0] == B + 1
    assert rel_logits.shape == (NT, H, rel_extent)
    assert block_kv % BLOCK == 0
    # Strided cache views are the E2E norm (_split_kv_cache packs K/V in the
    # last dim of one buffer); only the innermost dim must be dense.
    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    assert q.is_contiguous()

    if out is None:
        out = torch.empty_like(q)
    o = out.view(NT, H, D)

    max_mblocks = triton.cdiv(max_seqlen_q, block_m)
    grid = (B * max_mblocks, H)
    _rel_prefill_attn_kernel[grid](
        q, key_cache, value_cache,
        block_table, cu_seqlens_q, cache_seqlens,
        rel_logits.contiguous(),
        o,
        softmax_scale,
        block_table.stride(0),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        H=H, HKV=HKV, GQA=H // HKV, D=D, BLOCK=BLOCK,
        BLOCK_M=block_m, BLOCK_KV=block_kv,
        REL_EXT=rel_extent, WINDOW_LEFT=window_left,
        MAX_MBLOCKS=max_mblocks,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    return out
