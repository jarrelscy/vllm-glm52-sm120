# SPDX-License-Identifier: Apache-2.0
"""Fused quantized gemv kernels for the Inkling hybrid MoE decode path.

Decode profiling (2026-07-18, FULL_DECODE_ONLY cudagraphs at 8K) showed the
decode step is GPU-work-bound at ~0.95 s/token: ``_apply_decode_graphsafe``
materializes each routed expert's full w13/w2 to bf16 (with fp32 LUT-gather /
``repeat_interleave`` intermediates) per token per MoE layer just to do a
single-token gemv, and the ``torch.where(is_hot, hot, cold)`` select pays for
BOTH dequants on every slot. That is hundreds of MB of memory traffic per
slot; graph capture removed launch overhead and changed nothing (1.046 ->
1.049 tok/s), proving the cost is the traffic itself.

These kernels compute ``y[slot] = W_slot @ x[slot]`` reading the quantized
codes directly — NVFP4 nibbles + fp8 block scales for hot experts, AQLM
codes + codebooks for cold experts — decoding in-register and never
materializing a bf16 weight tensor. Hot-vs-cold is selected per slot by a
SCALAR predicate on the load masks, so a slot only reads the format it
actually uses.

Graph-safety: grid shape and all trip counts depend only on (S, N, K);
``is_hot``/indices are consumed as device tensor VALUES. No host syncs.

Layout facts (must match hybrid_moe.py create_weights / dequant reference):
- hot NVFP4: packed uint8 [n_hot, N, K//2] (low nibble = even k, high = odd),
  scale fp8_e4m3fn [n_hot, N, K//16] (per 16 in-elements), scale2 fp32
  [n_hot] per expert. value = lut[nib] * scale * scale2.
- hot bf16: plain [n_hot, N, K].
- cold AQLM: per book b, codes [n_cold, N, K//8] (int16 or uint8), codebook
  [entries_b, 8] fp16; weight = (sum_b cb_b[code_b & (entries_b-1)]) *
  scales[n_cold, N] (per out-row). All book sizes are powers of two.
"""

import os

import torch
import triton
import triton.language as tl

_G = tl.constexpr(8)  # AQLM group size along the in (K) dim

# INKLING_GEMV_V2 (default ON since task #137; =0 disables): hot-NVFP4 gemv
# load-issue variant. The baseline gathers the per-16-element fp8 block
# scale once PER PACKED BYTE (BLOCK_K//2 gathers per row per k-chunk, 8x
# redundant) and the activations as two stride-2 gathers; V2 loads the
# BLOCK_K//16 distinct scale bytes and expands in-register via
# tl.interleave (the _grouped_hot_acc pattern), and loads the activation
# chunk contiguously, splitting even/odd in-register. Decoded values,
# products and the tl.sum reduction shape are IDENTICAL to the baseline, so
# results are bit-identical (gated by the exact-equality test in
# tests/models/inkling/test_hybrid_moe.py). Although only the hot branch
# changes, the win is kernel-wide: the fat hot branch sets the kernel's
# register footprint, so shrinking it lifts occupancy for the (dominant)
# cold-AQLM programs too. Measured: 1.36x on the gemv pair at S=6
# (161 -> 118 us), server ns=0 decode 42 -> 50.7 tok/s stacked with the
# fused attention combine.
_GEMV_V2 = os.environ.get("INKLING_GEMV_V2", "1") == "1"

# INKLING_GEMV_SPLITK=<k> (default 1 = off): split the gemv K-loop across k
# programs (each owns every k-th BLOCK_K chunk), writing fp32 partials to a
# [S, k, N] buffer reduced by _splitk_reduce_kernel. Rationale: at decode
# S = num_tokens*top_k is tiny, the grid is ~1-2k programs on 188 SMs and
# each program serially walks K/BLOCK_K dependent DRAM reads, so the kernel
# runs memory-LATENCY-bound far below peak bandwidth; split-K multiplies the
# in-flight parallelism without changing per-element math. Accumulation
# order changes (chunk partials summed in a fixed deterministic order), so
# results differ from the baseline only by fp32 rounding (see the
# equivalence test's tight tolerance). Graph-safe: shapes depend only on
# (S, N, K, k).
# VERDICT (task #137 microbench @real shapes): weak -- 1.08x alone, ~0 on
# top of _GEMV_V2 (1.36x -> 1.38x). The latency problem was register
# footprint (V2), not K-parallelism. Kept for future shapes; default off.
_GEMV_SPLITK = max(1, int(os.environ.get("INKLING_GEMV_SPLITK", "1")))

# INKLING_GROUPED_SPLITK=<k> (default 1 = off): same split-K treatment for
# the graph-safe grouped decode path (_apply_decode_grouped -- the MTP
# verify batch). The decode grouped grid is also tiny (S candidate m-blocks
# x N/BLOCK_N), so the same latency-bound argument applies. Decode-regime
# only: the prefill caller keeps split_k=1 (partials would be [S, k, N] at
# prefill S). Rounding-only numerics, same reduce kernel.
# VERDICT (task #137 microbench): 1.04-1.06x at S=18, <=1.0x at S=36 --
# not worth graph memory; superseded anyway by _DECODE_GROUPED_DISABLED
# default (verify now takes the V2 gemv path). Kept for triage; default off.
_GROUPED_SPLITK = max(1, int(os.environ.get("INKLING_GROUPED_SPLITK", "1")))

# NVFP4 magnitude table for code&7: 0, .5, 1, 1.5, 2, 3, 4, 6


@triton.jit
def _nvfp4_mag(nib):
    """Magnitude of an NVFP4 code's low 3 bits (float32)."""
    m = (nib & 7).to(tl.float32)
    # m in {0..7} -> {0, .5, 1, 1.5, 2, 3, 4, 6}
    small = m * 0.5  # correct for m < 4
    big = tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0)))
    return tl.where(m < 4, small, big)


@triton.jit
def _fp8e4m3_decode(b):
    """Decode a uint8 holding an fp8 e4m3fn bit pattern to float32."""
    s = (b >> 7) & 1
    e = ((b >> 3) & 0xF).to(tl.float32)
    m = (b & 7).to(tl.float32)
    sub = m * 0.001953125  # m * 2^-9 (subnormal: e == 0)
    nrm = (8.0 + m) * tl.exp2(e - 10.0)
    v = tl.where(e == 0.0, sub, nrm)
    return tl.where(s == 1, -v, v)


@triton.jit
def _hybrid_gemv_kernel(
    # inputs
    x_ptr,  # [S, K] bf16 (already gathered per slot)
    is_hot_ptr,  # [S] int32 (1 = hot, 0 = cold)
    hot_idx_ptr,  # [S] int32 local hot index (valid even when cold: clamped 0)
    cold_idx_ptr,  # [S] int32 local cold index (valid even when hot)
    # hot expert weights (NVFP4 or bf16 depending on HOT_MODE)
    hot_packed_ptr,  # nvfp4: uint8 [n_hot, N, K//2]; bf16: bf16 [n_hot, N, K]
    hot_scale_ptr,  # nvfp4: uint8 view of fp8 [n_hot, N, K//16]; unused for bf16
    hot_scale2_ptr,  # fp32 [n_hot]; unused for bf16
    # cold expert weights (AQLM, up to 2 books)
    cold_codes0_ptr,  # [n_cold, N, K//8] int16-or-uint8
    cold_cb0_ptr,  # [entries0, 8] fp16
    cold_codes1_ptr,  # book 1 (dummy = book 0 ptrs when NUM_BOOKS == 1)
    cold_cb1_ptr,
    cold_scales_ptr,  # [n_cold, N] fp16
    # output
    out_ptr,  # [S, N] fp32
    # sizes
    N, K,
    ENTRIES0: tl.constexpr,
    ENTRIES1: tl.constexpr,
    NUM_BOOKS: tl.constexpr,
    HOT_MODE: tl.constexpr,  # 0 = none, 1 = nvfp4, 2 = bf16
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_EXPAND: tl.constexpr = False,  # INKLING_GEMV_V2 (see module docs)
    SPLIT_K: tl.constexpr = 1,  # INKLING_GEMV_SPLITK (see module docs)
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    if SPLIT_K == 1:
        sk = 0
        s = pid // n_blocks
        nb = pid % n_blocks
    else:
        sk = pid % SPLIT_K
        pid2 = pid // SPLIT_K
        s = pid2 // n_blocks
        nb = pid2 % n_blocks

    rows = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    row_ok = rows < N
    # SPLIT_K == 1 writes out[s, n]; SPLIT_K > 1 writes partials[s, sk, n].
    out_base = out_ptr + (s * SPLIT_K + sk) * N

    ih = tl.load(is_hot_ptr + s)
    he = tl.load(hot_idx_ptr + s).to(tl.int64)
    ce = tl.load(cold_idx_ptr + s).to(tl.int64)

    # Hot-vs-cold is a uniform runtime branch on the slot's scalar predicate:
    # a program executes only its own format's decode ALU. The previous
    # masked-both-paths version skipped the OTHER format's memory traffic but
    # still paid its full dequant ALU on every slot, just to discard it in a
    # tl.where. Arithmetic within each branch is unchanged (bit-identical).
    if (HOT_MODE != 0) and (ih != 0):
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(sk, tl.cdiv(K, BLOCK_K), SPLIT_K):
            k = k0 * BLOCK_K
            if HOT_MODE == 1:
                # bytes j cover in-positions (k + 2j, k + 2j + 1)
                j = tl.arange(0, BLOCK_K // 2)
                xm = (k + 2 * j) < K
                if SCALE_EXPAND:
                    # One contiguous activation load; even/odd split in-register.
                    kk = k + tl.arange(0, BLOCK_K)
                    xb = tl.load(
                        x_ptr + s * K + kk, mask=kk < K, other=0.0
                    ).to(tl.float32)
                    x_even, x_odd = tl.split(
                        tl.reshape(xb, (BLOCK_K // 2, 2))
                    )
                else:
                    x_even = tl.load(
                        x_ptr + s * K + k + 2 * j, mask=xm, other=0.0
                    ).to(tl.float32)
                    x_odd = tl.load(
                        x_ptr + s * K + k + 2 * j + 1, mask=xm, other=0.0
                    ).to(tl.float32)
                p_off = (he * N + rows.to(tl.int64))[:, None] * (K // 2) + (k // 2) + j[None, :]
                pk = tl.load(
                    hot_packed_ptr + p_off,
                    mask=row_ok[:, None] & xm[None, :],
                    other=0,
                )
                lo = _nvfp4_mag(pk & 0xF) * tl.where((pk & 0x8) != 0, -1.0, 1.0)
                hi = _nvfp4_mag((pk >> 4) & 0xF) * tl.where((pk & 0x80) != 0, -1.0, 1.0)
                # per-16-in-element block scale; byte j sits in block (k + 2j)//16
                if SCALE_EXPAND:
                    # Load each distinct scale byte once, expand 8x along j
                    # (each 16-in-element block covers 8 packed bytes).
                    sb = tl.arange(0, BLOCK_K // 16)
                    sbm = (k + 16 * sb) < K
                    sc8 = _fp8e4m3_decode(
                        tl.load(
                            hot_scale_ptr
                            + (he * N + rows.to(tl.int64))[:, None] * (K // 16)
                            + (k // 16)
                            + sb[None, :],
                            mask=row_ok[:, None] & sbm[None, :],
                            other=0,
                        )
                    )
                    sc = tl.interleave(sc8, sc8)
                    sc = tl.interleave(sc, sc)
                    sc = tl.interleave(sc, sc)
                else:
                    s_off = (he * N + rows.to(tl.int64))[:, None] * (K // 16) \
                        + ((k + 2 * j[None, :]) // 16)
                    sc = _fp8e4m3_decode(
                        tl.load(
                            hot_scale_ptr + s_off,
                            mask=row_ok[:, None] & xm[None, :],
                            other=0,
                        )
                    )
                acc += tl.sum(sc * (lo * x_even[None, :] + hi * x_odd[None, :]), axis=1)
            else:  # HOT_MODE == 2
                kk = k + tl.arange(0, BLOCK_K)
                xm = kk < K
                xb = tl.load(x_ptr + s * K + kk, mask=xm, other=0.0).to(tl.float32)
                w_off = (he * N + rows.to(tl.int64))[:, None] * K + kk[None, :]
                wb = tl.load(
                    hot_packed_ptr + w_off,
                    mask=row_ok[:, None] & xm[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(wb * xb[None, :], axis=1)
        if HOT_MODE == 1:
            acc *= tl.load(hot_scale2_ptr + he)
        tl.store(out_base + rows, acc, mask=row_ok)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(sk, tl.cdiv(K, BLOCK_K), SPLIT_K):
            k = k0 * BLOCK_K
            g = tl.arange(0, BLOCK_K // _G)  # group index within the k block
            gi = tl.arange(0, _G)
            kk2 = k + g[:, None] * _G + gi[None, :]  # [BK//G, G] in-positions
            xm2 = kk2 < K
            x2 = tl.load(
                x_ptr + s * K + kk2, mask=xm2, other=0.0
            ).to(tl.float32)  # [BK//G, G]
            c_off = (ce * N + rows.to(tl.int64))[:, None] * (K // _G) + (k // _G) + g[None, :]
            c_mask = row_ok[:, None] & ((k + g[None, :] * _G) < K)
            code0 = tl.load(cold_codes0_ptr + c_off, mask=c_mask, other=0).to(tl.int32)
            code0 = code0 & (ENTRIES0 - 1)
            cb_off0 = code0.to(tl.int64)[:, :, None] * _G + gi[None, None, :]
            w3 = tl.load(
                cold_cb0_ptr + cb_off0, mask=c_mask[:, :, None], other=0.0
            ).to(tl.float32)  # [BN, BK//G, G]
            if NUM_BOOKS == 2:
                code1 = tl.load(cold_codes1_ptr + c_off, mask=c_mask, other=0).to(tl.int32)
                code1 = code1 & (ENTRIES1 - 1)
                cb_off1 = code1.to(tl.int64)[:, :, None] * _G + gi[None, None, :]
                w3 += tl.load(
                    cold_cb1_ptr + cb_off1, mask=c_mask[:, :, None], other=0.0
                ).to(tl.float32)
            acc += tl.sum(tl.sum(w3 * x2[None, :, :], axis=2), axis=1)
        cold_rs = tl.load(
            cold_scales_ptr + ce * N + rows.to(tl.int64), mask=row_ok, other=0.0
        ).to(tl.float32)
        tl.store(out_base + rows, acc * cold_rs, mask=row_ok)


@triton.jit
def _splitk_reduce_kernel(
    parts_ptr,  # [S, SPLIT_K, N] fp32 partials
    out_ptr,  # [S, N] fp32
    N,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = tl.cdiv(N, BLOCK)
    s = pid // nb
    off = (pid % nb) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(SPLIT_K):
        acc += tl.load(parts_ptr + (s * SPLIT_K + i) * N + off, mask=m, other=0.0)
    tl.store(out_ptr + s * N + off, acc, mask=m)


@triton.jit
def _grouped_hot_acc(
    x_ptr, xrows, m_ok, rw, row_ok, he,
    hot_packed_ptr, hot_scale_ptr,
    N, K, sk,
    HOT_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """K-loop accumulator for a hot-expert m-block (NVFP4 or bf16 decode)."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(sk, tl.cdiv(K, BLOCK_K), SPLIT_K):
        k = k0 * BLOCK_K
        kk = k + tl.arange(0, BLOCK_K)
        k_ok = kk < K
        xt = tl.load(
            x_ptr + xrows[:, None] * K + kk[None, :],
            mask=m_ok[:, None] & k_ok[None, :],
            other=0.0,
        )  # [BM, BK] bf16
        if HOT_MODE == 1:
            j = tl.arange(0, BLOCK_K // 2)
            jm = (k + 2 * j) < K
            pk = tl.load(
                hot_packed_ptr + (he * N + rw) * (K // 2) + (k // 2) + j[None, :],
                mask=row_ok[:, None] & jm[None, :],
                other=0,
            )
            lo = _nvfp4_mag(pk & 0xF) * tl.where((pk & 0x8) != 0, -1.0, 1.0)
            hi = _nvfp4_mag((pk >> 4) & 0xF) * tl.where((pk & 0x80) != 0, -1.0, 1.0)
            wv = tl.interleave(lo, hi)  # [BN, BK]
            sb = tl.arange(0, BLOCK_K // 16)
            sbm = (k + 16 * sb) < K
            sc = tl.load(
                hot_scale_ptr + (he * N + rw) * (K // 16) + (k // 16) + sb[None, :],
                mask=row_ok[:, None] & sbm[None, :],
                other=0,
            ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # [BN, BK//16] -> expand 16x along k via 4 self-interleaves
            sc = tl.interleave(sc, sc)
            sc = tl.interleave(sc, sc)
            sc = tl.interleave(sc, sc)
            sc = tl.interleave(sc, sc)
            w = (wv * sc).to(tl.bfloat16)
        else:
            w = tl.load(
                hot_packed_ptr + (he * N + rw) * K + kk[None, :],
                mask=row_ok[:, None] & k_ok[None, :],
                other=0.0,
            ).to(tl.bfloat16)
        acc = tl.dot(xt, tl.trans(w), acc)
    return acc


@triton.jit
def _grouped_cold_acc(
    x_ptr, xrows, m_ok, rw, row_ok, ce,
    cold_codes0_ptr, cold_cb0_ptr, cold_codes1_ptr, cold_cb1_ptr,
    N, K, sk,
    ENTRIES0: tl.constexpr,
    ENTRIES1: tl.constexpr,
    NUM_BOOKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """K-loop accumulator for a cold-expert m-block (AQLM codebook decode)."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    gi = tl.arange(0, _G)
    for k0 in range(sk, tl.cdiv(K, BLOCK_K), SPLIT_K):
        k = k0 * BLOCK_K
        kk = k + tl.arange(0, BLOCK_K)
        k_ok = kk < K
        xt = tl.load(
            x_ptr + xrows[:, None] * K + kk[None, :],
            mask=m_ok[:, None] & k_ok[None, :],
            other=0.0,
        )  # [BM, BK] bf16
        g = tl.arange(0, BLOCK_K // _G)
        gm = (k + g * _G) < K
        c_mask = row_ok[:, None] & gm[None, :]
        c_off = (ce * N + rw) * (K // _G) + (k // _G) + g[None, :]
        code0 = tl.load(cold_codes0_ptr + c_off, mask=c_mask, other=0).to(tl.int32)
        code0 = code0 & (ENTRIES0 - 1)
        w3 = tl.load(
            cold_cb0_ptr + code0.to(tl.int64)[:, :, None] * _G + gi[None, None, :],
            mask=c_mask[:, :, None],
            other=0.0,
        ).to(tl.float32)
        if NUM_BOOKS == 2:
            code1 = tl.load(cold_codes1_ptr + c_off, mask=c_mask, other=0).to(tl.int32)
            code1 = code1 & (ENTRIES1 - 1)
            w3 += tl.load(
                cold_cb1_ptr + code1.to(tl.int64)[:, :, None] * _G + gi[None, None, :],
                mask=c_mask[:, :, None],
                other=0.0,
            ).to(tl.float32)
        w = tl.reshape(w3, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
        acc = tl.dot(xt, tl.trans(w), acc)
    return acc


@triton.jit
def _hybrid_grouped_gemm_kernel(
    x_ptr,  # [*, K] bf16 input rows (token features, or sorted-slot acts)
    row_idx_ptr,  # [S] int32: x row for each sorted slot
    block_expert_ptr,  # [n_mblocks] int32 GLOBAL expert id per m-block
    block_slot0_ptr,  # [n_mblocks] int32 first sorted slot of the block
    block_mlen_ptr,  # [n_mblocks] int32 valid slots in the block (<= BLOCK_M)
    hot_lut_ptr,  # [n_experts] int32 local hot idx, -1 if cold
    cold_lut_ptr,  # [n_experts] int32 local cold idx, -1 if hot
    hot_packed_ptr, hot_scale_ptr, hot_scale2_ptr,
    cold_codes0_ptr, cold_cb0_ptr, cold_codes1_ptr, cold_cb1_ptr,
    cold_scales_ptr,
    out_ptr,  # [S, N] fp32, indexed by sorted slot
    N, K,
    ENTRIES0: tl.constexpr,
    ENTRIES1: tl.constexpr,
    NUM_BOOKS: tl.constexpr,
    HOT_MODE: tl.constexpr,  # 0 = none, 1 = nvfp4, 2 = bf16
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,  # INKLING_GROUPED_SPLITK (see module docs)
):
    """Grouped GEMM over expert-sorted slots: out[slot] = x[row(slot)] @ W_e^T
    for the expert e owning the slot's m-block. Same in-register NVFP4/AQLM
    decode as _hybrid_gemv_kernel but tiled with tl.dot for the prefill
    many-tokens-per-expert regime. Eager-only (grid is data-dependent).

    Hot-vs-cold is a SCALAR per m-block (one expert owns the block), so the
    two decode paths live behind a uniform runtime branch: a block executes
    only its own format's decode ALU. The previous masked-both-paths version
    paid the NVFP4 nibble/interleave ALU on every cold block (and the AQLM
    gather ALU on every hot block) just to discard it in a tl.where."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    sk = tl.program_id(2)  # always 0 when SPLIT_K == 1

    mlen = tl.load(block_mlen_ptr + pid_m)
    # Graph-safe decode maps launch one candidate block per slot and mark
    # non-head slots with mlen == 0; those programs no-op here so the grid
    # shape stays a function of S only (values, not shapes, carry the data
    # dependence). The eager prefill map never emits mlen == 0.
    if mlen != 0:
        e = tl.load(block_expert_ptr + pid_m).to(tl.int64)
        slot0 = tl.load(block_slot0_ptr + pid_m)
        hot_raw = tl.load(hot_lut_ptr + e)
        he = tl.maximum(hot_raw, 0).to(tl.int64)
        ce = tl.maximum(tl.load(cold_lut_ptr + e), 0).to(tl.int64)

        ms = tl.arange(0, BLOCK_M)
        m_ok = ms < mlen
        xrows = tl.load(row_idx_ptr + slot0 + ms, mask=m_ok, other=0).to(tl.int64)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = rows < N
        rw = (rows.to(tl.int64))[:, None]

        if (HOT_MODE != 0) and (hot_raw >= 0):
            acc = _grouped_hot_acc(
                x_ptr, xrows, m_ok, rw, row_ok, he,
                hot_packed_ptr, hot_scale_ptr,
                N, K, sk,
                HOT_MODE=HOT_MODE,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                SPLIT_K=SPLIT_K,
            )
            if HOT_MODE == 1:
                result = acc * tl.load(hot_scale2_ptr + he)
            else:
                result = acc
        else:
            acc = _grouped_cold_acc(
                x_ptr, xrows, m_ok, rw, row_ok, ce,
                cold_codes0_ptr, cold_cb0_ptr, cold_codes1_ptr, cold_cb1_ptr,
                N, K, sk,
                ENTRIES0=ENTRIES0, ENTRIES1=ENTRIES1, NUM_BOOKS=NUM_BOOKS,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                SPLIT_K=SPLIT_K,
            )
            cold_rs = tl.load(
                cold_scales_ptr + ce * N + rows.to(tl.int64), mask=row_ok, other=0.0
            ).to(tl.float32)
            result = acc * cold_rs[None, :]
        # SPLIT_K == 1 writes out[slot, n]; SPLIT_K > 1 writes fp32 partials
        # at parts[slot, sk, n], reduced by _splitk_reduce_kernel.
        out_off = ((slot0 + ms).to(tl.int64) * SPLIT_K + sk)[:, None] * N \
            + rows[None, :]
        tl.store(out_ptr + out_off, result, mask=m_ok[:, None] & row_ok[None, :])


def fused_hybrid_grouped_gemm(
    x: torch.Tensor,  # [R, K] bf16 input rows
    row_idx: torch.Tensor,  # [S] int32 x row per sorted slot
    block_expert: torch.Tensor,  # [n_mblocks] int32
    block_slot0: torch.Tensor,  # [n_mblocks] int32
    block_mlen: torch.Tensor,  # [n_mblocks] int32
    hot_lut: torch.Tensor,
    cold_lut: torch.Tensor,
    hot_mode: int,
    hot_packed: torch.Tensor | None,
    hot_scale: torch.Tensor | None,
    hot_scale2: torch.Tensor | None,
    cold_codes: list[torch.Tensor],
    cold_codebooks: list[torch.Tensor],
    cold_scales: torch.Tensor,
    S: int,
    N: int,
    K: int,
    block_m: int = 64,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 8,
    split_k: int = 1,
) -> torch.Tensor:
    """out[slot] = x[row_idx[slot]] @ W_expert(slot)^T -> fp32 [S, N].

    split_k > 1 (decode-regime callers only: the partials buffer is [S,
    split_k, N] fp32, prohibitive at prefill S) partitions each m-block's
    K-loop across grid axis 2 and reduces with _splitk_reduce_kernel; same
    latency-bound rationale and rounding-only numerics as the gemv split-K.
    """
    n_mblocks = block_expert.shape[0]
    out = torch.empty(S, N, dtype=torch.float32, device=x.device)
    num_books = len(cold_codes)
    assert num_books in (1, 2)
    c1 = cold_codes[1] if num_books == 2 else cold_codes[0]
    b1 = cold_codebooks[1] if num_books == 2 else cold_codebooks[0]
    if hot_mode == 1:
        hp, hs, hs2 = hot_packed, hot_scale.view(torch.uint8), hot_scale2
    elif hot_mode == 2:
        hp, hs, hs2 = hot_packed, cold_scales, cold_scales  # scale ptrs unused
    else:
        hp, hs, hs2 = cold_scales, cold_scales, cold_scales  # all unused
    split_k = min(split_k, triton.cdiv(K, block_k))
    kern_out = out
    if split_k > 1:
        kern_out = torch.empty(S, split_k, N, dtype=torch.float32, device=x.device)
    grid = (n_mblocks, triton.cdiv(N, block_n), split_k)
    _hybrid_grouped_gemm_kernel[grid](
        x, row_idx, block_expert, block_slot0, block_mlen,
        hot_lut, cold_lut,
        hp, hs, hs2,
        cold_codes[0], cold_codebooks[0], c1, b1, cold_scales,
        kern_out,
        N, K,
        ENTRIES0=cold_codebooks[0].shape[0],
        ENTRIES1=b1.shape[0],
        NUM_BOOKS=num_books,
        HOT_MODE=hot_mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )
    if split_k > 1:
        red_block = 256
        _splitk_reduce_kernel[(S * triton.cdiv(N, red_block),)](
            kern_out, out, N, SPLIT_K=split_k, BLOCK=red_block, num_warps=4
        )
    return out


def fused_hybrid_gemv(
    x: torch.Tensor,  # [S, K] bf16, one row per slot
    is_hot: torch.Tensor,  # [S] int32
    hot_idx: torch.Tensor,  # [S] int32/int64
    cold_idx: torch.Tensor,  # [S] int32/int64
    hot_mode: int,  # 0 none, 1 nvfp4, 2 bf16
    hot_packed: torch.Tensor | None,
    hot_scale: torch.Tensor | None,  # fp8_e4m3fn (viewed as uint8 here)
    hot_scale2: torch.Tensor | None,
    cold_codes: list[torch.Tensor],
    cold_codebooks: list[torch.Tensor],
    cold_scales: torch.Tensor,
    N: int,
    K: int,
    block_n: int = 8,
    block_k: int = 128,
    num_warps: int = 2,
) -> torch.Tensor:
    """y[s] = W_slot(s) @ x[s] -> fp32 [S, N], decoding quantized codes
    in-register. See module docstring for layout contracts."""
    S = x.shape[0]
    out = torch.empty(S, N, dtype=torch.float32, device=x.device)
    num_books = len(cold_codes)
    assert num_books in (1, 2)
    c1 = cold_codes[1] if num_books == 2 else cold_codes[0]
    b1 = cold_codebooks[1] if num_books == 2 else cold_codebooks[0]
    if hot_mode == 1:
        hp, hs, hs2 = hot_packed, hot_scale.view(torch.uint8), hot_scale2
    elif hot_mode == 2:
        hp, hs, hs2 = hot_packed, cold_scales, cold_scales  # scale ptrs unused
    else:
        hp, hs, hs2 = cold_scales, cold_scales, cold_scales  # all unused
    # Split-K: never more ways than K-chunks (empty partials are legal but
    # pure overhead). SPLIT_K is a constexpr keyed on (K, env), so the graph
    # per shape family is stable.
    split_k = min(_GEMV_SPLITK, triton.cdiv(K, block_k))
    kern_out = out
    if split_k > 1:
        kern_out = torch.empty(S, split_k, N, dtype=torch.float32, device=x.device)
    grid = (S * triton.cdiv(N, block_n) * split_k,)
    _hybrid_gemv_kernel[grid](
        x, is_hot, hot_idx, cold_idx,
        hp, hs, hs2,
        cold_codes[0], cold_codebooks[0], c1, b1, cold_scales,
        kern_out,
        N, K,
        ENTRIES0=cold_codebooks[0].shape[0],
        ENTRIES1=b1.shape[0],
        NUM_BOOKS=num_books,
        HOT_MODE=hot_mode,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SCALE_EXPAND=_GEMV_V2,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )
    if split_k > 1:
        red_block = 256
        _splitk_reduce_kernel[(S * triton.cdiv(N, red_block),)](
            kern_out, out, N, SPLIT_K=split_k, BLOCK=red_block, num_warps=4
        )
    return out
