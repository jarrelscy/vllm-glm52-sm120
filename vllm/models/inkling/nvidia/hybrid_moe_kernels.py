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

import torch
import triton
import triton.language as tl

_G = tl.constexpr(8)  # AQLM group size along the in (K) dim

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
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    s = pid // n_blocks
    nb = pid % n_blocks

    rows = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    row_ok = rows < N

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
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k = k0 * BLOCK_K
            if HOT_MODE == 1:
                # bytes j cover in-positions (k + 2j, k + 2j + 1)
                j = tl.arange(0, BLOCK_K // 2)
                xm = (k + 2 * j) < K
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
        tl.store(out_ptr + s * N + rows, acc, mask=row_ok)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
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
        tl.store(out_ptr + s * N + rows, acc * cold_rs, mask=row_ok)


@triton.jit
def _grouped_hot_acc(
    x_ptr, xrows, m_ok, rw, row_ok, he,
    hot_packed_ptr, hot_scale_ptr,
    N, K,
    HOT_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """K-loop accumulator for a hot-expert m-block (NVFP4 or bf16 decode)."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
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
    N, K,
    ENTRIES0: tl.constexpr,
    ENTRIES1: tl.constexpr,
    NUM_BOOKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """K-loop accumulator for a cold-expert m-block (AQLM codebook decode)."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    gi = tl.arange(0, _G)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
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
                N, K,
                HOT_MODE=HOT_MODE,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            if HOT_MODE == 1:
                result = acc * tl.load(hot_scale2_ptr + he)
            else:
                result = acc
        else:
            acc = _grouped_cold_acc(
                x_ptr, xrows, m_ok, rw, row_ok, ce,
                cold_codes0_ptr, cold_cb0_ptr, cold_codes1_ptr, cold_cb1_ptr,
                N, K,
                ENTRIES0=ENTRIES0, ENTRIES1=ENTRIES1, NUM_BOOKS=NUM_BOOKS,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            cold_rs = tl.load(
                cold_scales_ptr + ce * N + rows.to(tl.int64), mask=row_ok, other=0.0
            ).to(tl.float32)
            result = acc * cold_rs[None, :]
        out_off = (slot0 + ms).to(tl.int64)[:, None] * N + rows[None, :]
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
) -> torch.Tensor:
    """out[slot] = x[row_idx[slot]] @ W_expert(slot)^T -> fp32 [S, N]."""
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
    grid = (n_mblocks, triton.cdiv(N, block_n))
    _hybrid_grouped_gemm_kernel[grid](
        x, row_idx, block_expert, block_slot0, block_mlen,
        hot_lut, cold_lut,
        hp, hs, hs2,
        cold_codes[0], cold_codebooks[0], c1, b1, cold_scales,
        out,
        N, K,
        ENTRIES0=cold_codebooks[0].shape[0],
        ENTRIES1=b1.shape[0],
        NUM_BOOKS=num_books,
        HOT_MODE=hot_mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
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
    grid = (S * triton.cdiv(N, block_n),)
    _hybrid_gemv_kernel[grid](
        x, is_hot, hot_idx, cold_idx,
        hp, hs, hs2,
        cold_codes[0], cold_codebooks[0], c1, b1, cold_scales,
        out,
        N, K,
        ENTRIES0=cold_codebooks[0].shape[0],
        ENTRIES1=b1.shape[0],
        NUM_BOOKS=num_books,
        HOT_MODE=hot_mode,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out
