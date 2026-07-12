# SPDX-License-Identifier: Apache-2.0
"""CPU reference implementation of the GLM-5.2 hybrid MoE decode kernels.

This module codifies the EXACT accumulation tree of the shipped
``aqlm_moe_v2.cu`` kernels (``aqlm_slot_gemv`` / ``nvfp4_slot_gemv`` /
``HybridMatVecMoE`` and the two prefill dequant kernels) as an executable,
numpy-only specification.  It is the "documented tree" that Tier-1 tests
compare the GPU kernels against with a maxdiff==0 gate, and the golden
property that makes "order-preserving" claims (e.g. the w2 lane-occupancy
repack) mechanically checkable.

===========================================================================
THE ACCUMULATION TREE (normative)
===========================================================================

AQLM path (aqlm_slot_gemv, one warp per (slot,row)):
  - The K axis is processed in int4-sized "code groups": one int4 holds 8
    uint16 codes; each code indexes an 8-wide fp16 codebook entry, so one
    int4 covers 64 K-values.  a_gl_stride = K/64 int4s per row.
  - Lane l (0..31) consumes int4 indices l, l+32, l+64, ... in ascending
    order (one per outer iteration).
  - Per int4, codes u = 0..7 are processed in ascending order (AQLM_MLP
    batching only reorders LOADS, never arithmetic).  Per code u:
        wsum[0..7]  = book0[code0]            (fp16 values)
        wsum        = __hadd2(wsum, book1[code1])       if BOOKS == 2
        res2 (half2) = 0
        for j = 0..3:  res2 = __hfma2(wsum[2j:2j+2], b[2j:2j+2], res2)
          -> res2.x accumulates even elements, res2.y odd elements,
             in half precision with FUSED rounding per step.
        res (fp32) += ( float(res2.x) + float(res2.y) )
          -> one fp32 add to form the pair-sum, one fp32 add into res.
  - Cross-lane reduction: shfl_down offsets 16, 8, 4, 2, 1 in fp32:
        res[l] += res[l + off]     (src lane >= 32 -> lane's own value)
    Lane 0 holds the row result.
  - Output: C[row] = __float2half( res * float(scales[expert, row]) )
    (one fp32 multiply, then round to half).

NVFP4 path (nvfp4_slot_gemv):
  - K in uint4 chunks of 32 fp4 values (16 bytes); byte i holds values
    (2i, 2i+1) as (low nibble, high nibble).  a_gl_stride = K/32.
  - Lane l consumes chunk indices l, l+32, ... ascending.
  - Per chunk, two 16-value scale blocks (bs.x for bytes 0..7,
    bs.y for bytes 8..15):
        acc0 (half2) = 0;  for i = 0..7:  acc0 = __hfma2(w[i], b[i], acc0)
        acc1 (half2) = 0;  for i = 8..15: acc1 = __hfma2(w[i], b[i], acc1)
        res (fp32) += fp8(bs.x) * (float(acc0.x) + float(acc0.y))
                    + fp8(bs.y) * (float(acc1.x) + float(acc1.y))
    The fp32 evaluation of that two-term combine is subject to nvcc FMA
    contraction (-fmad=true); the exact codegen is pinned by calibration
    against the shipped kernel -- see FMA_MODES / calibrate_fma_mode().
    All other steps are intrinsic-explicit and unambiguous.
  - Same shfl reduction; output
        C[row] = __float2half( res * scale2[e*s2n + (row*s2n)//prob_m] ).

Hybrid kernel: per slot, exactly one of aqlm_ids/nv_ids >= 0 selects the
path; both < 0 writes half(+0.0).

Prefill dequant kernels:
  - CodeKx16DequantMoE: out = __hmul2( hadd2-sum of books , half2(scale) )
  - NvFp4DequantMoE:    s = fp8(bs) * gscale        (fp32 multiply)
                        out = __floats2half2_rn( lut[nibble] * s )  (fp32)

===========================================================================
"""

from __future__ import annotations

import numpy as np

from .fp_codecs import (
    F16,
    F32,
    F64,
    f32_to_half,
    floats2half2_rn,
    fp8_e4m3_to_f32,
    FP4_E2M1_TABLE,
    half_add,
    half_fma,
    half_mul,
)

# Candidate codegens for the NVFP4 fp32 two-block scale combine
# ``res += s0*A + s1*B`` (A = acc0.x+acc0.y, B = acc1.x+acc1.y):
#   plain     : p0 = s0*A; p1 = s1*B; res += (p0 + p1)
#   fma_b     : p0 = s0*A; res += fma(s1, B, p0)
#   fma_a     : p1 = s1*B; res += fma(s0, A, p1)
#   fma_chain : res = fma(s1, B, fma(s0, A, res))
FMA_MODES = ("plain", "fma_b", "fma_a", "fma_chain")


def _f32_fma(a, b, c):
    """Single-rounding fp32 fma (exact in float64, one rounding)."""
    return (np.asarray(a, F64) * np.asarray(b, F64) +
            np.asarray(c, F64)).astype(F32)


def _shfl_down_reduce(res: np.ndarray) -> np.ndarray:
    """The exact shfl_down 16/8/4/2/1 fp32 tree.  res: [32, ...] fp32.

    __shfl_down_sync returns the caller's own value when the source lane
    is out of range; we replicate that even though only lane 0 matters.
    """
    r = res.astype(F32, copy=True)
    for off in (16, 8, 4, 2, 1):
        src = np.arange(32) + off
        src = np.where(src < 32, src, np.arange(32))
        r = (r + r[src]).astype(F32)
    return r[0]


# ---------------------------------------------------------------------------
# AQLM slot gemv reference
# ---------------------------------------------------------------------------


def aqlm_slot_gemv_ref(
    codes: np.ndarray,      # [E, books, M, K/8] int16 (uint16 payload)
    codebooks: np.ndarray,  # [books, 65536, 8] float16
    scales: np.ndarray,     # [E, M] float16
    expert: int,
    b_vec: np.ndarray,      # [K] float16 activations for this slot
) -> np.ndarray:            # [M] float16
    books, m = codes.shape[1], codes.shape[2]
    k = codes.shape[3] * 8
    k64 = k // 64
    codes_u = codes[expert].view(np.uint16)  # [books, M, K/8]
    res = np.zeros((32, m), dtype=F32)

    for i4 in range(k64):
        lane = i4 % 32
        for u in range(8):
            g = i4 * 8 + u
            wsum = codebooks[0][codes_u[0, :, g]]          # [M, 8] f16
            if books == 2:
                wsum = half_add(wsum, codebooks[1][codes_u[1, :, g]])
            res2 = np.zeros((m, 2), dtype=F16)
            base = 64 * i4 + 8 * u
            for j in range(4):
                bb = b_vec[base + 2 * j: base + 2 * j + 2]  # [2] f16
                res2 = half_fma(wsum[:, 2 * j: 2 * j + 2], bb[None, :], res2)
            pair = (res2[:, 0].astype(F32) + res2[:, 1].astype(F32)).astype(F32)
            res[lane] = (res[lane] + pair).astype(F32)

    lane0 = _shfl_down_reduce(res)                         # [M] f32
    s = scales[expert].astype(F32)                         # float(half) exact
    return f32_to_half((lane0 * s).astype(F32))


# ---------------------------------------------------------------------------
# NVFP4 slot gemv reference
# ---------------------------------------------------------------------------


def nvfp4_slot_gemv_ref(
    packed: np.ndarray,   # [E, M, K/2] uint8
    bscale: np.ndarray,   # [E, M, K/16] uint8 (fp8 e4m3 block scales)
    scale2: np.ndarray,   # [E, s2n] float32
    expert: int,
    b_vec: np.ndarray,    # [K] float16
    fma_mode: str = "plain",
) -> np.ndarray:          # [M] float16
    assert fma_mode in FMA_MODES, fma_mode
    m, k = packed.shape[1], packed.shape[2] * 2
    s2n = scale2.shape[1]
    stride = k // 32
    pk = packed[expert]   # [M, K/2]
    bs = bscale[expert]   # [M, K/16]
    res = np.zeros((32, m), dtype=F32)

    for c in range(stride):
        lane = c % 32
        chunk = pk[:, c * 16:(c + 1) * 16]                    # [M, 16] bytes
        lo = FP4_E2M1_TABLE[chunk & 0xF]                      # [M, 16] f16
        hi = FP4_E2M1_TABLE[chunk >> 4]
        s0 = fp8_e4m3_to_f32(bs[:, 2 * c])                    # [M] f32
        s1 = fp8_e4m3_to_f32(bs[:, 2 * c + 1])

        acc0 = np.zeros((m, 2), dtype=F16)
        acc1 = np.zeros((m, 2), dtype=F16)
        for i in range(8):
            bb = b_vec[32 * c + 2 * i: 32 * c + 2 * i + 2]
            w = np.stack([lo[:, i], hi[:, i]], axis=1)        # [M, 2] f16
            acc0 = half_fma(w, bb[None, :], acc0)
        for i in range(8, 16):
            bb = b_vec[32 * c + 2 * i: 32 * c + 2 * i + 2]
            w = np.stack([lo[:, i], hi[:, i]], axis=1)
            acc1 = half_fma(w, bb[None, :], acc1)

        a = (acc0[:, 0].astype(F32) + acc0[:, 1].astype(F32)).astype(F32)
        b = (acc1[:, 0].astype(F32) + acc1[:, 1].astype(F32)).astype(F32)

        if fma_mode == "plain":
            t = ((s0 * a).astype(F32) + (s1 * b).astype(F32)).astype(F32)
            res[lane] = (res[lane] + t).astype(F32)
        elif fma_mode == "fma_b":
            t = _f32_fma(s1, b, (s0 * a).astype(F32))
            res[lane] = (res[lane] + t).astype(F32)
        elif fma_mode == "fma_a":
            t = _f32_fma(s0, a, (s1 * b).astype(F32))
            res[lane] = (res[lane] + t).astype(F32)
        else:  # fma_chain
            res[lane] = _f32_fma(s1, b, _f32_fma(s0, a, res[lane]))

    lane0 = _shfl_down_reduce(res)
    row = np.arange(m, dtype=np.int64)
    g = scale2[expert][(row * s2n) // m].astype(F32)
    return f32_to_half((lane0 * g).astype(F32))


# ---------------------------------------------------------------------------
# Fused hybrid gemv reference
# ---------------------------------------------------------------------------


def hybrid_moe_gemv_ref(
    x: np.ndarray,          # [S, K] float16
    codes: np.ndarray,      # [nc, books, M, K/8] int16
    codebooks: np.ndarray,  # [books, 65536, 8] float16
    scales: np.ndarray,     # [nc, M] float16
    aqlm_ids: np.ndarray,   # [S] int32 (-1 = not AQLM)
    packed: np.ndarray,     # [na, M, K/2] uint8 (may be empty)
    bscale: np.ndarray,     # [na, M, K/16] uint8
    scale2: np.ndarray,     # [na, s2n] float32
    nv_ids: np.ndarray,     # [S] int32 (-1 = not NVFP4)
    fma_mode: str = "plain",
) -> np.ndarray:            # [S, M] float16
    s = x.shape[0]
    m = codes.shape[2]
    out = np.zeros((s, m), dtype=F16)  # masked slots -> half(+0.0)
    for slot in range(s):
        a_id = int(aqlm_ids[slot])
        n_id = int(nv_ids[slot])
        if a_id >= 0:
            out[slot] = aqlm_slot_gemv_ref(codes, codebooks, scales, a_id,
                                           x[slot])
        elif n_id >= 0:
            out[slot] = nvfp4_slot_gemv_ref(packed, bscale, scale2, n_id,
                                            x[slot], fma_mode=fma_mode)
        # else: stays +0.0
    return out


# ---------------------------------------------------------------------------
# Prefill dequant references
# ---------------------------------------------------------------------------


def aqlm_dequant_ref(
    codes: np.ndarray,      # [E, books, M, K/8] int16
    codebooks: np.ndarray,  # [books, 65536, 8] float16
    scales: np.ndarray,     # [E, M] float16
    expert_list: np.ndarray,
) -> np.ndarray:            # [G, M, K] float16
    books, m, k8 = codes.shape[1], codes.shape[2], codes.shape[3]
    out = np.zeros((len(expert_list), m, k8 * 8), dtype=F16)
    for gi, e in enumerate(expert_list):
        cu = codes[int(e)].view(np.uint16)          # [books, M, K/8]
        w = codebooks[0][cu[0]]                     # [M, K/8, 8] f16
        for b in range(1, books):
            w = half_add(w, codebooks[b][cu[b]])
        # scale path in-kernel: s = float(half scale); s2 = half2(s)
        # -> float->half of a half is the identity, so this is a plain
        #    correctly-rounded half multiply.
        sc = scales[int(e)].astype(F16)             # [M]
        out[gi] = half_mul(w, sc[:, None, None]).reshape(m, k8 * 8)
    return out


def nvfp4_dequant_ref(
    packed: np.ndarray,     # [E, M, K/2] uint8
    bscale: np.ndarray,     # [E, M, K/16] uint8
    scale2: np.ndarray,     # [E, s2n] float32
    expert_list: np.ndarray,
) -> np.ndarray:            # [G, M, K] float16
    e_, m, k2 = packed.shape
    k = k2 * 2
    s2n = scale2.shape[1]
    out = np.zeros((len(expert_list), m, k), dtype=F16)
    row = np.arange(m, dtype=np.int64)
    for gi, e in enumerate(expert_list):
        e = int(e)
        g = scale2[e][(row * s2n) // m].astype(F32)     # [M]
        # per 16-byte chunk: s[j] = fp8(bs.j) * gscale   (fp32)
        bsf = fp8_e4m3_to_f32(bscale[e])                # [M, K/16] f32
        s = (bsf * g[:, None]).astype(F32)              # [M, K/16]
        lo = FP4_E2M1_TABLE[packed[e] & 0xF].astype(F32)   # [M, K/2]
        hi = FP4_E2M1_TABLE[packed[e] >> 4].astype(F32)
        vals = np.empty((m, k), dtype=F32)
        vals[:, 0::2] = lo
        vals[:, 1::2] = hi
        sblock = np.repeat(s, 16, axis=1)               # [M, K]
        out[gi] = floats2half2_rn((vals * sblock).astype(F32))
    return out


# ---------------------------------------------------------------------------
# Calibration of the one compiler-discretion point
# ---------------------------------------------------------------------------


def calibrate_fma_mode(run_kernel, case) -> str:
    """Determine which FMA_MODES candidate matches the shipped kernel.

    ``run_kernel(case) -> np.ndarray[S, M] f16`` runs the GPU kernel on a
    small all-NVFP4 case; ``case`` carries the same arrays the reference
    needs.  Exactly one candidate must match bit-exactly; anything else
    means the kernel numerics changed and is a hard failure.
    """
    from .fp_codecs import canon_half_bits

    got = run_kernel(case)
    matches = []
    for mode in FMA_MODES:
        ref = hybrid_moe_gemv_ref(
            case["x"], case["codes"], case["codebooks"], case["scales"],
            case["aqlm_ids"], case["packed"], case["bscale"], case["scale2"],
            case["nv_ids"], fma_mode=mode)
        if np.array_equal(canon_half_bits(ref), canon_half_bits(got)):
            matches.append(mode)
    if len(matches) != 1:
        raise AssertionError(
            f"NVFP4 fp32-combine calibration failed: candidates matching "
            f"the shipped kernel = {matches!r} (expected exactly one). "
            "If empty, the kernel's accumulation tree CHANGED — "
            "stop and investigate before trusting any variant.")
    return matches[0]
