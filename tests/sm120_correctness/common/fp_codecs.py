# SPDX-License-Identifier: Apache-2.0
"""Exact floating-point building blocks for the CPU reference kernels.

Everything here is numpy-only (no torch, no CUDA) so the CPU tier can run
on any machine.

Bit-exactness argument (documented, relied on by tier1):

* ``half``x``half`` products have <= 22 significant bits and ``half``
  addends <= 11; any half a*b+c evaluated in float64 is EXACT (the exact
  value needs < 53 mantissa bits across the full half exponent range,
  subnormals included).  A single float64->float16 rounding therefore
  reproduces CUDA's fused ``__hfma2`` / ``__hadd2`` / ``__hmul2``
  (round-to-nearest-even) with NO double-rounding hazard.
* numpy's float64->float16 astype performs a single correctly-rounded
  conversion (npy_doublebits_to_halfbits), matching cvt.rn.f16.f64.
* fp32 accumulation steps are emulated with np.float32 arithmetic, which
  is IEEE binary32 RNE exactly like the GPU's non-contracted add.f32 /
  mul.f32 (the kernels are compiled -O3 without fast-math; contraction
  ambiguities are handled explicitly by the caller, see
  moe_reference.FMA_MODES).

NaN policy: CUDA and numpy may produce different NaN payloads.  Bitwise
comparisons in the suite canonicalize every NaN to 0x7E00 (half) /
0x7FC00000 (fp32) first, and separately assert NaN-ness matches.
"""

from __future__ import annotations

import numpy as np

F16 = np.float16
F32 = np.float32
F64 = np.float64

FLT_MIN = np.float32(1.1754943508222875e-38)  # smallest normal fp32
FP8_E4M3_MAX = np.float32(448.0)

# ---------------------------------------------------------------------------
# half arithmetic with CUDA-intrinsic semantics
# ---------------------------------------------------------------------------


def half_fma(a, b, c):
    """__hfma2 lane: round_to_half(a*b + c) computed exactly."""
    return (np.asarray(a, F64) * np.asarray(b, F64) +
            np.asarray(c, F64)).astype(F16)


def half_add(a, b):
    """__hadd2 lane: correctly-rounded half addition."""
    return (np.asarray(a, F64) + np.asarray(b, F64)).astype(F16)


def half_mul(a, b):
    """__hmul2 lane: correctly-rounded half multiplication."""
    return (np.asarray(a, F64) * np.asarray(b, F64)).astype(F16)


def f32_to_half(x):
    """__float2half (cvt.rn.f16.f32)."""
    return np.asarray(x, F32).astype(F16)


def floats2half2_rn(x):
    """__floats2half2_rn lane (same rounding as __float2half)."""
    return np.asarray(x, F32).astype(F16)


# ---------------------------------------------------------------------------
# fp8 e4m3 (finite-NaN variant used by CUDA __nv_fp8_e4m3)
# ---------------------------------------------------------------------------


def _build_e4m3_decode_table() -> np.ndarray:
    tab = np.zeros(256, dtype=F32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exp = (byte >> 3) & 0xF
        man = byte & 0x7
        if exp == 0xF and man == 0x7:
            val = np.nan  # e4m3fn: 0x7F/0xFF is NaN, there is no inf
        elif exp == 0:
            val = sign * (man / 8.0) * 2.0**-6  # subnormal
        else:
            val = sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        tab[byte] = np.float32(val)
    return tab


E4M3_DECODE = _build_e4m3_decode_table()


def fp8_e4m3_to_f32(byte):
    """fp8_e4m3_to_float in aqlm_moe_v2.cu / cache_kernels.cu (exact)."""
    return E4M3_DECODE[np.asarray(byte, dtype=np.uint8)]


def f32_to_fp8_e4m3_satfinite(x) -> np.ndarray:
    """__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3).

    Round-to-nearest-even; overflow saturates to +-448 (0x7E/0xFE);
    NaN encodes to 0x7F (sign preserved -> 0xFF for negative NaN).
    """
    x = np.asarray(x, F32)
    out = np.zeros(x.shape, dtype=np.uint8)
    flat = x.ravel()
    res = out.ravel()
    # Candidate magnitudes for all 256 codes, decode-table driven RNE:
    # for each input, find the nearest representable; ties to even code
    # (even mantissa bit).  Implemented scalar-wise for clarity; test
    # inputs are small.
    finite_codes = [b for b in range(0x80) if not np.isnan(E4M3_DECODE[b])]
    finite_vals = np.array([E4M3_DECODE[b] for b in finite_codes], dtype=F64)
    for i, v in enumerate(flat):
        if np.isnan(v):
            res[i] = 0x7F | (0x80 if np.signbit(v) else 0)
            continue
        sign = 0x80 if np.signbit(v) else 0
        mag = abs(np.float64(v))
        if np.isinf(mag) or mag > 448.0:
            res[i] = 0x7E | sign  # SATFINITE clamps to max finite
            continue
        # nearest representable magnitude, ties-to-even (lower code has
        # even LSB when codes are consecutive)
        idx = int(np.searchsorted(finite_vals, mag))
        if idx == 0:
            best = finite_codes[0]
        elif idx >= len(finite_vals):
            best = finite_codes[-1]
        else:
            lo, hi = idx - 1, idx
            dlo = mag - finite_vals[lo]
            dhi = finite_vals[hi] - mag
            if dlo < dhi:
                best = finite_codes[lo]
            elif dhi < dlo:
                best = finite_codes[hi]
            else:  # tie -> even mantissa (LSB 0)
                best = finite_codes[lo] if finite_codes[lo] % 2 == 0 \
                    else finite_codes[hi]
        res[i] = best | sign
    return out


# ---------------------------------------------------------------------------
# fp4 e2m1 (NVFP4 weight nibbles)
# ---------------------------------------------------------------------------

# Matches kFp4Lut in aqlm_moe_v2.cu and cvt.rn.f16x2.e2m1x2 exactly.
FP4_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=F16,
)


def fp4x2_to_half2(byte):
    """fp4x2_to_half2: low nibble -> element 0, high nibble -> element 1."""
    byte = np.asarray(byte, dtype=np.uint8)
    return FP4_E2M1_TABLE[byte & 0xF], FP4_E2M1_TABLE[byte >> 4]


# ---------------------------------------------------------------------------
# bitwise comparison helpers
# ---------------------------------------------------------------------------


def canon_half_bits(x: np.ndarray) -> np.ndarray:
    """uint16 view of a float16 array with all NaNs canonicalized."""
    x = np.ascontiguousarray(x, dtype=F16)
    bits = x.view(np.uint16).copy()
    bits[np.isnan(x)] = 0x7E00
    return bits


def assert_half_bitexact(ref: np.ndarray, test: np.ndarray, what: str = ""):
    rb, tb = canon_half_bits(ref), canon_half_bits(test)
    if np.array_equal(rb, tb):
        return
    bad = np.argwhere(rb != tb)
    i = tuple(bad[0])
    raise AssertionError(
        f"{what}: {bad.shape[0]}/{rb.size} elements differ; first at "
        f"{i}: ref=0x{rb[i]:04x} ({np.float16(ref[i])}) "
        f"test=0x{tb[i]:04x} ({np.float16(test[i])})")


def max_ulp_diff_half(ref: np.ndarray, test: np.ndarray) -> int:
    """Diagnostic: max distance in half-code space (monotone mapping)."""
    def key(bits):
        b = bits.astype(np.int32)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), 0x8000 + b)
    return int(np.abs(key(canon_half_bits(ref)) -
                      key(canon_half_bits(test))).max(initial=0))
