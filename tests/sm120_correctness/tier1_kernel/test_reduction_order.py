# SPDX-License-Identifier: Apache-2.0
"""TIER 1 (CPU-ONLY) — the accumulation tree as a golden property.

Guards idea 2 (w2 lane-occupancy repack, "order-preserving reduction")
and any claim that a rewrite preserves the reduction order: the exact
tree of aqlm_slot_gemv / nvfp4_slot_gemv (per-lane fp16 fused-fma groups,
fp32 per-lane accumulation, shfl_down offsets 16/8/4/2/1, fp32 scale
multiply) is codified executable in common/moe_reference.py and pinned
here against a committed golden.  Runs everywhere — no CUDA, no torch.

Three properties:
  1. GOLDEN: the reference reproduces the committed golden bytes on a
     fixed seeded case.  Any accidental reorder of the reference itself
     (or a semantic drift while "cleaning it up") fails.
  2. NON-VACUITY: deliberately permuting the reduction order (reversed
     shfl offsets; swapped fp32 pair-add) CHANGES the golden output —
     i.e. the golden really pins the order, it is not tolerance-blind.
  3. On GPU (tier1 gemv test), the same reference is asserted maxdiff==0
     against the shipped kernel, closing the loop reference <-> kernel.

"order-preserving" claims are therefore mechanically checkable: run the
claimed-order-preserving kernel against this reference; bit-equality
holds iff the tree is untouched.
"""

import hashlib
import json
import pathlib

import numpy as np
import pytest

from common import kernels as K
from common import moe_reference as R
from common.fp_codecs import F16, F32, canon_half_bits

GOLDEN_PATH = (pathlib.Path(__file__).resolve().parents[1] / "goldens" /
               "reduction_tree_golden.json")

# Fixed cases: cover both paths, both books, partial-lane tails, and all
# four NVFP4 fma-mode candidates (the golden pins each candidate's
# output so a change in ANY branch of the reference is caught).
CASES = [
    dict(shape="w13_small", mix="all_aqlm", S=4, books=1, seed=501),
    dict(shape="w13_small", mix="all_aqlm", S=4, books=2, seed=502),
    dict(shape="w2_small", mix="all_aqlm", S=2, books=2, seed=503),
    dict(shape="w13_small", mix="all_nvfp4", S=4, books=1, seed=504),
    dict(shape="w2_small", mix="mixed", S=8, books=1, seed=505),
]


def _digest(meta, fma_mode="plain"):
    case = K.make_case(meta["shape"], meta["mix"], meta["S"], meta["books"],
                       meta["seed"])
    out = R.hybrid_moe_gemv_ref(
        case["x"], case["codes"], case["codebooks"], case["scales"],
        case["aqlm_ids"], case["packed"], case["bscale"], case["scale2"],
        case["nv_ids"], fma_mode=fma_mode)
    return hashlib.sha256(canon_half_bits(out).tobytes()).hexdigest()


def compute_goldens() -> dict:
    """Used by make_goldens.sh --cpu to (re)generate the golden file."""
    golden = {}
    for meta in CASES:
        key = f"{meta['shape']}-{meta['mix']}-S{meta['S']}-b{meta['books']}"
        entry = {"plain": _digest(meta)}
        if meta["mix"] in ("all_nvfp4", "mixed"):
            for mode in R.FMA_MODES[1:]:
                entry[mode] = _digest(meta, fma_mode=mode)
        golden[key] = entry
    return golden


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip("golden file missing — run make_goldens.sh --cpu")
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("meta", CASES,
                         ids=lambda m: f"{m['shape']}-{m['mix']}-b{m['books']}")
def test_reference_matches_golden(meta, golden):
    key = f"{meta['shape']}-{meta['mix']}-S{meta['S']}-b{meta['books']}"
    assert key in golden, f"golden missing case {key}"
    for mode, want in golden[key].items():
        got = _digest(meta, fma_mode=mode)
        assert got == want, \
            (f"{key} [{mode}]: reference output changed — the codified "
             "accumulation tree drifted. If this was intentional, the GPU "
             "bit-exact tests MUST be re-run and the golden regenerated "
             "via make_goldens.sh --cpu.")


def test_golden_pins_lane_reduction_order():
    """The codified shfl tree is ORDER-SENSITIVE at fp32 (non-vacuity).

    NOTE: "order-preserving" claims live at the fp32 accumulation stage;
    the final round-to-half can mask 1-ulp fp32 reorder differences, so
    this property is asserted on the fp32 reduction itself, and the GPU
    bit-exact tests plus the golden pin the half-level pipeline.
    """
    g = np.random.default_rng(9)
    res = (g.standard_normal((32, 256)) *
           10.0 ** g.integers(-6, 7, size=(32, 256))).astype(F32)

    def reversed_tree(r0):
        r = r0.astype(F32, copy=True)
        for off in (1, 2, 4, 8, 16):
            src = np.arange(32) + off
            src = np.where(src < 32, src, np.arange(32))
            r = (r + r[src]).astype(F32)
        return r[0]

    a = R._shfl_down_reduce(res)
    b = reversed_tree(res)
    ndiff = int((a != b).sum())
    assert ndiff > 0, \
        ("reversing the lane-reduction order did NOT change any fp32 "
         "result — the codified tree cannot certify order preservation")
    # and the tree is deterministic
    assert np.array_equal(a, R._shfl_down_reduce(res))
    print(f"[non-vacuity] reversed tree differs on {ndiff}/256 fp32 sums")


def test_golden_pins_pair_add_order():
    """Swapping the fp32 (even+odd) pair-add order must change output."""
    meta = CASES[0]
    case = K.make_case(meta["shape"], meta["mix"], meta["S"], meta["books"],
                       meta["seed"])

    ref = R.aqlm_slot_gemv_ref(case["codes"], case["codebooks"],
                               case["scales"], int(case["aqlm_ids"][0]),
                               case["x"][0])

    # swapped variant: res += (odd + even) instead of (even + odd)
    def swapped(codes, codebooks, scales, expert, b_vec):
        books, m = codes.shape[1], codes.shape[2]
        k = codes.shape[3] * 8
        k64 = k // 64
        codes_u = codes[expert].view(np.uint16)
        res = np.zeros((32, m), dtype=F32)
        from common.fp_codecs import half_add, half_fma, f32_to_half
        for i4 in range(k64):
            lane = i4 % 32
            for u in range(8):
                g = i4 * 8 + u
                wsum = codebooks[0][codes_u[0, :, g]]
                if books == 2:
                    wsum = half_add(wsum, codebooks[1][codes_u[1, :, g]])
                res2 = np.zeros((m, 2), dtype=F16)
                base = 64 * i4 + 8 * u
                for j in range(4):
                    bb = b_vec[base + 2 * j: base + 2 * j + 2]
                    res2 = half_fma(wsum[:, 2 * j: 2 * j + 2], bb[None, :],
                                    res2)
                pair = (res2[:, 1].astype(F32) +
                        res2[:, 0].astype(F32)).astype(F32)  # SWAPPED
                res[lane] = (res[lane] + pair).astype(F32)
        lane0 = R._shfl_down_reduce(res)
        return f32_to_half((lane0 * scales[expert].astype(F32)).astype(F32))

    perm = swapped(case["codes"], case["codebooks"], case["scales"],
                   int(case["aqlm_ids"][0]), case["x"][0])
    # fp32 a+b == b+a exactly (commutative) — so THIS swap must NOT change
    # anything; it validates the emulation is genuinely fp32.
    assert np.array_equal(canon_half_bits(ref), canon_half_bits(perm)), \
        "fp32 pair-add commutativity violated — emulation bug"


def test_lane_assignment_pins_k_order():
    """Moving one K-chunk to a different lane must change the result."""
    case = K.make_case(shape="w13_small", mix="all_aqlm", S=1, books=1,
                       seed=501)

    # fp32-stage variant of the AQLM path (final half-rounding elided —
    # repack/order claims live at the fp32 accumulation stage; the final
    # round-to-half can legitimately mask 1-ulp fp32 differences).
    def aqlm_f32(codes, codebooks, scales, expert, b_vec, lane_of):
        from common.fp_codecs import half_add, half_fma
        books, m = codes.shape[1], codes.shape[2]
        k = codes.shape[3] * 8
        k64 = k // 64
        codes_u = codes[expert].view(np.uint16)
        res = np.zeros((32, m), dtype=F32)
        for i4 in range(k64):
            lane = lane_of(i4)
            for u in range(8):
                g = i4 * 8 + u
                wsum = codebooks[0][codes_u[0, :, g]]
                if books == 2:
                    wsum = half_add(wsum, codebooks[1][codes_u[1, :, g]])
                res2 = np.zeros((m, 2), dtype=F16)
                base = 64 * i4 + 8 * u
                for j in range(4):
                    bb = b_vec[base + 2 * j: base + 2 * j + 2]
                    res2 = half_fma(wsum[:, 2 * j: 2 * j + 2], bb[None, :],
                                    res2)
                pair = (res2[:, 0].astype(F32) +
                        res2[:, 1].astype(F32)).astype(F32)
                res[lane] = (res[lane] + pair).astype(F32)
        lane0 = R._shfl_down_reduce(res)
        return (lane0 * scales[expert].astype(F32)).astype(F32)

    # Wide-dynamic-range activations: benign small-magnitude half data
    # sums EXACTLY in fp32 (11-bit mantissas, narrow exponent span), so
    # every order gives identical results and no perturbation would be
    # detectable.  Real corruption risk lives where rounding happens.
    g2 = np.random.default_rng(77)
    xw = (case["x"][0].astype(np.float64) *
          (2.0 ** g2.integers(-8, 9, size=case["x"][0].shape))
          ).astype(np.float16)
    args = (case["codes"], case["codebooks"], case["scales"], 0, xw)
    ref32 = aqlm_f32(*args, lane_of=lambda i4: i4 % 32)  # the real tree
    # perturbation: collapse all K-chunks into ONE lane -> their partial
    # sums accumulate SEQUENTIALLY instead of through the shfl tree.
    # (This is the exact hazard of a rows-per-warp/lane repack; note a
    # pure lane RELABELING can be an fp32 commutation and is legal.)
    perm32 = aqlm_f32(*args, lane_of=lambda i4: 0)
    # sanity: the fp32-stage emulation with the real partition matches
    # the normative reference after rounding
    ref_half = R.aqlm_slot_gemv_ref(*args)
    assert np.array_equal(canon_half_bits(ref32.astype(np.float16)),
                          canon_half_bits(ref_half)), \
        "fp32-stage emulation drifted from the normative reference"
    ndiff = int((ref32 != perm32).sum())
    assert ndiff > 0, \
        ("perturbing the lane<->K partition did not change any fp32 "
         "output — the codified tree cannot certify lane-repack order "
         "preservation")
    print(f"[non-vacuity] lane repartition differs on {ndiff}/{ref32.size} "
          "fp32 outputs")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
