# SPDX-License-Identifier: Apache-2.0
"""TIER 1 — hybrid_moe_gemv vs the slow CPU reference, bit-exact.

Guards: EVERY change to the decode-path MoE gemv — DSMEM/cluster-resident
codebook (idea 1), w2 lane-occupancy repack (idea 2), L2 residency /
__ldcs hints (idea 5), the per-layer MoE megakernel (idea 6).

Reference: ``common/moe_reference.py`` — a dequant-to-fp32^W numpy
emulation that reproduces the kernel's EXACT accumulation tree
(documented there: per-lane fp16 fused-fma groups of 8, fp32 per-lane
accumulation in K-ascending order, shfl_down 16/8/4/2/1 fp32 tree, fp32
scale multiply, one final round to half).  Gate: maxdiff == 0 (bitwise,
NaN-canonicalized).

The one compiler-discretion point (fp32 two-block combine in the NVFP4
chunk dot, subject to nvcc -fmad contraction) is pinned by calibration
against the shipped kernel at session start; exactly one candidate mode
must match or the test hard-fails ("the kernel numerics changed").

Matrix: {w13 1024x6144, w2 6144x512 (real, sampled), small variants with
partial-lane tails} x {all-AQLM, all-NVFP4, mixed 30/70, masked slots
(both ids < 0), single slot, S=1/4/8/32} x {s2n 1,2 (via shape)} x
{BOOKS 1,2} + seeded fuzzing + adversarial values (denormal codebooks/
activations, max-magnitude fp16 scales, zero scales, NaN block scales).

NOTE: for FUTURE kernel variants the SHIPPED kernel is the reference
(see test_kernel_variant_equivalence.py); this test anchors the shipped
kernel itself to the documented tree.
"""

import os

import numpy as np
import pytest

from common import kernels as K
from common import moe_reference as R
from common.fp_codecs import assert_half_bitexact, max_ulp_diff_half

torch = pytest.importorskip("torch", reason="requires torch (run in the "
                            "glm52-sm120 container via run_all.sh)")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")


@pytest.fixture(scope="module")
def gpu():
    idx = K.pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory "
                    f"(need {K.min_free_mib_required()} MiB; box rules: "
                    "never pressure the live server)")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


@pytest.fixture(scope="module")
def ext():
    return K.load_ext()


@pytest.fixture(scope="module")
def fma_mode(ext, gpu):
    """Calibrate the NVFP4 fp32-combine codegen against the shipped kernel."""
    case = K.make_case("w13_small", "all_nvfp4", S=4, books=1, seed=1234)

    def run(c):
        return K.run_hybrid_gemv(ext, K.case_to_torch(c, gpu))

    mode = R.calibrate_fma_mode(run, case)
    print(f"\n[calibration] NVFP4 fp32 combine codegen = {mode!r}")
    return mode


QUICK = os.environ.get("GLM_SM120_QUICK") == "1"
MATRIX = list(K.full_matrix(quick=QUICK))


def _case_id(m):
    adv = f"-{m['adversarial']}" if m["adversarial"] else ""
    return f"{m['shape']}-{m['mix']}-S{m['S']}-b{m['books']}{adv}"


@pytest.mark.parametrize("meta", MATRIX, ids=_case_id)
def test_hybrid_gemv_bitexact(meta, ext, gpu, fma_mode):
    case = K.make_case(meta["shape"], meta["mix"], meta["S"], meta["books"],
                       meta["seed"], meta["adversarial"])
    got = K.run_hybrid_gemv(ext, K.case_to_torch(case, gpu))
    ref = R.hybrid_moe_gemv_ref(
        case["x"], case["codes"], case["codebooks"], case["scales"],
        case["aqlm_ids"], case["packed"], case["bscale"], case["scale2"],
        case["nv_ids"], fma_mode=fma_mode)
    ulp = max_ulp_diff_half(ref, got)
    assert_half_bitexact(
        ref, got,
        what=f"hybrid_moe_gemv {_case_id(meta)} (max ulp diff {ulp})")


@pytest.mark.parametrize("seed", range(8))
def test_hybrid_gemv_fuzz(seed, ext, gpu, fma_mode):
    """Randomized fuzzing: random small shapes, mixes and slot counts."""
    rng = np.random.default_rng(1000 + seed)
    shape = ["w13_small", "w2_small"][seed % 2]
    mix = ["all_aqlm", "all_nvfp4", "mixed", "masked"][int(rng.integers(4))]
    s = int(rng.choice([1, 2, 4, 8, 16, 32]))
    books = int(rng.choice([1, 2]))
    case = K.make_case(shape, mix, s, books, seed=2000 + seed)
    got = K.run_hybrid_gemv(ext, K.case_to_torch(case, gpu))
    ref = R.hybrid_moe_gemv_ref(
        case["x"], case["codes"], case["codebooks"], case["scales"],
        case["aqlm_ids"], case["packed"], case["bscale"], case["scale2"],
        case["nv_ids"], fma_mode=fma_mode)
    assert_half_bitexact(ref, got, what=f"fuzz seed={seed} {shape} {mix} S={s}")


def test_standalone_kernels_match_fused(ext, gpu):
    """aqlm_moe_gemv + nvfp4_moe_gemv (masked, summed) == hybrid_moe_gemv.

    The fused kernel replaced the two-launch path; they must stay
    interchangeable (each slot computed by exactly one format, others
    contribute half zeros -> fp16 x + 0 == x for all finite x).
    """
    case = K.make_case("w13_small", "mixed", S=8, books=1, seed=77)
    tc = K.case_to_torch(case, gpu)
    fused = K.run_hybrid_gemv(ext, tc)
    a = ext.aqlm_moe_gemv(tc["x"], tc["codes"], tc["codebooks"],
                          tc["scales"], tc["aqlm_ids"]).cpu().numpy()
    n = ext.nvfp4_moe_gemv(tc["x"], tc["packed"], tc["bscale"],
                           tc["scale2"], tc["nv_ids"]).cpu().numpy()
    # exactly one path active per slot: masked slots are zero in both
    active_a = case["aqlm_ids"] >= 0
    combined = np.where(active_a[:, None], a, n)
    both_masked = (case["aqlm_ids"] < 0) & (case["nv_ids"] < 0)
    combined[both_masked] = np.float16(0.0)
    assert_half_bitexact(combined, fused,
                         what="fused vs standalone kernels")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
