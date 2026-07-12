# SPDX-License-Identifier: Apache-2.0
"""TIER 1 — prefill dequant kernels vs the CPU reference, bit-exact.

Guards idea 7 (prefill chunk-size changes + prefill dequant kernel
rewrites): the prefill path dequantizes experts to fp16 and applies
plain GEMMs, so any dequant rewrite must stay bit-exact with the current
shipped dequant.  Reference semantics (see moe_reference.py):

  CodeKx16DequantMoE:  out = __hmul2( hadd2(book0, book1), half2(scale) )
  NvFp4DequantMoE:     s   = fp8(bscale) * gscale            (fp32 mul)
                       out = __floats2half2_rn( lut[nibble] * s )

Gate: maxdiff == 0 (bitwise, NaN-canonicalized) vs the shipped kernels.
"""

import numpy as np
import pytest

from common import kernels as K
from common import moe_reference as R
from common.fp_codecs import assert_half_bitexact

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")


@pytest.fixture(scope="module")
def gpu():
    idx = K.pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


@pytest.fixture(scope="module")
def ext():
    return K.load_ext()


CASES = [
    ("w13_small", 1), ("w13_small", 2),
    ("w2_small", 1), ("w2_small", 2),
    ("w2_real", 1),  # real w2 shard shape [6144, 512]
]


@pytest.mark.parametrize("shape,books", CASES,
                         ids=lambda c: str(c))
def test_aqlm_dequant_bitexact(shape, books, ext, gpu):
    case = K.make_case(shape, "all_aqlm", S=1, books=books, seed=31)
    tc = K.case_to_torch(case, gpu)
    nc = case["codes"].shape[0]
    # a strict subset + duplicate expert entries in the list
    el_np = np.array(list(range(nc)) + [0, nc - 1], dtype=np.int32)
    el = torch.from_numpy(el_np).to(gpu)
    got = ext.aqlm_moe_dequant(tc["codes"], tc["codebooks"], tc["scales"],
                               el).cpu().numpy()
    ref = R.aqlm_dequant_ref(case["codes"], case["codebooks"],
                             case["scales"], el_np)
    assert_half_bitexact(ref, got, what=f"aqlm_moe_dequant {shape} b{books}")


@pytest.mark.parametrize("shape", ["w13_small", "w2_small", "w2_real"])
def test_nvfp4_dequant_bitexact(shape, ext, gpu):
    case = K.make_case(shape, "all_nvfp4", S=1, books=1, seed=32)
    tc = K.case_to_torch(case, gpu)
    na = case["packed"].shape[0]
    el_np = np.arange(na, dtype=np.int32)
    el = torch.from_numpy(el_np).to(gpu)
    got = ext.nvfp4_moe_dequant(tc["packed"], tc["bscale"], tc["scale2"],
                                el).cpu().numpy()
    ref = R.nvfp4_dequant_ref(case["packed"], case["bscale"],
                              case["scale2"], el_np)
    assert_half_bitexact(ref, got, what=f"nvfp4_moe_dequant {shape}")


def test_dequant_consistent_with_gemv(ext, gpu):
    """Cross-check: dequant + fp32 matmul approximates the gemv closely.

    Not bit-exact (different accumulation trees by design) — this is a
    sanity net that the two paths implement the SAME weights.  A format/
    layout regression in either kernel produces gross errors here.
    """
    case = K.make_case("w13_small", "all_aqlm", S=4, books=2, seed=33)
    tc = K.case_to_torch(case, gpu)
    gemv = ext.aqlm_moe_gemv(tc["x"], tc["codes"], tc["codebooks"],
                             tc["scales"], tc["aqlm_ids"]).cpu().numpy()
    el = tc["aqlm_ids"].to(torch.int32)
    deq = ext.aqlm_moe_dequant(tc["codes"], tc["codebooks"], tc["scales"],
                               el)  # [S, M, K]
    ref = torch.einsum("sk,smk->sm", tc["x"].float(),
                       deq.float()).cpu().numpy()
    np.testing.assert_allclose(gemv.astype(np.float32), ref, atol=0.5,
                               rtol=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
