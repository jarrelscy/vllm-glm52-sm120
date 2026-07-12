# SPDX-License-Identifier: Apache-2.0
"""TIER 1 — parameterized A/B: every registered kernel variant vs shipped.

Guards ideas 1 (DSMEM codebook), 2 (w2 lane repack), 5 (__ldcs/L2
residency), 6 (MoE megakernel) and any future env/compile-gated kernel
flag.  The SHIPPED kernel (default build of aqlm_moe_v2.cu) IS the
reference; variants gate bit-exact unless their registry entry declares
and justifies a tolerance.

Auto-discovery: ``kernel_variants.json`` at the suite root.  Optimizer
agents MUST append their flag there (schema in
common/variant_registry.py) — this test then covers the new variant with
zero test edits.  The registry ships with three real compile-time
variants of the shipped source (NVFP4_LUT256, AQLM_CB_L1, AQLM_MLP=4) so
the machinery is exercised, not vacuous.
"""

import zlib

import numpy as np
import pytest

from common import kernels as K
from common.fp_codecs import assert_half_bitexact
from common.variant_registry import load_registry, resolve_compare

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")

VARIANTS = load_registry()

# A/B cases: cover both storage formats, masked slots, both books,
# partial-lane tails and (one) real shape.
AB_CASES = [
    ("w13_small", "mixed", 8, 1),
    ("w13_small", "mixed", 32, 2),
    ("w2_small", "all_aqlm", 8, 2),
    ("w2_small", "all_nvfp4", 8, 1),
    ("w13_small", "masked", 8, 1),
    ("w2_real", "mixed", 32, 1),
]


@pytest.fixture(scope="module")
def gpu():
    idx = K.pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


@pytest.fixture(scope="module")
def base_ext():
    return K.load_ext()


def _run_op(ext, op: str, tc: dict) -> np.ndarray:
    if op == "hybrid_moe_gemv":
        return K.run_hybrid_gemv(ext, tc)
    if op == "aqlm_moe_gemv":
        return ext.aqlm_moe_gemv(tc["x"], tc["codes"], tc["codebooks"],
                                 tc["scales"], tc["aqlm_ids"]).cpu().numpy()
    if op == "nvfp4_moe_gemv":
        if tc["packed"].numel() == 0:
            return None
        return ext.nvfp4_moe_gemv(tc["x"], tc["packed"], tc["bscale"],
                                  tc["scale2"], tc["nv_ids"]).cpu().numpy()
    if op == "aqlm_moe_dequant":
        el = torch.arange(tc["codes"].shape[0], dtype=torch.int32,
                          device=tc["x"].device)
        return ext.aqlm_moe_dequant(tc["codes"], tc["codebooks"],
                                    tc["scales"], el).cpu().numpy()
    if op == "nvfp4_moe_dequant":
        if tc["packed"].numel() == 0:
            return None
        el = torch.arange(tc["packed"].shape[0], dtype=torch.int32,
                          device=tc["x"].device)
        return ext.nvfp4_moe_dequant(tc["packed"], tc["bscale"],
                                     tc["scale2"], el).cpu().numpy()
    raise ValueError(f"unknown op {op!r}")


def test_registry_schema():
    """CPU-safe: the registry parses and every entry is well-formed."""
    assert isinstance(VARIANTS, list)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v["name"])
@pytest.mark.parametrize("case_key", AB_CASES,
                         ids=lambda c: f"{c[0]}-{c[1]}-S{c[2]}-b{c[3]}")
def test_variant_equivalence(variant, case_key, base_ext, gpu):
    shape, mix, s, books = case_key
    seed = zlib.crc32(repr(case_key).encode()) % 10000  # stable across runs
    case = K.make_case(shape, mix, s, books, seed=seed)
    tc = K.case_to_torch(case, gpu)

    if variant["kind"] == "cuda_cflag":
        var_ext = K.load_ext(cflags=tuple(variant["cflags"]))
        for op in variant["ops"]:
            ref = _run_op(base_ext, op, tc)
            if ref is None:
                continue
            got = _run_op(var_ext, op, tc)
            _gate(variant, ref, got, f"{variant['name']}:{op}")
    else:  # python_env
        import os
        compare = resolve_compare(variant["compare"])
        saved = {k: os.environ.get(k) for k in variant["env"]}
        os.environ.update(variant["env"])
        try:
            ref, got = compare(tc, gpu)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _gate(variant, np.asarray(ref), np.asarray(got), variant["name"])


def _gate(variant, ref, got, what):
    if variant["bit_exact"]:
        assert_half_bitexact(ref, got, what=what)
    else:
        tol = variant["tolerance"]
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(ref, np.float32),
            atol=tol["atol"], rtol=tol["rtol"],
            err_msg=f"{what}: outside declared tolerance {tol}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
