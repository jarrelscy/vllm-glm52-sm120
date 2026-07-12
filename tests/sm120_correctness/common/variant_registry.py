# SPDX-License-Identifier: Apache-2.0
"""Registry of env/compile-gated kernel variants.

Every optimization agent that adds a kernel variant (DSMEM codebook,
w2 lane repack, MoE megakernel, __ldcs residency hints, ...) MUST append
an entry to ``kernel_variants.json`` so that
``tier1_kernel/test_kernel_variant_equivalence.py`` auto-discovers and
A/B-gates it (bit-exact vs the shipped kernel) without any test edits.

Entry schema (see kernel_variants.json for live examples):

  {
    "name":      "dsmem_codebook",          # unique
    "kind":      "cuda_cflag",              # or "python_env"
    "cflags":    ["-DAQLM_DSMEM=1"],        # kind=cuda_cflag: rebuild
                                            #   aqlm_moe_v2.cu with these
    "env":       {"VLLM_MOE_DSMEM": "1"},   # kind=python_env: runtime flag
    "compare":   "module.path:function",    # kind=python_env only —
                                            #   fn(case_torch, device) ->
                                            #   (ref_out, test_out) numpy
    "ops":       ["hybrid_moe_gemv"],       # ops to A/B (cuda_cflag kind)
    "bit_exact": true,                      # gate; false needs "tolerance"
    "tolerance": null,                      # {"atol":..,"rtol":..} if not
                                            #   bit-exact (must justify)
    "notes":     "what the variant changes"
  }
"""

from __future__ import annotations

import json
import pathlib

SUITE_DIR = pathlib.Path(__file__).resolve().parents[1]
REGISTRY_PATH = SUITE_DIR / "kernel_variants.json"

REQUIRED_KEYS = {"name", "kind", "bit_exact", "notes"}
VALID_KINDS = {"cuda_cflag", "python_env"}


def load_registry() -> list[dict]:
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    assert data.get("version") == 1, "unknown registry version"
    variants = data["variants"]
    names = [v["name"] for v in variants]
    assert len(names) == len(set(names)), f"duplicate variant names: {names}"
    for v in variants:
        missing = REQUIRED_KEYS - set(v)
        assert not missing, f"variant {v.get('name')}: missing {missing}"
        assert v["kind"] in VALID_KINDS, v
        if v["kind"] == "cuda_cflag":
            assert v.get("cflags"), f"{v['name']}: cuda_cflag needs cflags"
            assert v.get("ops"), f"{v['name']}: cuda_cflag needs ops list"
        else:
            assert v.get("compare"), f"{v['name']}: python_env needs compare"
        if not v["bit_exact"]:
            assert v.get("tolerance"), \
                f"{v['name']}: non-bit-exact variants must declare tolerance"
    return variants


def resolve_compare(spec: str):
    mod_path, fn_name = spec.split(":")
    import importlib
    return getattr(importlib.import_module(mod_path), fn_name)
