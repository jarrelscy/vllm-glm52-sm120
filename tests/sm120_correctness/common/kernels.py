# SPDX-License-Identifier: Apache-2.0
"""GPU-side helpers: JIT-building the hybrid MoE extension, GPU picking
with a strict memory guard (Box rules: never pressure the live server),
and generation of the full Tier-1 shape/content matrix.
"""

from __future__ import annotations

import functools
import os
import pathlib
import subprocess

import numpy as np

SUITE_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_DIR.parents[1]
CU_SOURCE = REPO_ROOT / "csrc" / "quantization" / "aqlm_moe" / "aqlm_moe_v2.cu"

# Real production shapes (GLM-5.2 hybrid, TP4): H=6144, moe_inter=2048,
# TP4 -> I_shard=512; w13 gemv is [M=1024=2*512, K=6144], w2 is
# [M=6144, K=512].  Small shapes keep the CPU reference fast; the real
# shapes are exercised in the `real_shapes` cases.
SHAPES = {
    "w13_small": dict(M=128, K=256, s2n=2),
    "w2_small": dict(M=96, K=192, s2n=1),  # K=192: partial-lane tail (K/64=3)
    "w13_real": dict(M=1024, K=6144, s2n=2),
    "w2_real": dict(M=6144, K=512, s2n=1),
}


def min_free_mib_required() -> int:
    return int(os.environ.get("GLM_SM120_MIN_FREE_MIB", "8000"))


def pick_gpu() -> int | None:
    """Pick the GPU with the most free memory; None if none has enough.

    Box rule: a tiny JIT context (~1-2GB) only, and only when there is
    comfortable headroom so the live server is never pressured.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return 0  # respect the caller's pinning; index within visible set
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    best, best_free = None, -1
    for line in out.strip().splitlines():
        idx, free = (int(x) for x in line.split(","))
        if free > best_free:
            best, best_free = idx, free
    if best_free < min_free_mib_required():
        return None
    return best


@functools.lru_cache(maxsize=None)
def load_ext(name: str = "aqlm_moe_ext_v2", cflags: tuple[str, ...] = ()):
    """JIT-build the hybrid MoE extension from the checked-out source.

    Distinct ``cflags`` produce distinct extension names so compile-time
    variants (e.g. -DNVFP4_LUT256=1) can be loaded side by side.
    """
    from torch.utils.cpp_extension import load

    extra = ["-O3", *cflags]
    if cflags:
        suffix = "_" + "_".join(
            c.replace("-", "").replace("=", "").replace("D", "", 1)
            for c in cflags).lower()
        name = name + suffix
    return load(name=name, sources=[str(CU_SOURCE)],
                extra_cuda_cflags=extra, verbose=False)


# ---------------------------------------------------------------------------
# Case generation: the full Tier-1 matrix
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_case(
    shape: str,
    mix: str,            # all_aqlm | all_nvfp4 | mixed | masked | single
    S: int,
    books: int,
    seed: int,
    adversarial: str | None = None,  # denormals|max_scales|zero_scales|nan_bscale
) -> dict:
    """Build one hybrid_moe_gemv input case as numpy arrays.

    Content mirrors what the server feeds the kernel: fp16 activations,
    int16 AQLM codes over 65536-entry fp16 codebooks with per-row fp16
    scales, packed fp4 weights with fp8-e4m3 block scales and fp32
    per-half tensor scales.
    """
    p = SHAPES[shape]
    m, k, s2n = p["M"], p["K"], p["s2n"]
    g = _rng(seed)

    E = 8
    if mix == "all_aqlm":
        na, per_slot = 0, "a"
    elif mix == "all_nvfp4":
        na, per_slot = E, "n"
    elif mix == "mixed":     # ~30% hot / 70% cold, like production
        na, per_slot = 3, "?"
    elif mix == "masked":    # some slots with both ids < 0
        na, per_slot = 3, "m"
    elif mix == "single":
        na, per_slot = 3, "?"
        S = 1
    else:
        raise ValueError(mix)
    nc = E - na

    x = (g.standard_normal((S, k)) * 0.05).astype(np.float16)
    codes = g.integers(-32768, 32768, size=(max(nc, 1), books, m, k // 8),
                       dtype=np.int16)
    codebooks = (g.standard_normal((books, 65536, 8)) * 0.05).astype(np.float16)
    scales = (g.standard_normal((max(nc, 1), m)) * 0.3 + 0.5).astype(np.float16)
    packed = g.integers(0, 256, size=(max(na, 1), m, k // 2), dtype=np.uint8)
    # keep block-scale exponents small so fp16 partials stay finite
    bscale = g.integers(0, 40, size=(max(na, 1), m, k // 16), dtype=np.uint8)
    scale2 = (g.random((max(na, 1), s2n)) * 0.2 + 0.2).astype(np.float32)

    if adversarial == "denormals":
        cb = codebooks.view(np.uint16)
        pick = g.random(cb.shape) < 0.05
        cb[pick] = g.integers(1, 0x400, size=int(pick.sum())).astype(np.uint16)
        xv = x.view(np.uint16)
        pick = g.random(xv.shape) < 0.05
        xv[pick] = g.integers(1, 0x400, size=int(pick.sum())).astype(np.uint16)
    elif adversarial == "max_scales":
        scales[:] = np.float16(65504.0)          # max-magnitude fp16 scale
        scale2[:] = np.float32(3.0e38)           # large fp32 tensor scale
    elif adversarial == "zero_scales":
        scales[:] = np.float16(0.0)
        scale2[:] = np.float32(0.0)
        bscale[:, ::3] = 0                       # fp8 zero block scales
    elif adversarial == "nan_bscale":
        bscale[:, 0, 0] = 0x7F                   # e4m3 NaN block scale
    elif adversarial is not None:
        raise ValueError(adversarial)

    # slot -> expert assignment
    aqlm_ids = np.full(S, -1, dtype=np.int32)
    nv_ids = np.full(S, -1, dtype=np.int32)
    for s_ in range(S):
        kind = per_slot
        if kind == "?":
            kind = "n" if g.random() < 0.3 and na > 0 else "a"
        if kind == "m" and s_ % 3 == 2:
            continue  # masked slot: both ids stay -1
        elif kind == "m":
            kind = "n" if s_ % 2 == 0 and na > 0 else "a"
        if kind == "a" and nc > 0:
            aqlm_ids[s_] = int(g.integers(0, nc))
        elif na > 0:
            nv_ids[s_] = int(g.integers(0, na))
        else:
            aqlm_ids[s_] = int(g.integers(0, nc))

    return dict(x=x, codes=codes, codebooks=codebooks, scales=scales,
                aqlm_ids=aqlm_ids, packed=packed, bscale=bscale,
                scale2=scale2, nv_ids=nv_ids,
                meta=dict(shape=shape, mix=mix, S=S, books=books, seed=seed,
                          adversarial=adversarial, na=na, nc=nc))


def case_to_torch(case: dict, device: str):
    import torch

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    has_nv = case["meta"]["na"] > 0
    e8 = torch.empty(0, dtype=torch.uint8, device=device)
    ef = torch.empty(0, dtype=torch.float32, device=device)
    return dict(
        x=t(case["x"]),
        codes=t(case["codes"]),
        codebooks=t(case["codebooks"]),
        scales=t(case["scales"]),
        aqlm_ids=t(case["aqlm_ids"]),
        packed=t(case["packed"]) if has_nv else e8,
        bscale=t(case["bscale"]) if has_nv else e8,
        scale2=t(case["scale2"]) if has_nv else ef,
        nv_ids=t(case["nv_ids"]),
    )


def run_hybrid_gemv(ext, tc: dict):
    out = ext.hybrid_moe_gemv(
        tc["x"], tc["codes"], tc["codebooks"], tc["scales"], tc["aqlm_ids"],
        tc["packed"], tc["bscale"], tc["scale2"], tc["nv_ids"])
    return out.cpu().numpy()


def full_matrix(quick: bool = False):
    """Yield the exhaustive Tier-1 case list (meta only; build lazily)."""
    shapes = ["w13_small", "w2_small"] if quick else \
             ["w13_small", "w2_small", "w13_real", "w2_real"]
    mixes = ["all_aqlm", "all_nvfp4", "mixed", "masked", "single"]
    s_values = [1, 4, 8, 32]
    books_values = [1, 2]
    seed = 0
    for shape in shapes:
        for mix in mixes:
            for books in books_values:
                for S in ([4] if mix == "single" else s_values):
                    if quick and S not in (1, 4):
                        continue
                    # real shapes are slow on the CPU reference: sample
                    if shape.endswith("_real") and not (
                            S == 32 and books == 1 and mix in
                            ("mixed", "all_aqlm", "all_nvfp4")):
                        continue
                    seed += 1
                    yield dict(shape=shape, mix=mix, S=S, books=books,
                               seed=seed, adversarial=None)
    # adversarial content, small shapes only
    for adv in ["denormals", "max_scales", "zero_scales", "nan_bscale"]:
        for shape in ["w13_small", "w2_small"]:
            seed += 1
            yield dict(shape=shape, mix="mixed", S=8, books=1, seed=seed,
                       adversarial=adv)
