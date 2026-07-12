# SPDX-License-Identifier: Apache-2.0
"""TIER 1 — concat_and_cache_ds_mla_kernel invariants.

This is a direct regression net for the SM100 bug class: on B200,
fp8_ds_mla routed to a broken kernel and deep reasoning silently
collapsed (finish=stop at <=~9.5k tokens, wrong answers, NO crash).
The KV-write layout and scale semantics below are what a broken kernel
would violate FIRST.

656-byte entry layout for fp8_ds_mla (kv_lora_rank=512, pe_dim=64):
  bytes [  0, 512): 512x fp8-e4m3 quantized NoPE values (4 tiles of 128)
  bytes [512, 528): 4x fp32 per-tile scales (tile i at 512 + 4*i)
  bytes [528, 656): 64x 16-bit RoPE values, copied verbatim

Scale semantics (2026-07-12, commit 2dce07864): tile_scale = the SMALLEST
POWER OF TWO >= max( max|x_tile| / 448, FLT_MIN ).  Pow2 storage makes the
SM100 FlashMLA e8m0 scale read lossless (its truncation of an arbitrary
fp32 scale to 2^floor(log2) was the SM100 decode-corruption bug) while the
SM120 arbitrary-fp32 read path reads the same stored value exactly.  We pin
pow2-ness, the >= bound (no fp8 saturation), and minimality (< 2x raw).

Requires the compiled vLLM _C ops (run inside the glm52-sm120 container).
"""

import math

import numpy as np
import pytest

from common.fp_codecs import (
    F32,
    FLT_MIN,
    f32_to_fp8_e4m3_satfinite,
    fp8_e4m3_to_f32,
)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")

KV_LORA, PE_DIM, ENTRY = 512, 64, 656
BLOCK_SIZE = 64


@pytest.fixture(scope="module")
def gpu():
    from common.kernels import pick_gpu
    idx = pick_gpu()
    if idx is None:
        pytest.skip("no GPU with enough free memory")
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


@pytest.fixture(scope="module")
def cache_op(gpu):
    ops = pytest.importorskip(
        "vllm._custom_ops",
        reason="needs the compiled vLLM _C ops (glm52-sm120 container)")
    return ops.concat_and_cache_mla


def _write(cache_op, gpu, kv_c, k_pe, slots, num_blocks=4):
    """kv_c / k_pe: float32 numpy holding bf16-exact values."""
    kv_c_t = torch.from_numpy(kv_c).to(gpu).to(torch.bfloat16)
    k_pe_t = torch.from_numpy(k_pe).to(gpu).to(torch.bfloat16)
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, ENTRY, dtype=torch.uint8,
                           device=gpu)
    slot_map = torch.tensor(slots, dtype=torch.int64, device=gpu)
    scale = torch.ones(1, dtype=torch.float32, device=gpu)
    cache_op(kv_c_t, k_pe_t, kv_cache, slot_map, "fp8_ds_mla", scale)
    return kv_cache.cpu().numpy()


def _entry(cache, slot):
    return cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE]


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit pattern of an fp32 array (RNE via torch)."""
    return (torch.from_numpy(np.ascontiguousarray(a, np.float32))
            .to(torch.bfloat16).view(torch.uint16).numpy())


def _ref_pack(kv_row: np.ndarray, pe_row: np.ndarray) -> np.ndarray:
    """CPU reference of the 656-byte entry (normative layout).

    kv_row / pe_row: float32 numpy holding bf16-exact values.
    """
    out = np.zeros(ENTRY, dtype=np.uint8)
    x = kv_row.astype(F32)
    for tile in range(4):
        vals = x[tile * 128:(tile + 1) * 128]
        max_abs = np.max(np.abs(vals.astype(F32)))  # fmax tree, order-free
        scale = np.maximum((max_abs / np.float32(448.0)).astype(F32), FLT_MIN)
        q = f32_to_fp8_e4m3_satfinite((vals / scale).astype(F32))
        out[tile * 128:(tile + 1) * 128] = q
        out[512 + 4 * tile: 516 + 4 * tile] = np.frombuffer(
            np.float32(scale).tobytes(), dtype=np.uint8)
    out[528:656] = _bf16_bits(pe_row).view(np.uint8)
    return out


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round fp32 -> bf16 values, returned as bf16-exact float32."""
    return (torch.from_numpy(np.ascontiguousarray(a, np.float32))
            .to(torch.bfloat16).to(torch.float32).numpy())


def _mk_inputs(num_tokens, seed=0, scale_mag=1.0):
    g = np.random.default_rng(seed)
    kv = _bf16((g.standard_normal((num_tokens, KV_LORA)) * scale_mag)
               .astype(np.float32))
    pe = _bf16(g.standard_normal((num_tokens, PE_DIM)).astype(np.float32))
    return kv, pe


def test_layout_offsets(cache_op, gpu):
    """(a) NoPE / scales / RoPE land at the documented byte offsets."""
    kv, pe = _mk_inputs(3, seed=1)
    cache = _write(cache_op, gpu, kv, pe, [0, 5, 2 * BLOCK_SIZE + 7])
    for tok, slot in enumerate([0, 5, 2 * BLOCK_SIZE + 7]):
        e = _entry(cache, slot)
        # RoPE region: verbatim 16-bit copy
        got_pe = e[528:656].view(np.uint16)
        assert np.array_equal(got_pe, _bf16_bits(pe[tok])), \
            f"RoPE bytes wrong for token {tok}"
        # scales region: 4 finite positive fp32
        scales = e[512:528].view(np.float32)
        assert np.all(np.isfinite(scales)) and np.all(scales > 0)
        # NoPE region: dequant must correlate with the input (catches
        # swapped/garbage regions even before the exact-byte check below)
        deq = np.concatenate([
            fp8_e4m3_to_f32(e[t * 128:(t + 1) * 128]) * scales[t]
            for t in range(4)])
        x = kv[tok].astype(np.float32)
        corr = np.corrcoef(deq, x)[0, 1]
        assert corr > 0.99, f"NoPE region decorrelated (corr={corr:.4f})"


def test_bytes_match_reference(cache_op, gpu):
    """Entry bytes == CPU reference pack, bit-for-bit (normal values)."""
    kv, pe = _mk_inputs(8, seed=2)
    slots = list(range(8))
    cache = _write(cache_op, gpu, kv, pe, slots)
    for tok, slot in enumerate(slots):
        got = _entry(cache, slot)
        ref = _ref_pack(kv[tok], pe[tok])
        # scales must match bit-for-bit
        assert np.array_equal(got[512:528], ref[512:528]), \
            (f"tile scales differ for token {tok}: "
             f"got {got[512:528].view(np.float32)} "
             f"ref {ref[512:528].view(np.float32)}")
        # fp8 payload bit-for-bit
        mism = np.flatnonzero(got[:512] != ref[:512])
        assert mism.size == 0, \
            (f"token {tok}: {mism.size}/512 fp8 bytes differ, first at "
             f"{mism[0]}: got 0x{got[mism[0]]:02x} ref 0x{ref[mism[0]]:02x}")
        assert np.array_equal(got[528:], ref[528:])


def test_scales_are_pow2_rounded_up(cache_op, gpu):
    """(b) Scale convention pin (2026-07-12, commit 2dce07864): stored per-tile
    scales are the SMALLEST POWER OF TWO >= max_abs/448.  Exact pow2 storage
    makes SM100 FlashMLA's e8m0 scale read lossless while the SM120
    arbitrary-fp32 read path reads the same value exactly.  A stored scale that
    is NOT a pow2, or that is below max_abs/448 (would saturate fp8), or more
    than 2x above it (over-rounded), is a regression."""
    g = np.random.default_rng(3)
    # values chosen so max_abs/448 has an odd mantissa (never already a pow2)
    kv = _bf16((g.random((4, KV_LORA)).astype(np.float32) + 0.5) * 3.1416)
    pe = _bf16(g.standard_normal((4, PE_DIM)).astype(np.float32))
    cache = _write(cache_op, gpu, kv, pe, [0, 1, 2, 3])
    for tok in range(4):
        e = _entry(cache, tok)
        scales = e[512:528].view(np.float32)
        for t in range(4):
            vals = kv[tok].astype(np.float32)[t * 128:(t + 1) * 128]
            raw = np.maximum(
                (np.float32(np.max(np.abs(vals))) / np.float32(448.0))
                .astype(F32), FLT_MIN)
            frac, _ = math.frexp(float(scales[t]))
            assert frac == 0.5, \
                (f"tile {t}: stored scale {scales[t]!r} is not a power of two "
                 "— pow2 KV-scale convention regressed")
            assert scales[t] >= raw, \
                (f"tile {t}: pow2 scale {scales[t]!r} < max_abs/448 {raw!r} "
                 "— fp8 quotient would saturate")
            assert scales[t] < 2.0 * raw, \
                (f"tile {t}: pow2 scale {scales[t]!r} not the SMALLEST pow2 "
                 f">= {raw!r} — over-rounded")


def test_roundtrip_error_bounds(cache_op, gpu):
    """(c) dequant(write(x)) within fp8-e4m3 quantization error of x."""
    for mag in (0.01, 1.0, 100.0):
        kv, pe = _mk_inputs(4, seed=4, scale_mag=mag)
        cache = _write(cache_op, gpu, kv, pe, [0, 1, 2, 3])
        for tok in range(4):
            e = _entry(cache, tok)
            scales = e[512:528].view(np.float32)
            x = kv[tok].astype(np.float32)
            for t in range(4):
                deq = fp8_e4m3_to_f32(e[t * 128:(t + 1) * 128]) * scales[t]
                xt = x[t * 128:(t + 1) * 128]
                # e4m3 RN: rel err <= 2^-4 for normals, plus one subnormal
                # step of the scaled grid for tiny values
                err = np.abs(deq - xt)
                bound = np.maximum(np.abs(xt) * 2.0**-4,
                                   scales[t] * 2.0**-3)
                bad = np.flatnonzero(err > bound)
                assert bad.size == 0, \
                    (f"mag={mag} token {tok} tile {t}: {bad.size} elements "
                     f"outside fp8 error bounds, worst "
                     f"{float(err[bad].max()):.4g}")


def test_padded_slot_skipped(cache_op, gpu):
    """slot_idx == -1 (padding) must write NOTHING."""
    kv, pe = _mk_inputs(2, seed=5)
    kv_c_t = torch.from_numpy(kv).to(gpu)
    k_pe_t = torch.from_numpy(pe).to(gpu)
    kv_cache = torch.full((2, BLOCK_SIZE, ENTRY), 0xAB, dtype=torch.uint8,
                          device=gpu)
    slot_map = torch.tensor([3, -1], dtype=torch.int64, device=gpu)
    scale = torch.ones(1, dtype=torch.float32, device=gpu)
    cache_op(kv_c_t, k_pe_t, kv_cache, slot_map, "fp8_ds_mla", scale)
    c = kv_cache.cpu().numpy()
    assert not np.all(_entry(c, 3) == 0xAB), "real token was not written"
    untouched = np.delete(c.reshape(-1, ENTRY), 3, axis=0)
    assert np.all(untouched == 0xAB), "padded token corrupted other slots"


def test_nan_inf_isolation(cache_op, gpu):
    """(d) NaN/Inf in one tile never corrupts other tiles/tokens; the
    kernel must not crash.  Exact NaN encoding is reported, not gated."""
    kv, pe = _mk_inputs(3, seed=6)
    kv_f = kv.astype(np.float32)
    kv_f[1, 130] = np.nan     # tile 1 of token 1
    kv_f[1, 300] = np.inf     # tile 2 of token 1
    kv = _bf16(kv_f)
    cache = _write(cache_op, gpu, kv, pe, [0, 1, 2])
    ref0 = _ref_pack(kv[0], pe[0])
    ref2 = _ref_pack(kv[2], pe[2])
    assert np.array_equal(_entry(cache, 0), ref0), \
        "clean token 0 corrupted by NaN/Inf in token 1"
    assert np.array_equal(_entry(cache, 2), ref2), \
        "clean token 2 corrupted by NaN/Inf in token 1"
    e1 = _entry(cache, 1)
    scales1 = e1[512:528].view(np.float32)
    # tiles 0 and 3 of token 1 contain clean data: must match reference
    ref1 = _ref_pack(np.nan_to_num(kv[1].astype(np.float32),
                                   nan=0, posinf=0, neginf=0)
                     .astype(np.float32), pe[1])
    # only compare the clean tiles (0 and 3) where nan_to_num is a no-op
    assert np.array_equal(e1[0:128], ref1[0:128]), "clean tile 0 corrupted"
    assert np.array_equal(e1[384:512], ref1[384:512]), "clean tile 3 corrupted"
    print(f"\n[info] NaN-tile scale={scales1[1]!r} byte@130=0x"
          f"{e1[130]:02x}; Inf-tile scale={scales1[2]!r} byte@300=0x"
          f"{e1[300]:02x} (behavior documented, not gated)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
