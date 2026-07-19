# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform

# FA4's blockscaled MMA path is hard-coded to a 32-element scale group
# (see flash_attn_varlen_func: qk_sf_vec_size/v_sf_vec_size are always 32
# when sfq/sfv are provided at all -- there is no other supported value).
FA4_BLOCK_SCALE_VEC_SIZE = 32


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the FA4 scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


def quantize_q_to_fp8_blockscaled(
    q: torch.Tensor, vec_size: int = FA4_BLOCK_SCALE_VEC_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize Q to e4m3 with per-``vec_size`` ue8m0 scales.

    ``q`` is ``(num_tokens, num_heads, head_dim)``. Reuses vLLM's existing,
    production ``per_token_group_quant_fp8`` primitive (group axis = last
    dim = head_dim) rather than hand-rolling quantization math. Returns
    ``(q_fp8, sfq)`` where ``sfq`` is ``float8_e8m0fnu`` of shape
    ``(num_tokens, num_heads, head_dim // vec_size)``, matching what
    ``flash_attn_varlen_func`` expects for ``sfq``.
    """
    assert q.shape[-1] % vec_size == 0, (
        f"FA4 blockscaled attention requires head_dim ({q.shape[-1]}) "
        f"divisible by vec_size ({vec_size})"
    )
    num_tokens, num_heads, head_dim = q.shape
    # per_token_group_quant_fp8's Python-level shape logic looks general
    # (docstring says "ndim >= 2", and it happily computes a 3D output_s for
    # a 3D input), but the CUDA kernel it dispatches to
    # (per_token_group_quant.cu:204, `STD_TORCH_CHECK(output_s.dim() == 2)`)
    # hard-requires a 2D scale tensor. Every other caller in the tree
    # (deepseek_v32/nvidia/attention.py, fused_moe/utils.py,
    # deep_gemm_moe.py, input_quant_fp8.py) already flattens to 2D before
    # calling and reshapes back after -- follow that same convention here.
    q_c = q.reshape(num_tokens * num_heads, head_dim).contiguous()
    q_fp8, sfq_f32 = per_token_group_quant_fp8(
        q_c, group_size=vec_size, dtype=torch.float8_e4m3fn, use_ue8m0=True
    )
    q_fp8 = q_fp8.view(num_tokens, num_heads, head_dim)
    sfq_f32 = sfq_f32.view(num_tokens, num_heads, head_dim // vec_size)
    # use_ue8m0=True already rounds the scale to an exact power of two, so
    # this cast to float8_e8m0fnu (an exponent-only format) is lossless.
    return q_fp8, sfq_f32.to(torch.float8_e8m0fnu)


def uniform_ue8m0_block_scale(
    scalar_scale: torch.Tensor,
    ref: torch.Tensor,
    vec_size: int = FA4_BLOCK_SCALE_VEC_SIZE,
    *,
    cache: dict[tuple[int, ...], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Broadcast a single per-layer fp32 scale into FA4's per-block ue8m0 shape.

    NOTE(inkling-fp8-kv, first pass / task #106): ``scalar_scale`` here is
    the existing per-layer ``k_scale``/``v_scale`` buffer, which today is
    never calibrated (always 1.0) and never applied anywhere else in the
    Inkling model -- i.e. the KV cache is currently written to the fp8
    physical buffer with an implicit scale of 1.0. This function makes that
    *explicit* and consistent with what FA4's blockscaled kernel expects, by
    materializing a real (not just broadcast/stride-0) tensor of shape
    ``ref.shape[:-1] + (ref.shape[-1] // vec_size,)`` filled with the
    ue8m0-rounded scalar. It is deliberately a real allocation (cached by
    shape, not recomputed every call) rather than a zero-stride view, since
    the cute/cutlass-level kernel's tensor-layout assumptions for mSFK/mSFV
    have not been verified to tolerate broadcast strides.

    This is a coarse, per-tensor (not true per-32-element) scale -- it makes
    `--kv-cache-dtype fp8_e4m3` functional (no hard-assert, dtype/shape
    contract satisfied) rather than crashing, but does not yet give Inkling
    real adaptive block-wise fp8 precision. True per-block K/V scale
    computation would need to happen at cache-*write* time inside
    `fused_qkvr_prep` (which currently has no fp8/scale-aware logic at all --
    confirmed via inspection, this is a separate, larger follow-up, not
    addressed by this pass).
    """
    assert ref.shape[-1] % vec_size == 0, (
        f"FA4 blockscaled attention requires the cache's head_dim "
        f"({ref.shape[-1]}) divisible by vec_size ({vec_size})"
    )
    out_shape = tuple(ref.shape[:-1]) + (ref.shape[-1] // vec_size,)
    if cache is not None:
        cached = cache.get(out_shape)
        if cached is not None:
            return cached
    # ue8m0 is exponent-only: round the scalar up to the nearest power of two,
    # matching the same convention used elsewhere in vLLM (see
    # `tl.math.exp2(tl.ceil(tl.log2(scale_raw)))` in fp8_utils.py) before
    # casting down to float8_e8m0fnu.
    scalar_f32 = scalar_scale.detach().to(torch.float32).clamp(min=1e-12)
    rounded = torch.exp2(torch.ceil(torch.log2(scalar_f32)))
    block_scale = torch.full(
        out_shape, float(rounded.item()), dtype=torch.float8_e8m0fnu, device=ref.device
    )
    if cache is not None:
        cache[out_shape] = block_scale
    return block_scale


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Return the split-KV cap for FA4 with sheared relative bias."""
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major == 9:
        return 1
    if capability is not None and capability.major == 12:
        # SM120's shared SM80-derived kernel path asserts
        # `not is_split_kv` unconditionally (SplitKV not implemented there).
        # Force num_splits=1 so this heuristic can't hand it something > 1
        # for realistic batch/kv-length shapes, independent of paging.
        return 1
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    # Shearing makes split/combine overhead more visible. Multi-tile causal
    # prefill saturates around 64 CTAs. Batch-1 decode at very long context is
    # memory-bound and uses a TP-specific cap measured through 1M KV tokens.
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
        else:
            max_splits = 128
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


def inkling_fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 32,
    out: torch.Tensor | None = None,
    sfq: torch.Tensor | None = None,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged varlen FA4 over the bound K/V cache with the Inkling relative bias.

    ``q`` is ``(num_tokens, num_heads, head_dim)``; ``key_cache`` / ``value_cache``
    are the paged caches ``(num_blocks, block_size, num_kv_heads, head_dim)``;
    ``block_table`` is the per-request page table and ``cache_seqlens`` the
    per-request KV lengths (``seqused_k``). ``rel_logits`` is
    ``(num_tokens, num_heads, rel_extent)``.

    The bias uses tml-fa4's sheared relative-bias layout.

    ``sfq``/``sfk``/``sfv`` are optional ``float8_e8m0fnu`` per-32-element
    block scales (see ``quantize_q_to_fp8_blockscaled``/
    ``uniform_ue8m0_block_scale``). When ``key_cache``/``value_cache`` are
    fp8 (``--kv-cache-dtype fp8_e4m3``), FA4's non-blockscaled path asserts
    ``q.dtype == k.dtype == v.dtype`` and fails since Q stays bf16 -- passing
    ``sfq``/``sfk``/``sfv`` selects FA4's blockscaled MMA path instead, which
    allows Q (dynamically quantized to e4m3 by the caller) and the already-
    fp8 K/V cache to interoperate.
    """
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    # cute uses (None, None) to mean "no window".
    cute_window = (None, None) if window_size == (-1, -1) else window_size

    ret = flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=cute_window,
        num_splits=num_splits,
        return_lse=False,
        out=out,
        rel_bias=rel_logits.contiguous(),
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
    )
    if isinstance(ret, tuple):
        return ret[0]
    return ret
