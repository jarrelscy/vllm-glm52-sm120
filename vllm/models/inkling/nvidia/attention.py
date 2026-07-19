# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from ..configs import InklingModelConfig
from .layernorm import InklingRMSNorm
from .ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
    quantize_q_to_fp8_blockscaled,
    uniform_ue8m0_block_scale,
)
from .ops.fa4_warmup import InklingFA4WarmupConfig, register_fa4_warmup
from .ops.qkvr_prep import fused_qkvr_prep
from .ops.triton_decode_attention import triton_rel_decode_attention
from .ops.triton_prefill_attention import triton_rel_prefill_attention

# Task #124: the FA4 SM120 score-mod kernel costs a flat ~380us/layer at
# decode (128x128 tiles forced by rel_bias + pack-GQA grid of ~num_kv_heads
# CTAs + no split-KV on SM120). The Triton split-KV decode kernel replaces it
# for pure-decode batches. Opt-in while gates run; escape hatch mirrors the
# fused-MoE pattern.
# Split-KV Triton decode attention (default on; FA4's SM120 decode path is a
# serial KV scan). Set INKLING_DISABLE_TRITON_DECODE_ATTN=1 to fall back.
_TRITON_DECODE_ATTN = (
    os.environ.get("INKLING_DISABLE_TRITON_DECODE_ATTN", "0") != "1"
    and not os.path.exists("/tmp/INKLING_DISABLE_TRITON_DECODE_ATTN")
)
# Varlen Triton prefill attention (task #123). FA4's SM120 rel_bias path
# scans the FULL context even on sliding-window layers (55/66 layers have
# window 512): measured 3.2-39ms/layer-chunk vs 0.24-3.2ms for the Triton
# kernel (8.8x global, up to 155x SWA at 16K). bf16 KV only; the
# fp8-blockscaled path stays on FA4. Escape hatch mirrors decode.
# The env var may not reach mp-spawn workers on this fork, so a filesystem
# sentinel (filesystem-based because this fork's spawn-based multiproc
# executor wipes worker env vars) also disables it.
_TRITON_PREFILL_ATTN = (
    os.environ.get("INKLING_DISABLE_TRITON_PREFILL_ATTN", "0") != "1"
    and not os.path.exists("/tmp/INKLING_DISABLE_TRITON_PREFILL_ATTN")
)
# Uniform small-query batches (spec-decode verify / MTP draft passes) routed
# to the split-KV decode kernel via staggered virtual sequences (task #111).
# Escape hatch mirrors the two above.
_TRITON_VERIFY_ATTN = (
    os.environ.get("INKLING_DISABLE_TRITON_VERIFY_ATTN", "0") != "1"
    and not os.path.exists("/tmp/INKLING_DISABLE_TRITON_VERIFY_ATTN")
)
# Task #125 diagnostic: when /tmp/INKLING_ATTN_XCHECK exists, every Triton
# prefill-attention call is cross-checked against the FA4 reference on the
# same live inputs; the first divergent call's inputs are dumped to
# /tmp/inkling_attn_xcheck_rank<r>.pt as an offline reproducer. The offline
# kernel test passes (9/9), so the bug must be in what E2E feeds it.
_ATTN_XCHECK = os.path.exists("/tmp/INKLING_ATTN_XCHECK")
_ATTN_XCHECK_DUMPED = False
from .sconv_swa_attn import _K, _V, InklingConvState, InklingSconvMetadata
from .short_conv import InklingShortConv


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


class RelLogitsProj(nn.Module):
    """Project the per-head relative branch ``r`` to per-distance logits."""

    def __init__(self, d_rel: int, rel_extent: int) -> None:
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)

    def forward(self, r_out: torch.Tensor) -> torch.Tensor:
        # r_out: (T, num_heads, d_rel) -> (T, num_heads, rel_extent)
        return torch.einsum("thd,de->the", r_out, self.proj)


class InklingAttention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        config: InklingModelConfig,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rel_extent: int,
        local_extent: int,
        is_local: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        conv_owner: InklingConvState,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.d_rel = config.d_rel
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        # q/k are per-head RMS-normed (unit norm), so Inkling scales by 1/head_dim.
        self.scaling = 1.0 / head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_total_heads = num_heads
        self.num_total_kv_heads = num_kv_heads
        assert self.num_total_heads % tp_size == 0
        self.num_heads = self.num_total_heads // tp_size
        if self.num_total_kv_heads >= tp_size:
            assert self.num_total_kv_heads % tp_size == 0
        else:
            assert tp_size % self.num_total_kv_heads == 0
        self.num_kv_heads = max(1, self.num_total_kv_heads // tp_size)
        # When tp_size > num_kv_heads the K/V projections are padded up to
        # tp_size heads so each rank gets at least one (GQA replication).
        kv_total_for_sizing = max(self.num_total_kv_heads, tp_size)

        self.qkvr = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[
                head_dim * self.num_total_heads,
                head_dim * kv_total_for_sizing,
                head_dim * kv_total_for_sizing,
                self.d_rel * self.num_total_heads,
            ],
            bias=config.q_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            input_size=head_dim * self.num_total_heads,
            output_size=config.hidden_size,
            bias=config.o_bias,
            quant_config=quant_config,
            # reduce_results=False: the partial output is all-reduced below
            # (one-shot custom AR) so the attention-output sconv can run on the
            # full hidden width fused with the residual add + rmsnorm.
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_extent = local_extent if is_local else rel_extent
        self.local_extent = local_extent if is_local else None
        self.rel_logits_proj = RelLogitsProj(self.d_rel, self.rel_extent)
        self.q_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)

        # Short convolution on the K/V streams (per-head-width, TP sharded),
        # applied after the qkvr projection and before q/k norm.
        kv_conv_dim = self.num_kv_heads * head_dim
        self.conv_owner = conv_owner
        self.k_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_K
        )
        self.v_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_V
        )

        # FA4 left/right window; right=0 keeps it causal. local_extent-1 mirrors
        # the source (sliding_window_size - 1).
        self.window_size: tuple[int, int] = (
            (local_extent - 1, 0) if is_local else (-1, -1)
        )
        # Static per-layer-type KV length bound for the split heuristic: local
        # layers never see more than the sliding window.
        vllm_config = get_current_vllm_config()
        self._max_kv_len = (
            local_extent if is_local else vllm_config.model_config.max_model_len
        )

        # ---- KV-cache wiring (reuse FlashAttentionBackend for metadata) ----
        cache_config = vllm_config.cache_config
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.register_buffer("k_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("v_scale", torch.ones((), dtype=torch.float32))
        # Task #106: fp8 KV cache needs FA4's blockscaled MMA path (Q stays
        # bf16, so the non-blockscaled path's `q.dtype == k.dtype == v.dtype`
        # assert fails once k/v are fp8). See ops/fa4_rel_attention.py for the
        # sfq/sfk/sfv wiring and its known first-pass limitations (coarse,
        # per-tensor k_scale/v_scale rather than true per-block calibration).
        # Detect fp8 KV mode off the cache-dtype *string*, matching the
        # existing convention in triton_attn.py/flashinfer.py
        # (`kv_cache_dtype.startswith("fp8")`) -- not the resolved torch
        # dtype: kv_cache_dtype_str_to_dtype("fp8_e4m3", ...) resolves to
        # torch.uint8 (fp8 KV is stored as packed-byte buffers with a
        # separate scale, not as literal torch.float8_e4m3fn tensors), so a
        # torch-dtype comparison here was always False and silently skipped
        # the Q-quantization branch below, leaving Q in bf16 against a
        # uint8-backed physical K/V cache.
        self.kv_cache_is_fp8blockscaled = self.kv_cache_dtype.startswith("fp8")
        self._sfk_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._sfv_cache: dict[tuple[int, ...], torch.Tensor] = {}

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

        register_fa4_warmup(
            InklingFA4WarmupConfig(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rel_extent=self.rel_extent,
                window_size=self.window_size,
                is_local=self.is_local,
                max_kv_len=self._max_kv_len,
                dtype=vllm_config.model_config.dtype,
                kv_dtype=self.kv_cache_torch_dtype,
                block_size=vllm_config.cache_config.block_size,
                max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
            )
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.is_local:
            assert self.local_extent is not None
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.local_extent,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def _split_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        kv_cache = self.kv_cache
        if self.kv_cache_is_fp8blockscaled:
            # vLLM allocates fp8 KV caches as packed-byte uint8 buffers (see
            # kv_cache_dtype_str_to_dtype), but every consumer of this split
            # needs real fp8 element semantics:
            #  - the fused_qkvr_prep Triton kernels `tl.store` float K/V
            #    values into these pointers; through a uint8 pointer that is
            #    an integer cast (silent corruption), through an e4m3 pointer
            #    it is a correct float->fp8 conversion;
            #  - tml_fa4's blockscaled path hard-asserts
            #    q.dtype == k.dtype == v.dtype == float8_e4m3fn
            #    (interface.py:370/376-377).
            # Same storage-vs-kernel dtype-view convention as
            # deepseek_v32/nvidia/attention.py:294-296,510. uint8 and
            # float8_e4m3fn are both 1 byte, so this view is layout-preserving.
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_dim, dim=-1
        )
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkvr, _ = self.qkvr(hidden_states)

        attn_metadata = get_forward_context().attn_metadata
        attn_output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
        if not isinstance(attn_metadata, dict):
            attn_output.zero_()
        else:
            conv_meta = attn_metadata[self.conv_owner.prefix]
            md = attn_metadata[self.prefix]
            assert isinstance(conv_meta, InklingSconvMetadata)
            fa_md = cast(FlashAttentionMetadata, md)
            assert self.kv_cache.numel() > 0
            assert self.conv_owner.kv_cache.numel() > 0
            # One launch: K/V sconv (conv-cache insert + conv + residual),
            # Q/K per-head rmsnorm, and the attention KV-cache write. K/V are
            # consumed via the KV cache; only normed q is materialized.
            key_cache, value_cache = self._split_kv_cache()
            off_k, _ = self.conv_owner.stream_ranges[_K]
            off_v, _ = self.conv_owner.stream_ranges[_V]
            q, rel_logits = fused_qkvr_prep(
                qkvr,
                self.k_sconv.weight.squeeze(1),
                self.v_sconv.weight.squeeze(1),
                self.q_norm.weight,
                self.k_norm.weight,
                self.rel_logits_proj.proj,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.d_rel,
                self.conv_owner.kv_cache,
                key_cache,
                value_cache,
                positions,
                conv_meta.block_table,
                conv_meta.seq_idx,
                conv_meta.slot_mapping,
                conv_meta.query_start,
                fa_md.slot_mapping,
                off_k,
                off_v,
                self.conv_owner.block_size,
                log_scaling if not self.is_local else None,
            )
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            self._attention(q, rel_logits, attn_output)

        flat = attn_output.view(num_tokens, -1)
        output, _ = self.wo_ud(flat)
        return output

    def _attention(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        md = cast(FlashAttentionMetadata, attn_metadata[self.prefix])

        nt = md.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache()

        # Pure-decode batches (one query token per request): use the Triton
        # split-KV kernel (task #124) -- FA4's SM120 score-mod path is a
        # serial KV scan here (~380us at 272 KV tokens, linear in KV length).
        # bf16 KV only; the fp8-blockscaled path stays on FA4.
        if (
            _TRITON_DECODE_ATTN
            and md.max_query_len == 1
            and not self.kv_cache_is_fp8blockscaled
        ):
            triton_rel_decode_attention(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                rel_logits=rel_logits[:nt],
                softmax_scale=self.scaling,
                rel_extent=self.rel_extent,
                window_left=self.window_size[0] if self.is_local else -1,
                num_splits=8 if self.is_local else 64,
                out=output[:nt],
            )
            return

        # Uniform small-query batches (spec-decode verify at qlen=ns+1, MTP
        # draft window passes, tiny final prefill chunks): the varlen prefill
        # kernel below scans the WHOLE context with one CTA per (m-block,
        # head) -- ~35ms/full-layer at 512K, which made MTP decode
        # depth-linear (task #111: 466ms/round at 512K vs 52ms short-ctx).
        # Reuse the split-KV decode kernel instead, treating each query row
        # as its own virtual sequence over the shared block table: row j of
        # request i gets kv_len = seq_len_i - (qlen-1) + j, so the kernel's
        # dist = (kv_len-1) - js and window lo = kv_len-1-window_left equal
        # the row's true position math. Exact for ANY uniform batch (K/V for
        # all nt tokens are published to the cache before attention).
        if (
            _TRITON_VERIFY_ATTN
            and 1 < md.max_query_len <= 8
            and nt == md.max_query_len * md.seq_lens.shape[0]
            and not self.kv_cache_is_fp8blockscaled
        ):
            qlen = md.max_query_len
            sk = (
                md.seq_lens[:, None]
                - (qlen - 1)
                + torch.arange(
                    qlen, device=md.seq_lens.device, dtype=md.seq_lens.dtype
                )[None, :]
            ).view(-1)
            triton_rel_decode_attention(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table.repeat_interleave(qlen, dim=0),
                cache_seqlens=sk,
                rel_logits=rel_logits[:nt],
                softmax_scale=self.scaling,
                rel_extent=self.rel_extent,
                window_left=self.window_size[0] if self.is_local else -1,
                num_splits=8 if self.is_local else 64,
                out=output[:nt],
            )
            return

        # Prefill / mixed batches: varlen Triton kernel (task #123). Handles
        # qlen=1 requests inside a mixed batch too; pure-decode batches stay
        # on the split-KV kernel above (better parallelism at qlen=1).
        if _TRITON_PREFILL_ATTN and not self.kv_cache_is_fp8blockscaled:
            triton_rel_prefill_attention(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table,
                cu_seqlens_q=md.query_start_loc,
                cache_seqlens=md.seq_lens,
                max_seqlen_q=md.max_query_len,
                rel_logits=rel_logits[:nt],
                softmax_scale=self.scaling,
                rel_extent=self.rel_extent,
                window_left=self.window_size[0] if self.is_local else -1,
                out=output[:nt],
            )
            if _ATTN_XCHECK:
                self._xcheck_prefill(q, rel_logits, output, md, nt,
                                     key_cache, value_cache)
            return

        max_seqlen_q = bucket_max_seqlen_q(md.max_query_len)
        num_splits = inkling_fa4_num_splits(
            is_local=self.is_local,
            batch_size=md.seq_lens.shape[0],
            max_query_len=max_seqlen_q,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_kv_len=self._max_kv_len,
        )
        q_nt = q[:nt]
        sfq = sfk = sfv = None
        if self.kv_cache_is_fp8blockscaled:
            # key_cache/value_cache are already the physical fp8 KV cache
            # tensors; Q must be dynamically quantized to match (FA4 has no
            # mode where Q stays bf16 while K/V are fp8 -- see
            # ops/fa4_rel_attention.py docstrings for the full contract and
            # the first-pass caveat on sfk/sfv precision).
            q_nt, sfq = quantize_q_to_fp8_blockscaled(q_nt)
            sfk = uniform_ue8m0_block_scale(
                self.k_scale, key_cache, cache=self._sfk_cache
            )
            sfv = uniform_ue8m0_block_scale(
                self.v_scale, value_cache, cache=self._sfv_cache
            )
        inkling_fa4_rel_attention(
            q_nt,
            key_cache,
            value_cache,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scaling,
            causal=True,
            window_size=self.window_size,
            rel_extent=self.rel_extent,
            rel_logits=rel_logits[:nt],
            num_splits=num_splits,
            out=output[:nt],
            sfq=sfq,
            sfk=sfk,
            sfv=sfv,
        )

    def _xcheck_prefill(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
        md: FlashAttentionMetadata,
        nt: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        """Task #125 diagnostic (sentinel /tmp/INKLING_ATTN_XCHECK): re-run
        the FA4 reference on the exact inputs the Triton prefill kernel just
        consumed and report the divergence. bf16-KV only (the Triton branch
        is bf16-only). Dumps the first divergent call's inputs per rank."""
        global _ATTN_XCHECK_DUMPED
        from vllm.logger import init_logger
        log = init_logger(__name__)
        try:
            ref = torch.empty_like(output[:nt])
            max_seqlen_q = bucket_max_seqlen_q(md.max_query_len)
            num_splits = inkling_fa4_num_splits(
                is_local=self.is_local,
                batch_size=md.seq_lens.shape[0],
                max_query_len=max_seqlen_q,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                max_kv_len=self._max_kv_len,
            )
            inkling_fa4_rel_attention(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=self.scaling,
                causal=True,
                window_size=self.window_size,
                rel_extent=self.rel_extent,
                rel_logits=rel_logits[:nt],
                num_splits=num_splits,
                out=ref,
            )
            diff = (output[:nt].float() - ref.float()).abs()
            # RELATIVE threshold: activations run |x|~10-60 where one bf16
            # ulp is 0.0625-0.25, so an absolute cutoff flags pure rounding
            # (arbitrated vs fp32 truth 2026-07-19: both kernels identical
            # error vs truth, mutual diff exactly 1 ulp). Wrong-address
            # reads give O(1) relative error; ulp noise is ~0.008 relative.
            rel_err = diff / (ref.float().abs() + 1.0)
            md_ = rel_err.max().item()
            abs_ = diff.max().item()
            if md_ > 0.05:
                # (token, head) of the worst element
                flat_idx = (rel_err.view(nt, -1).max(dim=1).values
                            .argmax().item())
                log.error(
                    "ATTNXCHECK DIVERGE prefix=%s is_local=%s nt=%d "
                    "max_rel=%.4f max_abs=%.4f mean_abs=%.5f worst_tok=%d "
                    "seq_lens=%s qsl=%s max_q=%d",
                    self.prefix, self.is_local, nt, md_, abs_,
                    diff.mean().item(), flat_idx,
                    md.seq_lens.tolist()[:16],
                    md.query_start_loc.tolist()[:16], md.max_query_len,
                )
                if not _ATTN_XCHECK_DUMPED:
                    _ATTN_XCHECK_DUMPED = True
                    try:
                        from vllm.distributed import get_tensor_model_parallel_rank
                        rank = get_tensor_model_parallel_rank()
                    except Exception:  # noqa: BLE001
                        rank = 0
                    # The cache tensors are the FULL KV pool; dump only the
                    # blocks this batch references and remap the block table.
                    used = md.block_table.unique().clamp_min(0)
                    remap = torch.full(
                        (int(used.max().item()) + 1,), -1,
                        dtype=md.block_table.dtype,
                        device=md.block_table.device,
                    )
                    remap[used] = torch.arange(
                        used.numel(), dtype=md.block_table.dtype,
                        device=md.block_table.device,
                    )
                    bt_remap = remap[md.block_table.clamp_min(0)]
                    torch.save(
                        {
                            "prefix": self.prefix,
                            "is_local": self.is_local,
                            "q": q[:nt].cpu(),
                            "key_cache": key_cache[used].contiguous().cpu(),
                            "value_cache": value_cache[used].contiguous().cpu(),
                            "block_table": bt_remap.cpu(),
                            "seq_lens": md.seq_lens.cpu(),
                            "query_start_loc": md.query_start_loc.cpu(),
                            "max_query_len": md.max_query_len,
                            "rel_logits": rel_logits[:nt].cpu(),
                            "softmax_scale": self.scaling,
                            "rel_extent": self.rel_extent,
                            "window_left": (self.window_size[0]
                                            if self.is_local else -1),
                            "out_triton": output[:nt].cpu(),
                            "out_fa4": ref.cpu(),
                        },
                        f"/tmp/inkling_attn_xcheck_rank{rank}.pt",
                    )
                    log.error("ATTNXCHECK dumped repro to "
                              "/tmp/inkling_attn_xcheck_rank%d.pt", rank)
            else:
                log.warning(
                    "ATTNXCHECK ok prefix=%s is_local=%s nt=%d max_rel=%.4f "
                    "max_abs=%.4f",
                    self.prefix, self.is_local, nt, md_, abs_,
                )
        except Exception as exc:  # noqa: BLE001
            log.error("ATTNXCHECK failed to run: %r", exc)
