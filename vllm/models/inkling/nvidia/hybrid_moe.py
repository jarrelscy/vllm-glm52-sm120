# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 + AQLM hybrid routed-expert MoE method for Inkling (TP only).

Every routed-MoE layer keeps a small "hot" group of experts at higher
precision (NVFP4, or plain bf16 for the one layer with no NVFP4 base) and
compresses the rest ("cold") with AQLM (multi-codebook additive
quantization, group size 8 along the input dim, one fp16 codebook shared
by all experts of a projection/book). See ``aqlm_hybrid.py`` for the
checkpoint-config side and ``moe.py`` for how this is wired into
``InklingMoE``.

Checkpoint tensor layout per hybrid layer (``H`` = hidden_size, ``I`` =
intermediate_size, ``G`` = 8 the AQLM group dim; w13 out-rows are
INTERLEAVED gate/up ``[g0, u0, g1, u1, ...]``, w2 is not interleaved)::

    experts.hot_ids / experts.cold_ids           int32 [n_hot] / [n_cold]
    experts.{w13,w2}_hot_weight                  u8    NVFP4-packed (hot_format=nvfp4)
    experts.{w13,w2}_hot_weight.scale            f8e4m3 block scale
    experts.{w13,w2}_hot_weight.scale2           f32   [n_hot] per-expert scalar
    experts.{w13,w2}_hot_bf16                    bf16  (hot_format=bf16, no scale)
    experts.{w13,w2}_cold_codes.{b}              u8/i16 per-book AQLM codes
    experts.{w13,w2}_cold_codebook.{b}           f16   [entries_b, 8], REPLICATED
    experts.{w13,w2}_cold_scales                 f16   per-row (per out-channel) scale

TP sharding (T = tp_size, r = tp_rank; hidden_size is never sharded):
  w13 (gate_up): COLUMN-parallel over the 2I output rows. Each rank's slab
      is the de-interleaved [gate_slice ; up_slice] for its I/T columns
      (mirrors ``InklingMoE.load_expert_weight``'s existing per-expert
      NVFP4/bf16 de-interleave, generalized to a stacked hot/cold tensor).
  w2 (down): ROW-parallel over the I input dim -- no de-interleave needed.
  hot_ids/cold_ids, codebooks, and w2's per-row scales (indexed by the
  unsharded H dim) are REPLICATED. The routed output is a TP-partial sum;
  the MoE runner's late all-reduce combines it (this method sets no
  moe_kernel, matching the plain NVFP4/bf16 experts path).

v1 (this module): correctness-first. ``apply()`` always dequantizes the
routed experts and runs a dense per-expert grouped GEMM -- no fused CUDA
gemv/dequant kernel yet (that is deferred to the decode-speed pass, see
GLM-5.2's ``nvfp4_aqlm_hybrid.py`` for the intended design). This means
``apply()`` is NOT CUDA-graph-capture safe (it host-syncs on expert
counts); hybrid layers must run in eager mode until a graph-safe kernel
lands.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger

from ..aqlm_hybrid import HybridLayerInfo

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

_CODE_DTYPES = {"uint8": torch.uint8, "int16": torch.int16}
_G = 8  # AQLM group dim, fixed by InklingAqlmHybridConfig.group_size == 8.

# Fused quantized-gemv decode kernels (hybrid_moe_kernels.py). Escape hatch:
# INKLING_DISABLE_FUSED_DECODE=1 falls back to the dequant+bmm decode path.
try:
    import triton  # noqa: F401

    _HAS_FUSED_KERNELS = True
except ImportError:  # pragma: no cover - triton ships with vLLM
    _HAS_FUSED_KERNELS = False
# Filesystem sentinels mirror the env vars (env may not reach mp-spawn
# workers on this fork, which wipe worker subprocess environments).
_FUSED_DECODE_DISABLED = (
    os.environ.get("INKLING_DISABLE_FUSED_DECODE", "0") == "1"
    or os.path.exists("/tmp/INKLING_DISABLE_FUSED_DECODE")
)
_FUSED_PREFILL_DISABLED = (
    os.environ.get("INKLING_DISABLE_FUSED_PREFILL", "0") == "1"
    or os.path.exists("/tmp/INKLING_DISABLE_FUSED_PREFILL")
)
# Multi-token decode batches (spec-decode verify, concurrent seqs) route to
# the grouped kernel via a graph-safe block map instead of the per-slot gemv,
# which pays num_tokens x the single-token MoE weight traffic.
# DEFAULT DISABLED since task #137 (laneB): with the V2 gemv (see
# hybrid_moe_kernels._GEMV_V2) the per-slot gemv beats the grouped kernel at
# prod verify shapes -- router picks ~distinct experts per token, so expert
# runs are length ~1 and the grouped m-blocks amortize nothing while paying
# tl.dot on mostly-empty BM=16 tiles (microbench S=18: grouped 532 vs V2
# gemv ~340 us/layer-pair; server A/B ns=2: 48.9 -> 39.5 ms/round, -19%,
# acceptance unchanged). Set INKLING_DISABLE_DECODE_GROUPED=0 to re-enable
# (worth re-testing if max_num_seqs or top_k grows: expert-run lengths scale
# with S and the grouped path wins again once runs get long).
_DECODE_GROUPED_DISABLED = (
    os.environ.get("INKLING_DISABLE_DECODE_GROUPED", "1") == "1"
    or os.path.exists("/tmp/INKLING_DISABLE_DECODE_GROUPED")
)

# Empirically measured (not guessed) peak transient footprint of a single
# _compute_expert call (xe/h13/hact/ye below) -- continuously updated to the
# max seen so far via torch.cuda.max_memory_allocated() deltas, and
# subtracted from free memory before sizing dequant chunks. Attempts 14 and
# 15 both OOM'd *inside* _compute_expert even after the dequant-chunk budget
# was tightened (safety_margin 0.7 -> 0.5): that budget only ever accounted
# for chunk_w13/chunk_w2 (the dequant buffers), never this per-expert
# compute transient, which scales with routed-token count rather than
# weight size -- so retuning the dequant-side margin alone could not have
# fixed it regardless of the number picked.
_expert_compute_reserve_bytes = 0

# Below this token count we treat the call as the decode regime: the routed
# expert set is tiny (<=top_k unique), the per-expert compute transient is
# strictly smaller than any prefill/profile_run batch (it scales with routed
# tokens), and everything fits in a single dequant chunk. In that regime we
# skip the reserve-measurement d2h syncs (memory_allocated/reset_peak/
# max_memory_allocated -- OOM-hunt residue that otherwise fires ~top_k*layers
# times PER TOKEN and dominates decode), the _pick_chunk_size mem_get_info
# sync, and the per-chunk empty_cache() syncs. The reserve is learned once at
# profile_run (max-batch, worst case) which upper-bounds every decode call, so
# freezing measurement below this threshold is safe. Prefill/profile (mnbt up
# to 2048) stay above it and keep the full OOM-safe chunking path.
_RESERVE_MEASURE_MIN_TOKENS = 256

# Fixed slot-chunk width for the graph-safe decode path (_apply_decode_graphsafe).
# The decode path processes S = num_tokens*top_k routed slots in chunks of this
# many at a time so no more than this many dequantized expert weights are live
# at once. The loop trip count is ceil(S/this) -- a function of S (hence of the
# captured batch size) ONLY, never of the data -- so it unrolls to a static
# number of iterations at CUDA-graph capture time. Small: at decode S is tiny.
_DECODE_SLOT_CHUNK = 8


def _pick_chunk_size(
    n_total: int,
    bytes_per_expert: int,
    device: torch.device,
    default_chunk: int = 16,
    safety_margin: float = 0.5,
) -> int:
    """Chunk size for the expert-dim dequant loops in ``apply()``.

    ``default_chunk`` (16) was tuned against the ~11.5 GiB of headroom left
    once ``profile_run()`` has loaded TP4 weights, but other callers into
    this same ``apply()`` path (kernel_warmup's dummy_run, real decode)
    leave different -- often less -- headroom at call time: a real-checkpoint
    smoke test OOM'd at exactly this spot during kernel_warmup, at the same
    checkpoint that passes fine under profile_run(), because by then weights
    + KV cache + warmup activations had already consumed most of the
    gpu_memory_utilization budget. Querying live free memory instead of
    assuming any one caller's headroom makes this correct regardless of
    which call site got here.
    """
    if n_total <= 0 or bytes_per_expert <= 0:
        return default_chunk
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_bytes = max(0, free_bytes - _expert_compute_reserve_bytes)
    max_fit = int(free_bytes * safety_margin) // bytes_per_expert
    return max(1, min(default_chunk, n_total, max_fit))


def _noop_loader(param: Parameter, loaded: torch.Tensor) -> None:
    param.data.copy_(loaded.to(param.data.device))


def _gateup_deinterleave_shard_loader(rank: int, world: int, axis: int):
    """Loader for a w13 (gate_up) tensor whose ``axis`` holds the full 2I
    interleaved rows ``[g0, u0, g1, u1, ...]`` on disk. Produces this rank's
    de-interleaved ``[gate_slice ; up_slice]`` shard (I/world rows each)."""

    def load(param: Parameter, loaded: torch.Tensor) -> None:
        ish = param.data.shape[axis] // 2
        # Narrow to this rank's contiguous slab BEFORE moving to device: the
        # full untrimmed tensor may be an mmap'd checkpoint view (see the
        # plain-NVFP4 load_expert_weight, same rationale).
        slab = loaded.narrow(axis, rank * 2 * ish, 2 * ish).to(param.data.device)
        idx_even = [slice(None)] * slab.dim()
        idx_odd = [slice(None)] * slab.dim()
        idx_even[axis] = slice(0, None, 2)
        idx_odd[axis] = slice(1, None, 2)
        gate = slab[tuple(idx_even)]
        up = slab[tuple(idx_odd)]
        idx_g = [slice(None)] * param.data.dim()
        idx_u = [slice(None)] * param.data.dim()
        idx_g[axis] = slice(0, ish)
        idx_u[axis] = slice(ish, 2 * ish)
        param.data[tuple(idx_g)] = gate
        param.data[tuple(idx_u)] = up

    return load


def _row_shard_loader(rank: int, world: int, axis: int):
    """Loader for a w2 (down) tensor: plain TP shard along ``axis`` (the
    intermediate/I dim), no interleaving."""

    def load(param: Parameter, loaded: torch.Tensor) -> None:
        full = loaded.shape[axis]
        assert full % world == 0, f"{full} not divisible by TP={world}"
        shard = full // world
        idx = [slice(None)] * loaded.dim()
        idx[axis] = slice(rank * shard, (rank + 1) * shard)
        param.data.copy_(loaded[tuple(idx)].to(param.data.device))

    return load


class InklingHybridExpertsMoEMethod(FusedMoEMethodBase):
    """One MoE layer's NVFP4/bf16-hot + AQLM-cold routed experts (TP only,
    EP must be 1: Inkling's hybrid checkpoint is only intended for the TP4
    deployment target, mirroring GLM-5.2's ``TPHybridExpertsMoEMethod``)."""

    def __init__(
        self,
        moe_config,
        layer_id: int,
        info: HybridLayerInfo,
        w13_book_entries: list[int],
        w2_book_entries: list[int],
        w13_code_dtypes: list[str],
        w2_code_dtypes: list[str],
        *,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__(moe_config)
        if moe_config.moe_parallel_config.ep_size > 1:
            raise NotImplementedError(
                "Inkling hybrid MoE layers support TP only (ep_size must be 1)"
            )
        self.layer_id = layer_id
        self.info = info
        self.w13_book_entries = w13_book_entries
        self.w2_book_entries = w2_book_entries
        self.w13_code_dtypes = [_CODE_DTYPES[d] for d in w13_code_dtypes]
        self.w2_code_dtypes = [_CODE_DTYPES[d] for d in w2_code_dtypes]
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        n_hot, n_cold = self.info.n_hot, self.info.n_cold
        assert n_hot + n_cold == num_experts
        h = hidden_size
        ish = intermediate_size_per_partition
        self._hidden_size = h
        self._ish = ish
        r, t = self.tp_rank, self.tp_size

        def make(name: str, shape: tuple[int, ...], dtype: torch.dtype, loader):
            p = Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, {"weight_loader": loader})

        rep = _noop_loader
        gu = lambda axis: _gateup_deinterleave_shard_loader(r, t, axis)  # noqa: E731
        rowk = lambda axis: _row_shard_loader(r, t, axis)  # noqa: E731

        make("hot_ids", (n_hot,), torch.int32, rep)
        make("cold_ids", (n_cold,), torch.int32, rep)

        if n_hot > 0:
            if self.info.hot_format == "nvfp4":
                make("w13_hot_weight", (n_hot, 2 * ish, h // 2), torch.uint8, gu(1))
                make(
                    "w13_hot_weight_scale",
                    (n_hot, 2 * ish, h // 16),
                    torch.float8_e4m3fn,
                    gu(1),
                )
                make("w13_hot_weight_scale2", (n_hot,), torch.float32, rep)
                make("w2_hot_weight", (n_hot, h, ish // 2), torch.uint8, rowk(2))
                make(
                    "w2_hot_weight_scale",
                    (n_hot, h, ish // 16),
                    torch.float8_e4m3fn,
                    rowk(2),
                )
                make("w2_hot_weight_scale2", (n_hot,), torch.float32, rep)
            else:
                assert self.info.hot_format == "bf16"
                make("w13_hot_bf16", (n_hot, 2 * ish, h), params_dtype, gu(1))
                make("w2_hot_bf16", (n_hot, h, ish), params_dtype, rowk(2))

        for b, (entries, dt) in enumerate(
            zip(self.w13_book_entries, self.w13_code_dtypes)
        ):
            make(f"w13_cold_codes_{b}", (n_cold, 2 * ish, h // _G), dt, gu(1))
            make(f"w13_cold_codebook_{b}", (entries, _G), torch.float16, rep)
        make("w13_cold_scales", (n_cold, 2 * ish), torch.float16, gu(1))

        for b, (entries, dt) in enumerate(
            zip(self.w2_book_entries, self.w2_code_dtypes)
        ):
            make(f"w2_cold_codes_{b}", (n_cold, h, ish // _G), dt, rowk(2))
            make(f"w2_cold_codebook_{b}", (entries, _G), torch.float16, rep)
        make("w2_cold_scales", (n_cold, h), torch.float16, rep)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        num_experts = layer.global_num_experts
        device = layer.hot_ids.device
        hot_lookup = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        cold_lookup = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        hot_lookup[layer.hot_ids.long()] = torch.arange(
            self.info.n_hot, dtype=torch.int32, device=device
        )
        cold_lookup[layer.cold_ids.long()] = torch.arange(
            self.info.n_cold, dtype=torch.int32, device=device
        )
        layer._hybrid_hot_lookup = hot_lookup
        layer._hybrid_cold_lookup = cold_lookup
        # CPU mirrors of the static index maps: apply() indexes these per expert
        # per chunk, and `int(gpu_tensor[e])` is a d2h sync each time (thousands
        # per layer). The maps never change after load, so read them off CPU.
        layer._hybrid_hot_lookup_cpu = hot_lookup.tolist()
        layer._hybrid_cold_lookup_cpu = cold_lookup.tolist()
        # Pre-warm the NVFP4 dequant LUT on this device before CUDA-graph capture
        # so the captured decode path never triggers a CPU->CUDA copy.
        _nvfp4_lut(device)

    def get_fused_moe_quant_config(self, layer: "RoutedExperts"):
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    def _dequant_hot(
        self,
        layer: "RoutedExperts",
        proj: str,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Local hot experts of ``proj`` selected by ``idx`` -> bf16 [n, out, in].

        ``idx`` is a 1-D long tensor of local hot indices (fancy indexing, so
        the selection need not be contiguous). apply() passes exactly the
        experts routed to this call: at decode that is only the <=top_k unique
        experts, not a whole 16-wide chunk per touched expert. ``n_hot`` is
        per-layer (4 for early/mid layers, up to 127 for late 53-64ish layers
        that compress poorly under AQLM), so apply() still batches ``idx`` to
        bound transient memory in the many-experts-active (prefill) case.
        """
        if self.info.hot_format == "bf16":
            return getattr(layer, f"{proj}_hot_bf16")[idx].to(torch.bfloat16)
        packed = getattr(layer, f"{proj}_hot_weight")[idx]
        scale = getattr(layer, f"{proj}_hot_weight_scale")[idx]
        scale2 = getattr(layer, f"{proj}_hot_weight_scale2")[idx]
        return _dequant_nvfp4(packed, scale, scale2)

    def _dequant_cold(
        self,
        layer: "RoutedExperts",
        proj: str,
        idx: torch.Tensor,
        graph_safe: bool = False,
    ) -> torch.Tensor:
        """Local cold experts of ``proj`` selected by ``idx`` -> bf16 [n, out, in].

        ``graph_safe`` routes to a host-sync-free AQLM dequant (unconditional
        power-of-two codebook masking instead of a data-dependent ``.min()<0``
        check) for the CUDA-graph decode path.
        """
        books = self.w13_book_entries if proj == "w13" else self.w2_book_entries
        n_books = len(books)
        codes = [
            getattr(layer, f"{proj}_cold_codes_{b}")[idx] for b in range(n_books)
        ]
        cbs = [getattr(layer, f"{proj}_cold_codebook_{b}") for b in range(n_books)]
        scales = getattr(layer, f"{proj}_cold_scales")[idx]
        return _dequant_aqlm(codes, cbs, scales, graph_safe=graph_safe)

    def _hot_kernel_args(self, layer: "RoutedExperts", proj: str):
        """(hot_mode, packed, scale, scale2) for fused_hybrid_gemv."""
        if self.info.n_hot == 0:
            return 0, None, None, None
        if self.info.hot_format == "nvfp4":
            return (
                1,
                getattr(layer, f"{proj}_hot_weight"),
                getattr(layer, f"{proj}_hot_weight_scale"),
                getattr(layer, f"{proj}_hot_weight_scale2"),
            )
        return 2, getattr(layer, f"{proj}_hot_bf16"), None, None

    def _cold_kernel_args(self, layer: "RoutedExperts", proj: str):
        """(codes list, codebooks list, scales) for fused_hybrid_gemv."""
        books = self.w13_book_entries if proj == "w13" else self.w2_book_entries
        n_books = len(books)
        codes = [getattr(layer, f"{proj}_cold_codes_{b}") for b in range(n_books)]
        cbs = [getattr(layer, f"{proj}_cold_codebook_{b}") for b in range(n_books)]
        return codes, cbs, getattr(layer, f"{proj}_cold_scales")

    def _apply_decode_fused(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode MoE via fused quantized-gemv kernels (hybrid_moe_kernels).

        Reads NVFP4/AQLM codes directly and decodes in-register — never
        materializes bf16 expert weights. ~52x faster per layer than the
        dequant+bmm fallback below (0.25 vs 13.2 ms at real decode shapes)
        because that path's cost IS the dequant materialization traffic.
        Same graph-safety contract as _apply_decode_graphsafe (fixed shapes,
        no host syncs); slightly MORE accurate than the fallback (weights
        stay fp32 in-register instead of rounding to bf16).
        """
        from vllm.models.inkling.nvidia.hybrid_moe_kernels import (
            fused_hybrid_gemv,
        )

        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]

        flat_ids = topk_ids.reshape(-1).long()                 # [S]
        flat_w = topk_weights.reshape(-1).to(torch.float32)    # [S]
        slot_tok = torch.arange(
            num_tokens, device=x.device
        ).repeat_interleave(top_k)                             # [S]

        hot_raw = layer._hybrid_hot_lookup[flat_ids]           # [S] int32
        is_hot = (hot_raw >= 0).to(torch.int32)
        hot_idx = hot_raw.clamp_min(0)
        cold_idx = layer._hybrid_cold_lookup[flat_ids].clamp_min(0)

        x_slots = x.to(torch.bfloat16)[slot_tok].contiguous()  # [S, hidden]

        ish = self._ish
        m13, hp13, hs13, hs2_13 = self._hot_kernel_args(layer, "w13")
        c13, cb13, cs13 = self._cold_kernel_args(layer, "w13")
        h13 = fused_hybrid_gemv(
            x_slots, is_hot, hot_idx, cold_idx,
            m13, hp13, hs13, hs2_13, c13, cb13, cs13,
            N=2 * ish, K=hidden,
        )                                                      # [S, 2*ish] fp32
        act = _silu_and_mul(h13).to(torch.bfloat16).contiguous()  # [S, ish]

        m2, hp2, hs2_, hs22 = self._hot_kernel_args(layer, "w2")
        c2, cb2, cs2 = self._cold_kernel_args(layer, "w2")
        ye = fused_hybrid_gemv(
            act, is_hot, hot_idx, cold_idx,
            m2, hp2, hs2_, hs22, c2, cb2, cs2,
            N=hidden, K=ish,
        )                                                      # [S, hidden] fp32
        ye = ye * flat_w.unsqueeze(-1)

        out = torch.zeros(
            num_tokens, hidden, dtype=torch.float32, device=x.device
        )
        out.index_add_(0, slot_tok, ye)
        return out.to(x.dtype)

    def _apply_prefill_fused(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill/profile MoE via the fused grouped-GEMM kernel.

        Sorts slots by expert, builds the expert->m-block map with device ops
        (one host sync for the grid size — fine in the eager prefill regime),
        and runs both projections through _hybrid_grouped_gemm_kernel, which
        decodes NVFP4/AQLM codes in-register. Replaces the dequant-materialize
        + per-expert-GEMM loop below, whose cost at prefill is dominated by
        writing+reading every active expert's full bf16 weights per chunk
        (~14 GiB/layer at mnbt 2048) plus the per-chunk empty_cache() syncs.
        Peak transient here is just the [S, N] fp32 activations (~0.3 GiB at
        mnbt 2048), far below the dequant path's chunked budget.
        """
        from vllm.models.inkling.nvidia.hybrid_moe_kernels import (
            fused_hybrid_grouped_gemm,
        )

        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        S = num_tokens * top_k
        dev = x.device
        n_experts = layer._hybrid_hot_lookup.shape[0]
        block_m = 64  # must equal the kernel's block_m (map is built per-tile)

        flat_ids = topk_ids.reshape(-1).long()
        order = torch.argsort(flat_ids)
        sorted_tok = (order // top_k).to(torch.int32)
        sorted_w = topk_weights.reshape(-1).float()[order]

        counts = torch.bincount(flat_ids, minlength=n_experts)  # [E]
        slot_off = torch.cumsum(counts, 0) - counts             # exclusive
        nblk = (counts + block_m - 1) // block_m
        total_blocks = int(nblk.sum().item())                   # one host sync
        block_expert = torch.repeat_interleave(
            torch.arange(n_experts, device=dev), nblk
        )
        within = (
            torch.arange(total_blocks, device=dev)
            - (torch.cumsum(nblk, 0) - nblk)[block_expert]
        )
        block_slot0 = (slot_off[block_expert] + within * block_m).to(torch.int32)
        block_mlen = torch.clamp(
            counts[block_expert] - within * block_m, max=block_m
        ).to(torch.int32)
        block_expert = block_expert.to(torch.int32)

        x_bf = x.to(torch.bfloat16).contiguous()
        ish = self._ish
        hot_lut = layer._hybrid_hot_lookup
        cold_lut = layer._hybrid_cold_lookup

        m13, hp13, hs13, hs2_13 = self._hot_kernel_args(layer, "w13")
        c13, cb13, cs13 = self._cold_kernel_args(layer, "w13")
        h13 = fused_hybrid_grouped_gemm(
            x_bf, sorted_tok, block_expert, block_slot0, block_mlen,
            hot_lut, cold_lut,
            m13, hp13, hs13, hs2_13, c13, cb13, cs13,
            S=S, N=2 * ish, K=hidden,
        )                                                       # [S, 2*ish]
        act = _silu_and_mul(h13).to(torch.bfloat16).contiguous()
        del h13

        ident = torch.arange(S, dtype=torch.int32, device=dev)
        m2, hp2, hs2a, hs2b = self._hot_kernel_args(layer, "w2")
        c2, cb2, cs2 = self._cold_kernel_args(layer, "w2")
        ye = fused_hybrid_grouped_gemm(
            act, ident, block_expert, block_slot0, block_mlen,
            hot_lut, cold_lut,
            m2, hp2, hs2a, hs2b, c2, cb2, cs2,
            S=S, N=hidden, K=ish,
        )                                                       # [S, hidden]
        ye *= sorted_w.unsqueeze(-1)
        out = torch.zeros(
            num_tokens, hidden, dtype=torch.float32, device=dev
        )
        out.index_add_(0, sorted_tok.long(), ye)
        return out.to(x.dtype)

    def _apply_decode_grouped(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-token decode-regime MoE via the grouped kernel, graph-safe.

        The per-slot gemv path reads each slot's full expert weights
        independently, so a spec-decode verify batch (num_tokens = ns+1 per
        seq) pays num_tokens x the single-token MoE weight traffic — v36
        profiling showed that one kernel at 78% of MTP-round GPU time.
        Sorting slots by expert amortizes each expert's weight read across
        every row routed to it, and the grouped kernel's tl.dot + uniform
        hot/cold branch replace the gemv's scalar FMA chains.

        Graph-safety (same contract as _apply_decode_graphsafe): grid and all
        shapes depend only on S = num_tokens*top_k. The block map launches one
        CANDIDATE m-block per sorted slot; only slots at a BLOCK_M boundary of
        their expert run carry mlen > 0, the rest no-op in-kernel. Boundaries
        are tensor VALUES, so a captured graph replays across steps.
        """
        from vllm.models.inkling.nvidia.hybrid_moe_kernels import (
            fused_hybrid_grouped_gemm,
        )

        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        S = num_tokens * top_k
        dev = x.device
        block_m = 16  # tl.dot minimum; decode expert runs are short

        flat_ids = topk_ids.reshape(-1).long()                  # [S]
        order = torch.argsort(flat_ids)                         # fixed shape
        sorted_ids = flat_ids[order]
        sorted_tok = (order // top_k).to(torch.int32)
        sorted_w = topk_weights.reshape(-1).float()[order]

        # Expert-run boundaries as values: run_start[i] is the first slot of
        # i's run (cummax over first-of-run indices), run_len broadcast from a
        # one-hot count at each run start. head slots every block_m within a
        # run own a block of min(block_m, remaining) rows; all others mlen=0.
        idx = torch.arange(S, device=dev)
        first = torch.ones(S, dtype=torch.bool, device=dev)
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
        run_start = torch.cummax(
            torch.where(first, idx, torch.zeros_like(idx)), 0
        ).values                                                # [S]
        within = idx - run_start
        run_len = torch.zeros_like(idx).index_add_(
            0, run_start, torch.ones_like(idx)
        )[run_start]                                            # [S]
        head = (within % block_m) == 0
        block_mlen = torch.where(
            head,
            torch.clamp(run_len - within, max=block_m),
            torch.zeros_like(run_len),
        ).to(torch.int32)
        block_expert = sorted_ids.to(torch.int32)
        block_slot0 = idx.to(torch.int32)

        x_bf = x.to(torch.bfloat16).contiguous()
        ish = self._ish
        hot_lut = layer._hybrid_hot_lookup
        cold_lut = layer._hybrid_cold_lookup

        from vllm.models.inkling.nvidia.hybrid_moe_kernels import (
            _GROUPED_SPLITK,
        )

        m13, hp13, hs13, hs2_13 = self._hot_kernel_args(layer, "w13")
        c13, cb13, cs13 = self._cold_kernel_args(layer, "w13")
        h13 = fused_hybrid_grouped_gemm(
            x_bf, sorted_tok, block_expert, block_slot0, block_mlen,
            hot_lut, cold_lut,
            m13, hp13, hs13, hs2_13, c13, cb13, cs13,
            S=S, N=2 * ish, K=hidden, block_m=block_m,
            split_k=_GROUPED_SPLITK,
        )                                                       # [S, 2*ish]
        act = _silu_and_mul(h13).to(torch.bfloat16).contiguous()

        ident = torch.arange(S, dtype=torch.int32, device=dev)
        m2, hp2, hs2a, hs2b = self._hot_kernel_args(layer, "w2")
        c2, cb2, cs2 = self._cold_kernel_args(layer, "w2")
        ye = fused_hybrid_grouped_gemm(
            act, ident, block_expert, block_slot0, block_mlen,
            hot_lut, cold_lut,
            m2, hp2, hs2a, hs2b, c2, cb2, cs2,
            S=S, N=hidden, K=ish, block_m=block_m,
            split_k=_GROUPED_SPLITK,
        )                                                       # [S, hidden]
        ye *= sorted_w.unsqueeze(-1)
        out = torch.zeros(
            num_tokens, hidden, dtype=torch.float32, device=dev
        )
        out.index_add_(0, sorted_tok.long(), ye)
        return out.to(x.dtype)

    def _apply_decode_graphsafe(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape, host-sync-free MoE for the decode regime.

        Every op keys off tensor VALUES at SHAPES that depend only on the batch
        (S = num_tokens*top_k), never on the data, so a CUDA graph captured at a
        given decode batch size replays across steps. This is the path that
        makes Inkling decode capturable (see the eager ``apply`` loop below for
        the prefill/profile version, which uses ``.unique()``/``.nonzero()`` and
        data-dependent Python trip counts and is NOT capturable).

        Design notes:
        - Routing gives each of the S slots a global expert id (``flat_ids``).
          We gather each slot's hot AND cold local index from the GPU lookup
          tables (indexed by ``flat_ids`` -- a fixed-shape gather, no
          ``.unique``), dequant BOTH formats, and select per slot by an
          ``is_hot`` mask. At decode S is tiny so the 2x dequant is cheap, and
          it avoids the data-dependent hot/cold partition that would otherwise
          need ``.nonzero()``.
        - Token indices come from ``arange().repeat_interleave`` (static), not
          ``mask.nonzero()``.
        - Weights are materialized ``_DECODE_SLOT_CHUNK`` slots at a time to
          bound transient memory; the loop trip count is ``ceil(S/chunk)`` -- a
          function of S only -- so it unrolls statically at capture.
        """
        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        S = num_tokens * top_k

        hot_lut = layer._hybrid_hot_lookup   # int32 [n_experts], -1 if not hot
        cold_lut = layer._hybrid_cold_lookup  # int32 [n_experts], -1 if not cold

        flat_ids = topk_ids.reshape(-1).long()                # [S]
        flat_w = topk_weights.reshape(-1).to(torch.float32)   # [S]
        slot_tok = torch.arange(
            num_tokens, device=x.device
        ).repeat_interleave(top_k)                             # [S]

        hot_raw = hot_lut[flat_ids]                            # [S] int32, -1=cold
        is_hot = (hot_raw >= 0).view(-1, 1, 1)                 # [S,1,1] bool
        hot_idx = hot_raw.clamp_min(0).long()                 # valid gather idx
        cold_idx = cold_lut[flat_ids].clamp_min(0).long()

        xe_all = x.to(torch.bfloat16)                          # [num_tokens, hidden]
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

        chunk = _DECODE_SLOT_CHUNK
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            h_idx = hot_idx[start:end]
            c_idx = cold_idx[start:end]
            sel = is_hot[start:end]                            # [c,1,1]
            # Dequant both formats for these slots, then select. [c, out, in].
            w13 = torch.where(
                sel,
                self._dequant_hot(layer, "w13", h_idx),
                self._dequant_cold(layer, "w13", c_idx, graph_safe=True),
            )
            w2 = torch.where(
                sel,
                self._dequant_hot(layer, "w2", h_idx),
                self._dequant_cold(layer, "w2", c_idx, graph_safe=True),
            )
            st = slot_tok[start:end]                           # [c]
            xe = xe_all[st].unsqueeze(1)                       # [c,1,hidden]
            h13 = torch.bmm(xe, w13.transpose(1, 2))          # [c,1,2*inter]
            hact = _silu_and_mul(h13.float())                 # [c,1,inter]
            ye = torch.bmm(hact.to(torch.bfloat16), w2.transpose(1, 2)).float()
            ye = ye.squeeze(1) * flat_w[start:end].unsqueeze(-1)  # [c,hidden]
            out.index_add_(0, st, ye)
        return out.to(x.dtype)

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]

        # Decode regime -> fixed-shape, host-sync-free path (CUDA-graph
        # capturable). Prefill/profile fall through to the eager chunked loop
        # below (higher throughput at large token counts, not capturable).
        if num_tokens < _RESERVE_MEASURE_MIN_TOKENS:
            if (
                _HAS_FUSED_KERNELS
                and not _FUSED_DECODE_DISABLED
                and self.info.n_cold > 0
            ):
                # Multi-token decode (spec verify, concurrent seqs): grouped
                # kernel amortizes expert weight reads across rows. The
                # branch keys on num_tokens (a SHAPE, static per captured
                # graph), never on values. Single-token keeps the gemv.
                if num_tokens > 1 and not _DECODE_GROUPED_DISABLED:
                    return self._apply_decode_grouped(
                        layer, x, topk_weights, topk_ids
                    )
                return self._apply_decode_fused(
                    layer, x, topk_weights, topk_ids
                )
            return self._apply_decode_graphsafe(layer, x, topk_weights, topk_ids)

        # Prefill/profile regime -> fused grouped-GEMM kernel (eager-only,
        # data-dependent grid). Escape hatch: INKLING_DISABLE_FUSED_PREFILL=1
        # falls back to the dequant-materialize chunked loop below.
        if (
            _HAS_FUSED_KERNELS
            and not _FUSED_PREFILL_DISABLED
            and self.info.n_cold > 0
        ):
            return self._apply_prefill_fused(layer, x, topk_weights, topk_ids)

        # Attempt 16 OOM'd on this call's own *unchunked* xf allocation
        # (1.12 GiB, at a point where the chunking logic hasn't run yet) --
        # a different, earlier line than any prior attempt, with MORE free
        # memory at the point of failure (1.09 GiB) than attempts 14/15 saw
        # (76-164 MiB), meaning the _compute_expert reserve fix genuinely
        # helped locally but total headroom by this layer_id is still being
        # eaten by something across the whole dummy-run pass. Logging
        # allocated bytes per layer_id here to see whether that's honest
        # per-layer growth (leak/non-reclaimed accumulation) or just one
        # unusually expensive layer, instead of guessing again.
        # Hot path: called per MoE layer per chunk per decode step. Guard the
        # entry telemetry (two cuda.memory_* d2h syncs + a log line) behind an
        # explicit DEBUG check so it costs nothing in production -- this was
        # OOM-hunt residue that otherwise fires 64x/token and dominates decode.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[inkling hybrid_moe] apply() entry layer_id=%s num_tokens=%d "
                "top_k=%d allocated=%.1f MiB reserved=%.1f MiB",
                self.layer_id, num_tokens, top_k,
                torch.cuda.memory_allocated(x.device) / (1024 * 1024),
                torch.cuda.memory_reserved(x.device) / (1024 * 1024),
            )

        # CPU mirrors (built in process_weights_after_loading): indexing these
        # in the per-expert loops below avoids a d2h sync on every int(lookup[e]).
        hot_lookup = layer._hybrid_hot_lookup_cpu
        cold_lookup = layer._hybrid_cold_lookup_cpu

        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
        flat_ids = topk_ids.reshape(-1).long()
        flat_w = topk_weights.reshape(-1).float()
        unique_e = flat_ids.unique().tolist()

        def _compute_expert(e: int, w13_e: torch.Tensor, w2_e: torch.Tensor) -> None:
            global _expert_compute_reserve_bytes
            # Measure this call's own peak transient live (reset_peak before,
            # max_memory_allocated delta after) instead of guessing a margin
            # -- see _expert_compute_reserve_bytes above for why attempts 14
            # and 15 both still OOM'd here despite two guessed-margin fixes.
            # ONLY in the prefill/profile regime: these are 2 d2h syncs each,
            # firing ~top_k*layers times per token in decode where they buy
            # nothing (reserve already learned at profile). See
            # _RESERVE_MEASURE_MIN_TOKENS.
            measure = num_tokens >= _RESERVE_MEASURE_MIN_TOKENS
            if measure:
                before_alloc = torch.cuda.memory_allocated(x.device)
                torch.cuda.reset_peak_memory_stats(x.device)
            mask = flat_ids == e
            # Attempts 16/17 OOM'd on a *different*, earlier, unchunked
            # allocation: materializing xf = x.bf16().repeat_interleave(top_k)
            # up front costs num_tokens*top_k*hidden regardless of how many
            # tokens actually route to expert e. Gather straight from x using
            # the same tok_idx this function already needs for index_add_
            # below -- same result (xf[mask] == x[tok_idx] by construction of
            # repeat_interleave/reshape(-1) ordering) without ever forming the
            # full repeated tensor.
            tok_idx = mask.nonzero().flatten() // top_k
            xe = x[tok_idx].to(torch.bfloat16)
            h13 = xe @ w13_e.t()
            del xe
            hact = _silu_and_mul(h13.float())
            del h13
            ye = (hact.to(torch.bfloat16) @ w2_e.t()).float()
            del hact
            ye = ye * flat_w[mask].unsqueeze(-1)
            out.index_add_(0, tok_idx, ye)
            if measure:
                measured = torch.cuda.max_memory_allocated(x.device) - before_alloc
                if measured > _expert_compute_reserve_bytes:
                    _expert_compute_reserve_bytes = measured
                    logger.info(
                        "[inkling hybrid_moe] expert compute transient reserve "
                        "updated to %d bytes (%.1f MiB)",
                        measured, measured / (1024 * 1024),
                    )

        # Hot experts. Only the experts actually ROUTED to in this call are
        # dequantized -- at decode that is the <=top_k unique experts, versus
        # the old code's whole 16-wide chunk per touched expert (up to ~16x
        # more dequant work, the dominant per-layer fixed cost). We still batch
        # the routed set by chunk_size so the many-experts-active case (prefill
        # / profile_run, where routing hits ~all experts) holds no more than
        # one batch's dequant transient live at once -- the OOM guard #100
        # added. n_hot is per-layer (4 early/mid up to 127 for late 53-64ish
        # layers). Chunk size comes from live free memory -- see _pick_chunk_size.
        hot_pairs = [(e, int(hot_lookup[e])) for e in unique_e if hot_lookup[e] >= 0]
        # Decode regime: single chunk, no mem_get_info sync (see
        # _RESERVE_MEASURE_MIN_TOKENS). Prefill/profile: size chunks against
        # live free memory to hold the OOM guard #100.
        decode_regime = num_tokens < _RESERVE_MEASURE_MIN_TOKENS
        chunk_size = len(hot_pairs) if decode_regime else 16
        if hot_pairs and not decode_regime:
            if self.info.hot_format == "nvfp4":
                w13o, w13i = layer.w13_hot_weight.shape[1], layer.w13_hot_weight.shape[2] * 2
                w2o, w2i = layer.w2_hot_weight.shape[1], layer.w2_hot_weight.shape[2] * 2
                elem_bytes = 4  # _dequant_nvfp4 builds a full fp32 `vals` per expert
            else:
                w13o, w13i = layer.w13_hot_bf16.shape[1], layer.w13_hot_bf16.shape[2]
                w2o, w2i = layer.w2_hot_bf16.shape[1], layer.w2_hot_bf16.shape[2]
                elem_bytes = 2  # bf16 hot format: plain slice + no-op cast
            hot_bytes_per_expert = (w13o * w13i + w2o * w2i) * elem_bytes
            chunk_size = _pick_chunk_size(len(hot_pairs), hot_bytes_per_expert, x.device)
        chunk_size = max(chunk_size, 1)
        for start in range(0, len(hot_pairs), chunk_size):
            batch = hot_pairs[start:start + chunk_size]
            idx = torch.tensor([li for (_, li) in batch], dtype=torch.long, device=x.device)
            chunk_w13 = self._dequant_hot(layer, "w13", idx)
            chunk_w2 = self._dequant_hot(layer, "w2", idx)
            for j, (e, _li) in enumerate(batch):
                _compute_expert(e, chunk_w13[j], chunk_w2[j])
            del chunk_w13, chunk_w2
            # Return freed blocks to the driver now, not just at the end of
            # apply(): otherwise mem_get_info() in later chunks (this call
            # and later layers') sees stale, still-fragmented "free" memory,
            # since PyTorch's caching allocator holds freed-but-unreturned
            # blocks that cudaMemGetInfo doesn't count as driver-free anyway.
            # Decode holds one tiny chunk -> skip the sync entirely.
            if not decode_regime:
                torch.cuda.empty_cache()

        # Cold experts: dequant + compute in CHUNKS of the expert dim so we
        # never hold more than one chunk's worth of BOTH w13 and w2 live at
        # once. The prior version dequantized the FULL n_cold-sized w13
        # (~4.43 GiB) and w2 (~2.21 GiB) upfront, simultaneously, for the
        # whole call -- ~6.64 GiB combined, far larger than any of the
        # per-call transient waste fixed in earlier iterations (those
        # shaved hundreds of MiB to ~2 GiB), and sitting right alongside
        # 83+ GiB of already-loaded weights at TP4. Confirmed NOT
        # batch-size-dependent (an 8x smaller dummy profiling batch, 256 vs
        # 2048 tokens, OOM'd identically), which ruled out activation
        # memory as the driver and pointed at this simultaneous full
        # materialization as the actual dominant cost. Chunk size itself
        # must come from live free memory, not a constant tuned against one
        # caller's headroom -- see _pick_chunk_size.
        cold_pairs = [(e, int(cold_lookup[e])) for e in unique_e if cold_lookup[e] >= 0]
        chunk_size = len(cold_pairs) if decode_regime else 16
        if cold_pairs and not decode_regime:
            w13o = layer.w13_cold_codes_0.shape[1]
            w13i = layer.w13_cold_codes_0.shape[2] * _G
            w2o = layer.w2_cold_codes_0.shape[1]
            w2i = layer.w2_cold_codes_0.shape[2] * _G
            cold_bytes_per_expert = (w13o * w13i + w2o * w2i) * 4  # fp32 accumulator
            chunk_size = _pick_chunk_size(len(cold_pairs), cold_bytes_per_expert, x.device)
        chunk_size = max(chunk_size, 1)
        for start in range(0, len(cold_pairs), chunk_size):
            batch = cold_pairs[start:start + chunk_size]
            idx = torch.tensor([li for (_, li) in batch], dtype=torch.long, device=x.device)
            chunk_w13 = self._dequant_cold(layer, "w13", idx)
            chunk_w2 = self._dequant_cold(layer, "w2", idx)
            for j, (e, _li) in enumerate(batch):
                _compute_expert(e, chunk_w13[j], chunk_w2[j])
            del chunk_w13, chunk_w2
            if not decode_regime:
                torch.cuda.empty_cache()

        if not decode_regime:
            torch.cuda.empty_cache()
        return out.to(x.dtype)


# NVFP4 code->value table is a 16-entry compile-time constant. Building it via
# torch.tensor([...], device=cuda) on every _dequant_nvfp4 call is a CPU->CUDA
# copy, which is illegal inside CUDA-graph capture ("Cannot copy between CPU and
# CUDA tensors during CUDA graph capture"). Cache one device tensor per device;
# the entry is populated during eager warmup/profile (before capture), so the
# captured decode path just indexes the cached tensor with no host copy.
_NVFP4_LUT_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]
_nvfp4_lut_cache: dict[torch.device, torch.Tensor] = {}


def _nvfp4_lut(device: torch.device) -> torch.Tensor:
    lut = _nvfp4_lut_cache.get(device)
    if lut is None:
        lut = torch.tensor(
            _NVFP4_LUT_VALUES, dtype=torch.float32, device=device
        )
        _nvfp4_lut_cache[device] = lut
    return lut


def _dequant_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor
) -> torch.Tensor:
    """ModelOpt NVFP4 dequant for a stack of experts -> bf16 [n, out, in].

    In-place after the initial LUT gather: the original out-of-place
    ``vals * blk * s2`` chain allocated two extra full-size fp32
    temporaries on top of ``vals``/``blk`` themselves, which was enough
    (across many hot-expert layers in one dummy forward pass) to be the
    next profile_run() OOM after chunking _dequant_aqlm's accumulator.
    """
    lut = _nvfp4_lut(packed.device)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    n, out, half = packed.shape
    vals = torch.empty(n, out, half * 2, dtype=torch.float32, device=packed.device)
    vals[..., 0::2] = lut[lo]
    vals[..., 1::2] = lut[hi]
    del lo, hi
    blk = scale.to(torch.float32).repeat_interleave(16, dim=-1)
    vals *= blk
    del blk
    vals *= scale2.to(torch.float32).view(n, 1, 1)
    return vals.to(torch.bfloat16)


def _dequant_aqlm(
    codes: list[torch.Tensor],
    codebooks: list[torch.Tensor],
    scales: torch.Tensor,
    chunk_size: int = 16,
    graph_safe: bool = False,
) -> torch.Tensor:
    """AQLM dequant for a stack of experts -> bf16 [n, out, in].

    ``codes[b]``: [n, out, in/G] (uint8 or int16); ``codebooks[b]``:
    [entries_b, G]; ``scales``: [n, out] (per-row/per-out-channel).

    Processed in chunks along the expert (n) dim: each expert's dequant is
    fully independent (no cross-expert reduction), so this changes nothing
    about the result, only the peak transient memory. A full-n fp32
    accumulator here (e.g. ~8.86 GiB for a 252-cold-expert w13 layer at
    hidden=6144) was the direct cause of profile_run() OOMs at TP4: it sits
    alongside the *other* projection's already-dequantized bf16 result
    (also needed live for the compute loop below in apply()), and the two
    together exceeded the ~11.5 GiB of headroom left after loading this
    checkpoint's 83+ GiB/GPU weights. Chunking bounds the transient
    accumulator to chunk_size experts regardless of n.
    """
    n, out, ng = codes[0].shape
    result = torch.empty(
        n, out, ng * _G, dtype=torch.bfloat16, device=codes[0].device
    )
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        acc = torch.zeros(
            end - start, out, ng, _G, dtype=torch.float32, device=codes[0].device
        )
        for c, cb in zip(codes, codebooks):
            idx = c[start:end].long()
            if graph_safe:
                # Host-sync-free variant for the CUDA-graph decode path: skip
                # the `.min()<0` d2h read and mask unconditionally. int16 codes
                # may wrap negative for entries > 32767; codebook rows are
                # addressed modulo the (power-of-two) entry count. The `&`
                # is a no-op for already-in-range codes (e.g. uint8 into a
                # 256-row book), so it is always safe when the book size is a
                # power of two (all Inkling books are). The pow2 test is on a
                # static shape, not data, so it stays graph-capturable.
                size = cb.shape[0]
                if (size & (size - 1)) == 0:
                    idx = idx & (size - 1)
            elif idx.min() < 0:
                # int16 codes may wrap negative for entries > 32767;
                # codebook rows are addressed modulo the (power-of-two)
                # entry count.
                idx = idx & (cb.shape[0] - 1)
            acc += cb.to(torch.float32)[idx]
        w = acc.reshape(end - start, out, ng * _G)
        w *= scales[start:end].to(torch.float32).unsqueeze(-1)
        result[start:end] = w.to(torch.bfloat16)
    return result


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


class InklingHybridQuantConfig(QuantizationConfig):
    """Thin ``QuantizationConfig`` wrapper carrying one hybrid MoE layer's
    checkpoint-derived args (layer info + book layout), resolved by
    ``InklingMoE`` before ``FusedMoE()`` is built (unlike GLM-5.2's
    global-dispatch-by-prefix-regex config, Inkling constructs one of these
    per hybrid layer, so no prefix matching is needed).

    ``InklingHybridExpertsMoEMethod`` itself needs a ``FusedMoEConfig``
    (``moe_config``), which does not exist yet at ``InklingMoE.__init__``
    time -- ``FusedMoE()`` builds it internally and only afterwards calls
    ``get_quant_method(layer, prefix)`` with a ``RoutedExperts`` that already
    has ``layer.moe_config`` set (see ``RoutedExperts.__init__`` /
    ``_get_quant_method``). So the method is constructed lazily, here."""

    def __init__(
        self,
        layer_id: int,
        info: HybridLayerInfo,
        w13_book_entries: list[int],
        w2_book_entries: list[int],
        w13_code_dtypes: list[str],
        w2_code_dtypes: list[str],
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self._info = info
        self._w13_book_entries = w13_book_entries
        self._w2_book_entries = w2_book_entries
        self._w13_code_dtypes = w13_code_dtypes
        self._w2_code_dtypes = w2_code_dtypes

    @classmethod
    def get_name(cls) -> str:
        return "inkling_nvfp4_aqlm_hybrid"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InklingHybridQuantConfig":
        raise NotImplementedError(
            "InklingHybridQuantConfig is built directly by InklingMoE, not "
            "via the generic quantization-config registry"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        moe_config = layer.moe_config
        return InklingHybridExpertsMoEMethod(
            moe_config,
            self._layer_id,
            self._info,
            self._w13_book_entries,
            self._w2_book_entries,
            self._w13_code_dtypes,
            self._w2_code_dtypes,
            tp_size=moe_config.tp_size,
            tp_rank=moe_config.tp_rank,
        )


class InklingHybridTopLevelQuantConfig(QuantizationConfig):
    """Top-level ``QuantizationConfig`` for ``quant_method:
    "inkling_nvfp4_aqlm_hybrid"`` (registered in
    ``vllm.model_executor.layers.quantization``), built from
    ``config.json``'s minimal ``quantization_config`` block.

    This only satisfies vLLM's generic model-config quantization-method
    validation and the ``vllm_config.quant_config`` passed to non-expert
    layers (attention / dense MLP / lm_head, which are bf16 in this
    checkpoint -- see ``InklingNvfp4Config``'s docstring -- so
    ``get_quant_method`` always defers to the unquantized default). The
    actual per-layer hybrid NVFP4+AQLM dispatch happens independently: each
    routed-MoE layer is built by ``InklingMoE``, which constructs its own
    ``InklingHybridQuantConfig`` directly (see above) from
    ``hf_quant_config.json``'s ``aqlm_hybrid`` block, not through this
    class.
    """

    def __init__(self, group_size: int, base_nvfp4: str) -> None:
        super().__init__()
        self.group_size = group_size
        self.base_nvfp4 = base_nvfp4

    @classmethod
    def get_name(cls) -> str:
        return "inkling_nvfp4_aqlm_hybrid"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(
        cls, config: dict[str, Any]
    ) -> "InklingHybridTopLevelQuantConfig":
        return cls(
            group_size=int(config.get("group_size", 8)),
            base_nvfp4=str(config.get("base_nvfp4", "")),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # Non-expert weights stay bf16; routed experts are dispatched
        # per-layer by InklingMoE via InklingHybridQuantConfig, not here.
        # LinearBase.__init__ requires a non-None quant_method whenever
        # quant_config is set, so attention/dense-MLP/lm_head linears need an
        # explicit unquantized method rather than None (embedding/attention
        # layers tolerate None and fall back on their own).
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
