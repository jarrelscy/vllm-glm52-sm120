# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid NVFP4 + AQLM quantization with per-expert precision tiers.

Motivation: fit GLM-5.2 (744B, 76 routed-expert layers) plus a 1M-token
fp8_ds_mla KV cache into 4x 96GB SM120 GPUs. A few sensitive layers keep
all experts in ModelOpt NVFP4 (stock CUTLASS/Marlin fused MoE, W4A4). In
every other MoE layer precision is assigned per expert from measured
routing mass:

  hot  (kind 0): NVFP4, decoded by W4A16 gemv/dequant kernels    4.5 bpw
  base (kind 1): AQLM w13 1x16 codebook, w2 2x16                 2 / 4 bpw
  cold (kind 2): AQLM w13 1x16, w2 1x16                          2 / 2 bpw

AQLM = 2^16-entry fp16 codebooks over groups of 8 weights, shared across
the layer's experts, with per-expert per-output-channel fp16 scales. All
non-expert weights delegate to the wrapped ModelOpt NVFP4 config (they are
in its exclusion list => BF16).

Checkpoint layout per hybrid layer (nA hot, nM base, nC cold experts,
nB = nM + nC; within each group experts are packed by ascending global id
so local indices derive from hyb_kind; w13 rows are gate then up):
    experts.hyb_kind           int8  [256]
    experts.nvfp4_w13_packed   u8    [nA, 2I, H/2]   (2 fp4/byte, low first)
    experts.nvfp4_w13_bscale   u8    [nA, 2I, H/16]  (fp8 e4m3 block scales)
    experts.nvfp4_w13_scale2   f32   [nA, 2]         (gate, up)
    experts.nvfp4_w2_packed    u8    [nA, H, I/2]
    experts.nvfp4_w2_bscale    u8    [nA, H, I/16]
    experts.nvfp4_w2_scale2    f32   [nA, 1]
    experts.w13_codes          int16 [nB, 1, 2I, H/8]
    experts.w13_codebooks      fp16  [1, 65536, 8]
    experts.w13_scales         fp16  [nB, 2I]
    experts.w2m_codes          int16 [nM, 2, H, I/8]
    experts.w2m_codebooks      fp16  [2, 65536, 8]
    experts.w2m_scales         fp16  [nM, H]
    experts.w2c_codes          int16 [nC, 1, H, I/8]
    experts.w2c_codebooks      fp16  [1, 65536, 8]
    experts.w2c_scales         fp16  [nC, H]

config.json quantization_config:
    {
      "quant_method": "nvfp4_aqlm_hybrid",
      "nvfp4": { ... verbatim ModelOpt NVFP4 quantization_config ... },
      "aqlm": {"entries": 65536, "group_size": 8},
      "aqlm_layer_books":
        {"11": {"n_nvfp4": 20, "n_base": 215, "n_cold": 21}, ...}
    }
"""

import os
import re
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

_EXPERTS_PREFIX_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.mlp\.experts")

_ext = None


def _get_ext():
    """JIT-build and cache the hybrid MoE CUDA extension."""
    global _ext
    if _ext is None:
        import pathlib

        from torch.utils.cpp_extension import load

        src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "csrc"
            / "quantization"
            / "aqlm_moe"
            / "aqlm_moe.cu"
        )
        logger.info_once("Building aqlm_moe extension from %s", src)
        _ext = load(
            name="aqlm_moe_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


def _dequant_reference(
    codes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Pure-torch dequant of all experts, for testing: [E, M, K] fp16."""
    e, books, m, k8 = codes.shape
    idx = codes.view(torch.uint16).long()  # [E, books, M, K8]
    w = codebooks[0][idx[:, 0]]  # [E, M, K8, 8]
    for b in range(1, books):
        w = w + codebooks[b][idx[:, b]]
    w = w.reshape(e, m, k8 * 8)
    return w * scales.unsqueeze(-1)


class NvFp4AqlmHybridConfig(QuantizationConfig):
    """Per-layer dispatch between ModelOpt NVFP4 and hybrid MoE methods."""

    def __init__(
        self,
        nvfp4_config: ModelOptNvFp4Config,
        aqlm_layer_books: dict[int, dict[str, int]],
        entries: int = 65536,
        group_size: int = 8,
    ) -> None:
        super().__init__()
        self.nvfp4_config = nvfp4_config
        self.aqlm_layer_books = aqlm_layer_books
        self.entries = entries
        self.group_size = group_size
        if entries != 65536 or group_size != 8:
            raise ValueError("only 65536-entry codebooks over groups of 8")

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4_aqlm_hybrid"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NvFp4AqlmHybridConfig":
        nvfp4 = ModelOptNvFp4Config.from_config(config["nvfp4"])
        aqlm = config.get("aqlm", {})
        layer_books = {
            int(k): {
                "n_nvfp4": int(v["n_nvfp4"]),
                "n_base": int(v["n_base"]),
                "n_cold": int(v["n_cold"]),
            }
            for k, v in config["aqlm_layer_books"].items()
        }
        return cls(
            nvfp4_config=nvfp4,
            aqlm_layer_books=layer_books,
            entries=int(aqlm.get("entries", 65536)),
            group_size=int(aqlm.get("group_size", 8)),
        )

    def _aqlm_layer_idx(self, prefix: str) -> int | None:
        m = _EXPERTS_PREFIX_RE.search(prefix)
        if m is None:
            return None
        idx = int(m.group(1))
        return idx if idx in self.aqlm_layer_books else None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )

        if isinstance(layer, RoutedExperts):
            idx = self._aqlm_layer_idx(prefix)
            if idx is not None:
                b = self.aqlm_layer_books[idx]
                return HybridExpertsMoEMethod(
                    moe_config=layer.moe_config,
                    layer_idx=idx,
                    n_nvfp4=b["n_nvfp4"],
                    n_base=b["n_base"],
                    n_cold=b["n_cold"],
                )
        return self.nvfp4_config.get_quant_method(layer, prefix)

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        self.nvfp4_config.apply_vllm_mapper(hf_to_vllm_mapper)

    def get_cache_scale(self, name: str) -> str | None:
        return self.nvfp4_config.get_cache_scale(name)


class HybridExpertsMoEMethod(FusedMoEMethodBase):
    """Fused MoE mixing NVFP4 and AQLM experts within one layer.

    Decode (small batch): masked gemv passes, one per storage format;
    each (token, slot) row is computed by exactly one pass, the others
    contribute zeros.
    Prefill (large batch): experts are dequantized to fp16 in per-format
    groups and applied with per-expert GEMMs over expert-sorted tokens.

    v1 limitations: TP/EP must be 1 (PP is the intended parallelism);
    activation must be silu.
    """

    # Experts dequantized per prefill group; bounds scratch memory at
    # G * (2I*H + H*I) * 2 bytes (~600MB for GLM-5.2 at G=4).
    PREFILL_GROUP = 4
    # The grouped path host-syncs (expert counts), so it must never run
    # under CUDA graph capture; keep the gemv path for every batch size
    # vLLM captures.
    DECODE_MAX_TOKENS = 512

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        layer_idx: int,
        n_nvfp4: int,
        n_base: int,
        n_cold: int,
    ) -> None:
        super().__init__(moe_config)
        self.layer_idx = layer_idx
        self.n_nvfp4 = n_nvfp4
        self.n_base = n_base
        self.n_cold = n_cold
        if self.moe.moe_parallel_config.tp_size > 1 or (
            self.moe.moe_parallel_config.ep_size > 1
        ):
            raise NotImplementedError(
                "hybrid MoE layers support PP only (tp_size=ep_size=1)"
            )
        # Optional routing-stats collection (calibration for per-expert
        # precision assignment). Enabled via VLLM_HYBRID_EXPERT_STATS=<dir>.
        self._stats_dir = os.environ.get("VLLM_HYBRID_EXPERT_STATS")
        self._stats: torch.Tensor | None = None
        self._stats_calls = 0

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        h = hidden_size
        i = intermediate_size_per_partition
        g = 8
        entries = 65536
        na, nm, nc = self.n_nvfp4, self.n_base, self.n_cold
        nb = nm + nc
        assert na + nb == num_experts

        def make(name: str, shape: tuple[int, ...], dtype: torch.dtype):
            p = torch.nn.Parameter(
                torch.empty(*shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, p)
            # Checkpoint tensors are pre-packed per expert group.
            set_weight_attrs(p, {"weight_loader": default_weight_loader})

        make("hyb_kind", (num_experts,), torch.int8)
        make("w13_codes", (nb, 1, 2 * i, h // g), torch.int16)
        make("w13_codebooks", (1, entries, g), torch.float16)
        make("w13_scales", (nb, 2 * i), torch.float16)
        make("w2m_codes", (nm, 2, h, i // g), torch.int16)
        make("w2m_codebooks", (2, entries, g), torch.float16)
        make("w2m_scales", (nm, h), torch.float16)
        make("w2c_codes", (nc, 1, h, i // g), torch.int16)
        make("w2c_codebooks", (1, entries, g), torch.float16)
        make("w2c_scales", (nc, h), torch.float16)
        if na > 0:
            make("nvfp4_w13_packed", (na, 2 * i, h // 2), torch.uint8)
            make("nvfp4_w13_bscale", (na, 2 * i, h // 16), torch.uint8)
            make("nvfp4_w13_scale2", (na, 2), torch.float32)
            make("nvfp4_w2_packed", (na, h, i // 2), torch.uint8)
            make("nvfp4_w2_bscale", (na, h, i // 16), torch.uint8)
            make("nvfp4_w2_scale2", (na, 1), torch.float32)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        # Global-expert-id -> (group-local id or -1) lookups. Experts are
        # packed within each group in ascending global-id order.
        device = layer.w13_codes.device
        kind = layer.hyb_kind.to(torch.int64).cpu()

        def lookup(mask: torch.Tensor) -> torch.Tensor:
            local = torch.cumsum(mask.long(), 0) - 1
            return torch.where(mask, local, torch.full_like(local, -1)).to(
                device=device, dtype=torch.int32
            )

        is_nv = kind == 0
        is_base = kind == 1
        is_cold = kind == 2
        layer._nv_lookup = lookup(is_nv)
        layer._b_lookup = lookup(~is_nv)      # w13 AQLM array (base + cold)
        layer._w2m_lookup = lookup(is_base)
        layer._w2c_lookup = lookup(is_cold)
        layer._nv_globals = is_nv.nonzero().flatten().tolist()
        layer._base_globals = is_base.nonzero().flatten().tolist()
        layer._cold_globals = is_cold.nonzero().flatten().tolist()

    def get_fused_moe_quant_config(self, layer: "RoutedExperts"):
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # Shared experts are handled by the MoE runner (NO_OVERLAP order,
        # since mk_can_overlap_shared_experts is False for this method).
        act = str(getattr(layer.activation, "value", layer.activation))
        assert act.lower().endswith("silu"), layer.activation
        if self._stats_dir is not None:
            self._record_stats(topk_ids)
        num_tokens = x.shape[0]
        if (
            num_tokens <= self.DECODE_MAX_TOKENS
            or torch.cuda.is_current_stream_capturing()
        ):
            out = self._apply_gemv(layer, x, topk_weights, topk_ids)
        else:
            out = self._apply_grouped(layer, x, topk_weights, topk_ids)
        return out.to(x.dtype)

    def _record_stats(self, topk_ids: torch.Tensor) -> None:
        num_experts = self.moe.num_experts
        if self._stats is None:
            self._stats = torch.zeros(
                num_experts, dtype=torch.int64, device=topk_ids.device
            )
        self._stats += torch.bincount(
            topk_ids.reshape(-1).long(), minlength=num_experts
        )
        self._stats_calls += 1
        if self._stats_calls % 50 == 0:
            self._dump_stats()

    def _dump_stats(self) -> None:
        import numpy as np

        os.makedirs(self._stats_dir, exist_ok=True)
        np.save(
            os.path.join(self._stats_dir, f"layer_{self.layer_idx}.npy"),
            self._stats.cpu().numpy(),
        )

    def _apply_gemv(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        ext = _get_ext()
        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]

        xf = x.to(torch.float16)
        flat = topk_ids.reshape(-1).long()
        b_ids = layer._b_lookup[flat]
        w2m_ids = layer._w2m_lookup[flat]
        w2c_ids = layer._w2c_lookup[flat]
        nv_ids = layer._nv_lookup[flat] if self.n_nvfp4 > 0 else None
        # Each token row repeated top_k times: rows of xr line up with slots.
        xr = xf.repeat_interleave(top_k, dim=0)

        h13 = ext.aqlm_moe_gemv(
            xr, layer.w13_codes, layer.w13_codebooks, layer.w13_scales, b_ids
        )
        if nv_ids is not None:
            h13 += ext.nvfp4_moe_gemv(
                xr, layer.nvfp4_w13_packed, layer.nvfp4_w13_bscale,
                layer.nvfp4_w13_scale2, nv_ids,
            )
        hact = _silu_and_mul(h13).contiguous()
        out = None
        for codes, cbs, scales, ids, n in (
            (layer.w2m_codes, layer.w2m_codebooks, layer.w2m_scales,
             w2m_ids, self.n_base),
            (layer.w2c_codes, layer.w2c_codebooks, layer.w2c_scales,
             w2c_ids, self.n_cold),
        ):
            if n == 0:
                continue
            y = ext.aqlm_moe_gemv(hact, codes, cbs, scales, ids)
            out = y if out is None else out + y
        if nv_ids is not None:
            y = ext.nvfp4_moe_gemv(
                hact, layer.nvfp4_w2_packed, layer.nvfp4_w2_bscale,
                layer.nvfp4_w2_scale2, nv_ids,
            )
            out = y if out is None else out + y
        out = out.view(num_tokens, top_k, hidden).float()
        return (out * topk_weights.unsqueeze(-1).float()).sum(dim=1)

    def _apply_grouped(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        ext = _get_ext()
        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        num_experts = self.moe.num_experts

        xf = x.to(torch.float16)
        flat_ids = topk_ids.reshape(-1)
        order = torch.argsort(flat_ids)
        tok_of_slot = order // top_k
        counts = torch.bincount(flat_ids, minlength=num_experts)
        ends = counts.cumsum(0)

        xg = xf[tok_of_slot]  # [T*K, H] expert-sorted
        y = torch.empty_like(xg)

        counts_c = counts.cpu()
        ends_c = ends.cpu()
        group = self.PREFILL_GROUP
        b_lookup_c = layer._b_lookup.cpu()

        def w13_aqlm(globals_chunk):
            local = torch.tensor(
                [int(b_lookup_c[ge]) for ge in globals_chunk],
                dtype=torch.int32, device=x.device,
            )
            return ext.aqlm_moe_dequant(
                layer.w13_codes, layer.w13_codebooks, layer.w13_scales, local
            )

        def fmt_base(globals_chunk, local0):
            local = torch.arange(
                local0, local0 + len(globals_chunk), dtype=torch.int32,
                device=x.device,
            )
            return (
                w13_aqlm(globals_chunk),
                ext.aqlm_moe_dequant(layer.w2m_codes, layer.w2m_codebooks,
                                     layer.w2m_scales, local),
            )

        def fmt_cold(globals_chunk, local0):
            local = torch.arange(
                local0, local0 + len(globals_chunk), dtype=torch.int32,
                device=x.device,
            )
            return (
                w13_aqlm(globals_chunk),
                ext.aqlm_moe_dequant(layer.w2c_codes, layer.w2c_codebooks,
                                     layer.w2c_scales, local),
            )

        def fmt_nv(globals_chunk, local0):
            local = torch.arange(
                local0, local0 + len(globals_chunk), dtype=torch.int32,
                device=x.device,
            )
            return (
                ext.nvfp4_moe_dequant(layer.nvfp4_w13_packed,
                                      layer.nvfp4_w13_bscale,
                                      layer.nvfp4_w13_scale2, local),
                ext.nvfp4_moe_dequant(layer.nvfp4_w2_packed,
                                      layer.nvfp4_w2_bscale,
                                      layer.nvfp4_w2_scale2, local),
            )

        format_lists = [
            (layer._base_globals, fmt_base),
            (layer._cold_globals, fmt_cold),
        ]
        if self.n_nvfp4 > 0:
            format_lists.append((layer._nv_globals, fmt_nv))

        for globals_list, fmt in format_lists:
            for g0 in range(0, len(globals_list), group):
                gids = globals_list[g0 : g0 + group]
                if sum(int(counts_c[ge]) for ge in gids) == 0:
                    continue
                w13, w2 = fmt(gids, g0)  # [g, 2I, H], [g, H, I]
                for j, ge in enumerate(gids):
                    n = int(counts_c[ge])
                    if n == 0:
                        continue
                    sl = slice(int(ends_c[ge]) - n, int(ends_c[ge]))
                    h13 = xg[sl] @ w13[j].t()
                    y[sl] = _silu_and_mul(h13) @ w2[j].t()

        yw = y.float() * topk_weights.reshape(-1)[order].unsqueeze(-1).float()
        out = torch.zeros(
            num_tokens, hidden, dtype=torch.float32, device=x.device
        )
        out.index_add_(0, tok_of_slot, yw)
        return out


# Backwards-compatible alias (tests, older configs).
AqlmMoEMethod = HybridExpertsMoEMethod


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
