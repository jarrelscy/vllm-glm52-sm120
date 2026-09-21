# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialized NVFP4 + additive RVQ experts for SM120, with P4 activations.

ARVQ experts always execute their serialized weights, including prefill.
There is no AQLM fallback. Tensor-parallel output reduction remains owned by
vLLM's MoE runner, as in TPHybridExpertsMoEMethod.
"""

import ctypes
import os
from pathlib import Path
from typing import Any, Literal, cast

import regex as re
import torch

from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
    NvFp4AqlmHybridConfig,
)
from vllm.model_executor.layers.quantization.tp_hybrid_moe import (
    TPHybridExpertsMoEMethod,
    _gateup_loader,
    _rowk_loader,
)
from vllm.model_executor.utils import set_weight_attrs

_LIB = None


def _kernels():
    global _LIB
    if _LIB is None:
        path = Path(
            os.environ.get(
                "VLLM_ARVQ_KERNEL_LIB",
                str(Path(__file__).with_name("arvq") / "hybrid.so"),
            )
        )
        if not path.is_file():
            raise RuntimeError(
                f"Missing ARVQ CUDA library {path}; run arvq/build.sh with "
                "CUDA 12.9 or newer on the SM120 serving image."
            )
        _LIB = ctypes.CDLL(str(path))
        _LIB.hybrid_launch.argtypes = (
            [ctypes.c_void_p] * 12
            + [ctypes.c_float]
            + [ctypes.c_int] * 6
            + [ctypes.c_void_p]
        )
        _LIB.hybrid_launch.restype = ctypes.c_int
        _LIB.hybrid_pack.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        _LIB.hybrid_pack.restype = ctypes.c_int
    return _LIB


_EXPERT_FORMAT = "rvq256_256x8_expert"
_FP16_FORMAT = "rvq256_256x8_expert_fp16block"
_RS4_FORMAT = "rvq256_256x8_expert_fp16block_rs4"
_FORMATS = {
    "rvq256_128x8": (60, 384),
    "rvq256_256x8": (64, 512),
    _EXPERT_FORMAT: (64, 512),
    _FP16_FORMAT: (64, 512),
    _RS4_FORMAT: (64, 512),
}

# One checkpoint format per process; set when a config is constructed.
_ACTIVE_FORMAT = _EXPERT_FORMAT


def _set_active_format(fmt):
    global _ACTIVE_FORMAT
    _ACTIVE_FORMAT = fmt


def residual_scale_shift():
    """Powers of two by which the residual book's contribution is lowered.

    rs4 reconstructs each 8-weight group as
    global * fp16_block_scale * (c0[main] + c1[residual] / 4); every other
    format keeps the two books at equal weight.
    """
    return 2 if _ACTIVE_FORMAT == _RS4_FORMAT else 0


def _layout(codebooks):
    if codebooks.ndim == 1 and codebooks.shape[0] in (384, 512):
        entries = codebooks.shape[0]
    elif codebooks.ndim == 2 and codebooks.shape[1] == 512:
        entries = 512
    else:
        raise ValueError(
            f"Invalid ARVQ codebook size/shape {tuple(codebooks.shape)}; "
            "expected [384], [512], or expert [E,512]"
        )
    if not codebooks.is_contiguous():
        raise ValueError("ARVQ codebooks must be contiguous")
    return (60 if entries == 384 else 64), entries


def _expert_codebooks(codebooks, cold_slot):
    _layout(codebooks)
    return codebooks[cold_slot] if codebooks.ndim == 2 else codebooks


def _launch_for_codebooks(lib, codebooks):
    _, entries = _layout(codebooks)
    if entries == 384:
        return lib.hybrid_launch
    symbol = "hybrid_launch_8x8"
    if codebooks.ndim == 2:
        symbol += "_expert"
        if residual_scale_shift():
            symbol += "_rs4"
    elif residual_scale_shift():
        raise ValueError("rs4 requires per-expert [E,512] codebooks")
    try:
        launch = getattr(lib, symbol)
    except AttributeError as error:
        raise RuntimeError(f"ARVQ requires {symbol}; rebuild arvq/build.sh") from error
    launch.argtypes = lib.hybrid_launch.argtypes
    launch.restype = ctypes.c_int
    return launch


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _check(err):
    if err:
        raise RuntimeError(f"ARVQ CUDA kernel launch failed with error {err}")


def _projection(x, cold_ids, hot_ids, tensors, alpha, n, split, hot_parts):
    if os.environ.get("VLLM_ARVQ_REFERENCE_WEIGHTS", "0") == "1":
        from vllm.model_executor.layers.quantization.arvq_reference import projection

        return projection(x, cold_ids, hot_ids, tensors, alpha, n, split, hot_parts)
    # Dense hot-only callers alias dummy cold pointers to the packed hot weight.
    hot_only = tensors[0] is tensors[1]
    if not hot_only:
        _layout(tensors[1])
    lib = _kernels()
    launch = lib.hybrid_launch if hot_only else _launch_for_codebooks(lib, tensors[1])
    slots, k = x.shape
    planes = 4
    packed = torch.empty((slots, planes, k // 8), device=x.device, dtype=torch.int32)
    scales = torch.empty((slots, planes, k // 16), device=x.device, dtype=torch.uint8)
    partial = torch.empty((slots, n, split), device=x.device, dtype=torch.float32)
    out = torch.empty((slots, n), device=x.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check(
        lib.hybrid_pack(_ptr(x), _ptr(packed), _ptr(scales), k, slots, planes, stream)
    )
    # cw, cb, cs, hw, hs, hot_global, activations, activation_scales,
    # cold_ids, hot_ids, partial, output.
    args = [*tensors, packed, scales, cold_ids, hot_ids, partial, out]
    _check(
        launch(
            *[_ptr(t) for t in args],
            alpha,
            n,
            k,
            slots,
            split,
            planes,
            hot_parts,
            stream,
        )
    )
    return out


def _grouped_prefill_enabled(num_tokens: int, top_k: int, hidden: int) -> bool:
    """Bound eager execution; ``toggle`` is a diagnostic same-boot A/B mode.

    In diagnostic mode, marker-file presence enables grouped prefill. Keep its
    state fixed throughout an entire request on every TP rank. Normal ``1``
    mode performs no filesystem checks; captured paths never inspect the file.
    Eligibility starts at 2048 tokens, the measured grouped-MLP crossover;
    the routed FP32 output buffer remains bounded to one GiB.
    """
    mode = os.environ.get("VLLM_ARVQ_GROUPED_PREFILL", "0")
    eligible = (
        mode in ("1", "toggle")
        and num_tokens >= 2048
        and num_tokens * top_k * hidden * 4 <= 1024**3
        and not torch.cuda.is_current_stream_capturing()
    )
    if not eligible:
        return False
    return mode == "1" or Path("/dev/shm/vllm_arvq_grouped_prefill_on").exists()


@torch.library.custom_op("arvq_hybrid::mlp", mutates_args=())
def arvq_mlp(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    lookups: torch.Tensor,
    tensors: list[torch.Tensor],
    alphas: list[float],
    chunk_tokens: int,
) -> torch.Tensor:
    """Opaque graph-safe P4 pack, unified projection, SiLU and route combine."""
    reference_weights = os.environ.get("VLLM_ARVQ_REFERENCE_WEIGHTS", "0") == "1"
    if not reference_weights and _grouped_prefill_enabled(
        x.shape[0], topk_ids.shape[1], x.shape[1]
    ):
        from vllm.model_executor.layers.quantization.nvfp4_arvq_prefill import (
            grouped_cold_prefill,
        )

        return grouped_cold_prefill(
            x,
            topk_weights,
            topk_ids,
            lookups,
            tensors,
            alphas,
            projection=_projection,
            chunk_tokens=chunk_tokens,
        )
    outputs = []
    hidden = x.shape[1]
    n13 = tensors[3].shape[1] * 16
    top_k = topk_ids.shape[1]
    for start in range(0, x.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, x.shape[0])
        ids = lookups[:, topk_ids[start:stop].reshape(-1).long()]
        cold_ids, hot_ids = ids[0], ids[1]
        xr = x[start:stop].to(torch.float16).repeat_interleave(top_k, 0)
        slots = xr.shape[0]
        h13 = _projection(
            xr,
            cold_ids,
            hot_ids,
            tensors[:6],
            alphas[0],
            n13,
            16 if slots <= 32 else 8,
            2,
        )
        if (
            not reference_weights
            and slots <= 64
            and os.environ.get("VLLM_ARVQ_FUSED_ACTIVATION_PACK", "0") == "1"
        ):
            from vllm.model_executor.layers.quantization import (
                nvfp4_arvq_activation_pack as activation_pack,
            )

            packed_activation, activation_scales = activation_pack.pack_activation(h13)
            del h13
            down = activation_pack.down_prepacked(
                packed_activation,
                activation_scales,
                cold_ids,
                hot_ids,
                tensors[6:],
                alphas[1],
                hidden,
            )
        else:
            h13 = h13.to(torch.float16)
            gate, up = h13.chunk(2, dim=-1)
            hact = (torch.nn.functional.silu(gate) * up).to(torch.float16)
            down = _projection(
                hact, cold_ids, hot_ids, tensors[6:], alphas[1], hidden, 2, 1
            )
        weighted = down.reshape(stop - start, top_k, hidden)
        weighted = weighted * topk_weights[start:stop, :, None].float()
        outputs.append(weighted.sum(1).to(x.dtype))
    if not outputs:
        return torch.empty_like(x)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


@arvq_mlp.register_fake
def _arvq_mlp_fake(x, topk_weights, topk_ids, lookups, tensors, alphas, chunk_tokens):
    return torch.empty_like(x)


class NvFp4ArvqHybridConfig(NvFp4AqlmHybridConfig):
    """Explicit ARVQ checkpoint marker within the existing hybrid envelope."""

    def __init__(self, *args, arvq_format="rvq256_128x8", **kwargs):
        if arvq_format not in _FORMATS:
            raise ValueError(f"Unsupported ARVQ format: {arvq_format}")
        super().__init__(*args, **kwargs)
        self.arvq_format = arvq_format
        _set_active_format(arvq_format)

    @classmethod
    def get_name(cls) -> Literal["nvfp4_arvq_hybrid"]:
        return "nvfp4_arvq_hybrid"

    @classmethod
    def from_config(cls, config):
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        marker = config["arvq"]
        expected: dict[str, object] = {
            "activation_planes": 4,
            "weight_scale_group": 128,
        }
        # Published 8+8 checkpoints use version 2 for the same native layout.
        # Preserve version 1 support and the original 8+7 version policy.
        fmt = marker.get("format")
        versions = (1, 2) if fmt == "rvq256_256x8" else (1,)
        if fmt == _EXPERT_FORMAT:
            versions = (3,)
            expected.update(codebook_scope="expert", codebook_sizes=[256, 256])
        elif fmt in (_FP16_FORMAT, _RS4_FORMAT):
            versions = (3, 4)
            expected.update(codebook_scope="expert", codebook_sizes=[256, 256])
        elif marker.get("codebook_scope", "shared") != "shared":
            raise ValueError(f"Unsupported ARVQ checkpoint metadata: {marker}")
        elif "codebook_sizes" in marker:
            expected["codebook_sizes"] = [256, 256 if fmt == "rvq256_256x8" else 128]
        if (
            marker.get("format") not in _FORMATS
            or marker.get("version") not in versions
            or any(marker.get(k) != v for k, v in expected.items())
        ):
            raise ValueError(f"Unsupported ARVQ checkpoint metadata: {marker}")
        books: dict[int, dict[str, Any]] = {
            int(k): {n: int(v[n]) for n in ("n_nvfp4", "n_base", "n_cold")}
            for k, v in config["aqlm_layer_books"].items()
        }
        for key, value in config["aqlm_layer_books"].items():
            if "cold_expert_ids" in value:
                ids = value["cold_expert_ids"]
                count = books[int(key)]["n_cold"]
                total = count + books[int(key)]["n_nvfp4"]
                if (
                    not isinstance(ids, list)
                    or len(ids) != count
                    or any(type(i) is not int or i < 0 or i >= total for i in ids)
                    or len(set(ids)) != count
                ):
                    raise ValueError("Invalid ordered ARVQ cold_expert_ids")
                books[int(key)]["cold_expert_ids"] = ids
        if any(b["n_base"] != 0 for b in books.values()):
            raise ValueError("ARVQ checkpoint requires n_base=0")
        nvfp4 = cast(
            ModelOptNvFp4Config, ModelOptNvFp4Config.from_config(config["nvfp4"])
        )
        return cls(nvfp4, books, arvq_format=marker["format"])

    @classmethod
    def get_min_capability(cls):
        return 120

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.nvfp4_p4_linear import (
            NvFp4P4LinearMethod,
            matches,
        )

        if isinstance(layer, LinearBase) and matches(prefix):
            # MTP construction can use layers.78 without an mtp_block component.
            # The serialized expert layer range ends at the final target layer.
            target_layer = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if (
                target_layer is not None
                and self.aqlm_layer_books
                and int(target_layer.group(1)) <= max(self.aqlm_layer_books)
            ):
                return NvFp4P4LinearMethod()

        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )

        idx = self._aqlm_layer_idx(prefix)
        if isinstance(layer, RoutedExperts) and idx is not None:
            return ArvqExpertsMoEMethod(
                arvq_format=self.arvq_format,
                moe_config=layer.moe_config,
                layer_idx=idx,
                **self.aqlm_layer_books[idx],
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
            )
        return super().get_quant_method(layer, prefix)


class ArvqExpertsMoEMethod(TPHybridExpertsMoEMethod):
    arvq_format = "rvq256_128x8"

    def __init__(
        self, *args, arvq_format="rvq256_128x8", cold_expert_ids=None, **kwargs
    ):
        if arvq_format not in _FORMATS:
            raise ValueError(f"Unsupported ARVQ format: {arvq_format}")
        self.arvq_format = arvq_format
        _set_active_format(arvq_format)
        self.cold_expert_ids = cold_expert_ids
        super().__init__(*args, **kwargs)
        if self.n_base:
            raise ValueError("Serialized ARVQ supports hot and cold experts only")
        if self.moe.moe_parallel_config.ep_size > 1:
            raise NotImplementedError("ARVQ expert parallelism is not supported")
        self._chunk_tokens = int(os.environ.get("VLLM_ARVQ_CHUNK_TOKENS", "128"))
        if not 1 <= self._chunk_tokens <= 256:
            raise ValueError("VLLM_ARVQ_CHUNK_TOKENS must be between 1 and 256")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        words, entries = _FORMATS[self.arvq_format]
        h, ish = hidden_size, intermediate_size_per_partition
        t, rank = self._tp, self._tpr
        if h % 128 or ish % 128 or self.n_nvfp4 + self.n_cold != num_experts:
            raise ValueError("ARVQ requires K multiples of 128 and complete routes")
        layer._arvq_ish = ish

        def rep(param, loaded):
            # copy_ permits broadcasting: explicitly reject v2 books in v3.
            if loaded.shape != param.shape or loaded.dtype != param.dtype:
                raise ValueError(
                    "Serialized ARVQ tensor shape/dtype disagrees with format: "
                    f"expected {tuple(param.shape)} {param.dtype}, "
                    f"got {tuple(loaded.shape)} {loaded.dtype}"
                )
            param.data.copy_(loaded)

        def make(name, shape, dtype, loader):
            p = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, {"weight_loader": loader})

        make("hyb_kind", (num_experts,), torch.int8, rep)
        for proj, n, k in (("w13", 2 * ish, h), ("w2", h, ish)):
            if proj == "w13":
                shard = _gateup_loader(rank, t, ish * t // 16, axis=1)
            else:
                shard = _rowk_loader(rank, t, axis=2)
            make(
                f"arvq_{proj}_packed",
                (self.n_cold, n // 16, k // 64, words),
                torch.uint32,
                shard,
            )
            make(
                f"arvq_{proj}_scales",
                (self.n_cold, n // 16, k // 128, 16),
                torch.uint8,
                shard,
            )
            book_shape = (
                (self.n_cold, entries)
                if self.arvq_format == _EXPERT_FORMAT
                else (entries,)
            )
            make(f"arvq_{proj}_codebooks", book_shape, torch.uint32, rep)
            make(f"arvq_{proj}_global", (1,), torch.float32, rep)
        na = self.n_nvfp4
        make(
            "nvfp4_w13_packed",
            (na, 2 * ish, h // 2),
            torch.uint8,
            _gateup_loader(rank, t, ish * t, axis=1),
        )
        make(
            "nvfp4_w13_bscale",
            (na, 2 * ish, h // 16),
            torch.uint8,
            _gateup_loader(rank, t, ish * t, axis=1),
        )
        make("nvfp4_w13_scale2", (na, 2), torch.float32, rep)
        make(
            "nvfp4_w2_packed",
            (na, h, ish // 2),
            torch.uint8,
            _rowk_loader(rank, t, axis=2),
        )
        make(
            "nvfp4_w2_bscale",
            (na, h, ish // 16),
            torch.uint8,
            _rowk_loader(rank, t, axis=2),
        )
        make("nvfp4_w2_scale2", (na, 1), torch.float32, rep)

    def process_weights_after_loading(self, layer):
        words, entries = _FORMATS[self.arvq_format]
        for proj in ("w13", "w2"):
            packed = getattr(layer, f"arvq_{proj}_packed")
            cb = getattr(layer, f"arvq_{proj}_codebooks")
            expected_books = (
                (self.n_cold, entries)
                if self.arvq_format == _EXPERT_FORMAT
                else (entries,)
            )
            if (
                packed.ndim != 4
                or packed.shape[0] != self.n_cold
                or packed.shape[-1] != words
                or packed.dtype != torch.uint32
                or tuple(cb.shape) != expected_books
                or cb.dtype != torch.uint32
                or not cb.is_contiguous()
            ):
                raise ValueError(
                    "Serialized ARVQ packed/codebook layout disagrees with format"
                )
        lib = _kernels()
        _launch_for_codebooks(lib, layer.arvq_w13_codebooks)
        device = layer.hyb_kind.device
        kind = layer.hyb_kind.long()
        if not bool(((kind == 0) | (kind == 2)).all().item()):
            raise ValueError("Unexpected expert kind in serialized ARVQ model")
        lookups = []
        for value, count in ((2, self.n_cold), (0, self.n_nvfp4)):
            mask = kind == value
            if int(mask.sum().item()) != count:
                raise ValueError("ARVQ expert count differs from hyb_kind")
            ordered = getattr(self, "cold_expert_ids", None) if value == 2 else None
            if ordered is None:
                # Legacy exporters enumerate cold slots by ascending global ID.
                local = mask.long().cumsum(0) - 1
                lookup = torch.where(mask, local, -1).to(torch.int32)
            else:
                ids = torch.tensor(ordered, device=device, dtype=torch.long)
                if (
                    ids.numel() != count
                    or ids.unique().numel() != count
                    or bool(((ids < 0) | (ids >= kind.numel())).any().item())
                    or not bool(mask[ids].all().item())
                ):
                    raise ValueError("cold_expert_ids disagrees with hyb_kind")
                lookup = torch.full_like(kind, -1, dtype=torch.int32)
                lookup[ids] = torch.arange(count, device=device, dtype=torch.int32)
            lookups.append(lookup)
        layer._arvq_lookups = torch.stack(lookups).contiguous()
        tensors: list[torch.Tensor] = []
        alphas: list[float] = []
        for proj in ("w13", "w2"):
            packed = getattr(layer, f"arvq_{proj}_packed")
            guarded = torch.empty(packed.numel() + 1, device=device, dtype=torch.int32)
            guarded[:-1].copy_(packed.view(torch.int32).flatten())
            guarded[-1].zero_()
            packed.data = guarded
            cb = getattr(layer, f"arvq_{proj}_codebooks")
            cs = getattr(layer, f"arvq_{proj}_scales")
            if cs.dtype == torch.uint8:
                if bool((cs >= 127).any().item()):
                    raise ValueError(
                        "FP16 scale trial encountered invalid unsigned E4M3"
                    )
                cs.data = cs.view(torch.float8_e4m3fn).to(torch.float16)
            elif cs.dtype != torch.float16:
                raise ValueError("ARVQ cold scales must be uint8 E4M3 or fitted FP16")
            print(
                f"ARVQ_FP16_SCALES projection={proj} shape={tuple(cs.shape)} "
                f"elements={cs.numel()} dtype={cs.dtype}",
                flush=True,
            )
            hw = getattr(layer, f"nvfp4_{proj}_packed")
            e, n, k2 = hw.shape
            k = k2 * 2
            native = (
                hw.view(torch.int32)
                .reshape(e, n // 16, 2, 8, k // 64, 2, 4)
                .permute(0, 1, 4, 5, 2, 3, 6)
                .contiguous()
            )
            hw.data = native.reshape(e, n // 16, k // 64, 4, 32)
            hs = getattr(layer, f"nvfp4_{proj}_bscale")
            hs.data = hs.view(torch.int32)
            hg = getattr(layer, f"nvfp4_{proj}_scale2")
            tensors.extend((packed, cb, cs, hw, hs, hg))
            alphas.append(float(getattr(layer, f"arvq_{proj}_global").item()))
        layer._arvq_tensors = tensors
        layer._arvq_alphas = alphas

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        act = str(getattr(layer.activation, "value", layer.activation))
        if not act.lower().endswith("silu"):
            raise ValueError(f"ARVQ requires SiLU activation, got {act}")
        return arvq_mlp(
            x,
            topk_weights,
            topk_ids,
            layer._arvq_lookups,
            layer._arvq_tensors,
            layer._arvq_alphas,
            self._chunk_tokens,
        )
