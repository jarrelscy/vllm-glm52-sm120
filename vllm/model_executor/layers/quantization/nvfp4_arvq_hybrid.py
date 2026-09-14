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

import torch

from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
    NvFp4AqlmHybridConfig,
)
from vllm.model_executor.layers.quantization.tp_hybrid_moe import (
    TPHybridExpertsMoEMethod,
    _gateup_loader,
    _replicate_loader,
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


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _check(err):
    if err:
        raise RuntimeError(f"ARVQ CUDA kernel launch failed with error {err}")


def _projection(x, cold_ids, hot_ids, tensors, alpha, n, split, hot_parts):
    lib = _kernels()
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
        lib.hybrid_launch(
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

    @classmethod
    def from_config(cls, config):
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        marker = config["arvq"]
        expected = {
            "format": "rvq256_128x8",
            "activation_planes": 4,
            "weight_scale_group": 128,
            "version": 1,
        }
        if any(marker.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Unsupported ARVQ checkpoint metadata: {marker}")
        books = {
            int(k): {n: int(v[n]) for n in ("n_nvfp4", "n_base", "n_cold")}
            for k, v in config["aqlm_layer_books"].items()
        }
        if any(b["n_base"] != 0 for b in books.values()):
            raise ValueError("ARVQ checkpoint requires n_base=0")
        return cls(ModelOptNvFp4Config.from_config(config["nvfp4"]), books)

    @classmethod
    def get_min_capability(cls):
        return 120

    def get_quant_method(self, layer, prefix):
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
                moe_config=layer.moe_config,
                layer_idx=idx,
                **self.aqlm_layer_books[idx],
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
            )
        return super().get_quant_method(layer, prefix)


class ArvqExpertsMoEMethod(TPHybridExpertsMoEMethod):
    def __init__(self, *args, **kwargs):
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
        h, ish = hidden_size, intermediate_size_per_partition
        t, rank = self._tp, self._tpr
        if h % 128 or ish % 128 or self.n_nvfp4 + self.n_cold != num_experts:
            raise ValueError("ARVQ requires K multiples of 128 and complete routes")
        layer._arvq_ish = ish
        rep = _replicate_loader()

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
                (self.n_cold, n // 16, k // 64, 60),
                torch.uint32,
                shard,
            )
            make(
                f"arvq_{proj}_scales",
                (self.n_cold, n // 16, k // 128, 16),
                torch.uint8,
                shard,
            )
            make(f"arvq_{proj}_codebooks", (384,), torch.uint32, rep)
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
        _kernels()
        device = layer.hyb_kind.device
        kind = layer.hyb_kind.long()
        if not bool(((kind == 0) | (kind == 2)).all().item()):
            raise ValueError("Unexpected expert kind in serialized ARVQ model")
        lookups = []
        for value, count in ((2, self.n_cold), (0, self.n_nvfp4)):
            mask = kind == value
            if int(mask.sum().item()) != count:
                raise ValueError("ARVQ expert count differs from hyb_kind")
            local = mask.long().cumsum(0) - 1
            lookups.append(torch.where(mask, local, -1).to(torch.int32))
        layer._arvq_lookups = torch.stack(lookups).contiguous()
        tensors, alphas = [], []
        for proj in ("w13", "w2"):
            packed = getattr(layer, f"arvq_{proj}_packed")
            guarded = torch.empty(packed.numel() + 1, device=device, dtype=torch.int32)
            guarded[:-1].copy_(packed.view(torch.int32).flatten())
            guarded[-1].zero_()
            packed.data = guarded
            cb = getattr(layer, f"arvq_{proj}_codebooks")
            cs = getattr(layer, f"arvq_{proj}_scales")
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
