# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental load-time NVFP4 attention output weights with P4 activations.

Only ARVQ checkpoints opt in, using VLLM_ENABLE_NVFP4_P4_O_PROJ=1. The
checkpoint stays unchanged. Resident weights use 4.5 bits per weight plus one
FP32 matrix scale; prefill reconstructs a temporary BF16 matrix. Quantizing
formerly BF16 attention weights is lossy and needs model-level validation.
"""

import ctypes
import os
from pathlib import Path

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

_DENSE_LIB = None


def matches(prefix: str) -> bool:
    return (
        os.environ.get("VLLM_ENABLE_NVFP4_P4_O_PROJ", "0") == "1"
        and prefix.endswith("self_attn.o_proj")
        and "mtp_block" not in prefix.split(".")
    )


def quantize_weight(weight: torch.Tensor):
    """Quantize on CPU in bounded row chunks, returning native E=1 storage."""
    if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 64:
        raise ValueError("NVFP4 P4 requires matrix dimensions divisible by 16/64")
    if weight.dtype != torch.bfloat16:
        raise ValueError("Experimental NVFP4 P4 loads BF16 weights only")
    weight = weight.detach().cpu().contiguous()
    n, k = weight.shape
    global_scale = (weight.abs().amax().float() / (6 * 448)).clamp_min(1e-12)
    fragments = []
    scale_chunks = []
    for start in range(0, n, 256):
        block = weight[start : start + 256].float().reshape(-1, k // 16, 16)
        rows = block.shape[0]
        maxima = block.abs().amax(-1) / 6
        scales = (maxima / global_scale).clamp_min(2**-9).to(torch.float8_e4m3fn)
        norm = block / (scales.float() * global_scale).unsqueeze(-1)
        codes = torch.zeros_like(norm, dtype=torch.uint8)
        for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            codes += (norm.abs() > threshold).to(torch.uint8)
        codes |= (norm < 0).to(torch.uint8) << 3
        flat = codes.reshape(rows, k)
        packed = (flat[:, ::2] | (flat[:, 1::2] << 4)).contiguous()
        words = packed.view(torch.int32).reshape(1, rows // 16, 16, k // 64, 8)
        words = words.permute(0, 1, 3, 2, 4)
        fragment = torch.stack(
            [
                words[
                    :,
                    :,
                    :,
                    8 * (j % 2) : 8 * (j % 2) + 8,
                    4 * (j // 2) : 4 * (j // 2) + 4,
                ].reshape(1, rows // 16, k // 64, 32)
                for j in range(4)
            ],
            dim=3,
        )
        fragments.append(fragment)
        scale_chunks.append(scales.contiguous().view(torch.int32))
    return (
        torch.cat(fragments, dim=1).contiguous(),
        torch.cat(scale_chunks).contiguous(),
        global_scale.reshape(1, 1),
    )


def _dense_kernels():
    global _DENSE_LIB
    if _DENSE_LIB is None:
        path = Path(
            os.environ.get(
                "VLLM_NVFP4_P4_DENSE_LIB",
                str(Path(__file__).with_name("arvq") / "dense.so"),
            )
        )
        _DENSE_LIB = ctypes.CDLL(str(path))
        _DENSE_LIB.nvfp4_dense_dequant.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        )
        _DENSE_LIB.nvfp4_dense_dequant.restype = ctypes.c_int
    return _DENSE_LIB


def _dequantize(weight, scales, global_scale, n, k):
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import _check, _ptr

    lib = _dense_kernels()
    out = torch.empty((n, k), dtype=torch.bfloat16, device=weight.device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check(
        lib.nvfp4_dense_dequant(
            _ptr(weight),
            _ptr(scales),
            _ptr(global_scale),
            _ptr(out),
            n,
            k,
            stream,
        )
    )
    return out


def _paired_projection(x, weight, scales, global_scale, n, split):
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
        _check,
        _kernels,
        _ptr,
    )

    lib = _dense_kernels()
    lib.nvfp4_dense_paired.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    )
    lib.nvfp4_dense_paired.restype = ctypes.c_int
    slots, k = x.shape
    packed = torch.empty((slots, 4, k // 8), device=x.device, dtype=torch.int32)
    act_scales = torch.empty((slots, 4, k // 16), device=x.device, dtype=torch.uint8)
    partial = torch.empty((slots, n, split), device=x.device, dtype=torch.float32)
    out = torch.empty((slots, n), device=x.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check(
        _kernels().hybrid_pack(
            _ptr(x), _ptr(packed), _ptr(act_scales), k, slots, 4, stream
        )
    )
    _check(
        lib.nvfp4_dense_paired(
            *map(
                _ptr, (weight, scales, global_scale, packed, act_scales, partial, out)
            ),
            n,
            k,
            slots,
            split,
            stream,
        )
    )
    return out


@torch.library.custom_op("arvq_hybrid::dense_p4", mutates_args=())
def dense_p4(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    native_max_tokens: int,
) -> torch.Tensor:
    """Keep token dispatch inside the opaque op for dynamic CUDA graphs."""
    from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import _projection

    n = weight.shape[1] * 16
    k = weight.shape[2] * 64
    tokens = x.numel() // k
    flat = x.reshape(tokens, k)
    # Static process flag: CUDA graph capture freezes this dispatch. Changing
    # the flag requires restarting/recapturing graphs, never a live marker.
    paired = os.environ.get("VLLM_NVFP4_P4_PAIRED", "0") == "1"
    if paired and 1 < tokens <= min(native_max_tokens, 16):
        split = 8 if tokens <= 2 else 4 if tokens <= 4 else 2
        out = _paired_projection(
            flat.to(torch.float16).contiguous(), weight, scales, global_scale, n, split
        ).to(x.dtype)
    elif 0 < tokens <= native_max_tokens:
        split = 8 if tokens == 1 else 2 if tokens <= 4 else 1 if tokens <= 8 else 4
        tensors = [weight, weight, scales, weight, scales, global_scale]
        out = _projection(
            flat.to(torch.float16).contiguous(),
            route_ids[0, :tokens],
            route_ids[1, :tokens],
            tensors,
            0.0,
            n,
            split,
            1,
        ).to(x.dtype)
    else:
        decoded = _dequantize(weight, scales, global_scale, n, k)
        out = torch.nn.functional.linear(flat, decoded)
    return out.reshape(*x.shape[:-1], n)


@dense_p4.register_fake
def _dense_p4_fake(x, weight, scales, global_scale, route_ids, native_max_tokens):
    return x.new_empty((*x.shape[:-1], weight.shape[1] * 16))


class NvFp4P4LinearMethod(LinearMethodBase):
    """Load ordinary BF16 tensors, then retain only native NVFP4 weights."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra,
    ):
        loader = extra.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype == torch.int32:
            return
        device = layer.weight.device
        packed = quantize_weight(layer.weight)
        native_max = int(os.environ.get("VLLM_NVFP4_P4_MAX_TOKENS", "4"))
        if not 1 <= native_max <= 32:
            raise ValueError("VLLM_NVFP4_P4_MAX_TOKENS must be in [1, 32]")
        del layer.weight
        for name, value in zip(("weight", "weight_scale", "weight_global"), packed):
            layer.register_parameter(
                name, torch.nn.Parameter(value.to(device), requires_grad=False)
            )
        routes = torch.zeros((2, native_max), dtype=torch.int32, device=device)
        routes[0].fill_(-1)
        layer.register_buffer("p4_route_ids", routes, persistent=False)
        layer.p4_native_max_tokens = native_max

    def apply(self, layer, x, bias=None):
        out = dense_p4(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global,
            layer.p4_route_ids,
            layer.p4_native_max_tokens,
        )
        return out if bias is None else out + bias
