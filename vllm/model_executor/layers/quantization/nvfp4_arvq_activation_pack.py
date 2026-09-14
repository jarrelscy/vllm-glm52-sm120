# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact half-rounded SiLU multiplication and four-plane activation packing."""

import ctypes
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as base

LIB = None


def pack_activation(h13):
    global LIB
    if LIB is None:
        LIB = ctypes.CDLL(str(Path(__file__).with_name("arvq") / "activation_pack.so"))
        LIB.fused_silu_mul_pack.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        )
        LIB.fused_silu_mul_pack.restype = ctypes.c_int
    slots, n13 = h13.shape
    k = n13 // 2
    q = torch.empty((slots, 4, k // 8), dtype=torch.int32, device=h13.device)
    s = torch.empty((slots, 4, k // 16), dtype=torch.uint8, device=h13.device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    base._check(
        LIB.fused_silu_mul_pack(
            base._ptr(h13), base._ptr(q), base._ptr(s), None, k, slots, stream
        )
    )
    return q, s


def down_prepacked(q, s, cold, hot, tensors, alpha, n):
    slots = q.shape[0]
    k = q.shape[2] * 8
    partial = torch.empty((slots, n, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((slots, n), dtype=torch.float32, device=q.device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    lib = base._kernels()
    launch = base._launch_for_codebooks(lib, tensors[1])
    args = [*tensors, q, s, cold, hot, partial, out]
    base._check(launch(*map(base._ptr, args), alpha, n, k, slots, 2, 4, 1, stream))
    return out
