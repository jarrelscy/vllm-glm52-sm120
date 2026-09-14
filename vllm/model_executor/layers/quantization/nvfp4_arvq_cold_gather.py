# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated fused weight decode/input gather. No serving imports or edits."""

import ctypes
from pathlib import Path

import torch

_LIB = None


def decode_gather(x, route_slots, packed, codebooks, scales, global_scale, n, k, top_k):
    global _LIB
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert route_slots.dtype == torch.int64 and route_slots.is_contiguous()
    assert codebooks.numel() in (384, 512)
    if _LIB is None:
        _LIB = ctypes.CDLL(str(Path(__file__).with_name("arvq") / "decode_gather.so"))
    bits = 7 if codebooks.numel() == 384 else 8
    fn = getattr(_LIB, f"arvq_dequant_gather_fp16_{bits}")
    fn.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int, ctypes.c_int]
    )
    fn.restype = ctypes.c_int
    weight = torch.empty((n, k), device=x.device, dtype=torch.float16)
    rows = torch.empty((route_slots.numel(), k), device=x.device, dtype=torch.float16)
    ptr = lambda t: ctypes.c_void_p(t.data_ptr())
    status = fn(
        ptr(packed),
        ptr(codebooks),
        ptr(scales),
        global_scale,
        ptr(weight),
        n,
        k,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ptr(x),
        ptr(route_slots),
        ptr(rows),
        rows.shape[0],
        top_k,
    )
    if status:
        raise RuntimeError(f"decode/gather CUDA error{status}")
    return rows, weight
