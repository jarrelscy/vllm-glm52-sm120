# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-only export; original P4 pack is never invoked here."""

import ctypes
from pathlib import Path

import torch

_LIB = None


def run(h13):
    global _LIB
    assert h13.dtype == torch.float32 and h13.ndim == 2 and h13.is_contiguous()
    rows, width = h13.shape
    assert rows > 0 and width > 0 and width % 2 == 0
    if _LIB is None:
        _LIB = ctypes.CDLL(str(Path(__file__).with_name("arvq") / "activation_pack.so"))
        _LIB.fused_silu_mul_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _LIB.fused_silu_mul_f32.restype = ctypes.c_int
    act = torch.empty((rows, width // 2), device=h13.device, dtype=torch.float16)
    status = _LIB.fused_silu_mul_f32(
        h13.data_ptr(),
        act.data_ptr(),
        width // 2,
        rows,
        torch.cuda.current_stream().cuda_stream,
    )
    if status:
        raise RuntimeError(f"fused activation failed: CUDA status {status}")
    return act
