# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from pathlib import Path

import torch

LIB = None


def library():
    global LIB
    if LIB is None:
        LIB = ctypes.CDLL(str(Path(__file__).with_name("arvq") / "prefill_wide.so"))
        LIB.wide_launch_register.argtypes = (
            [ctypes.c_void_p] * 10 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        LIB.wide_launch_register.restype = ctypes.c_int
        LIB.hybrid_pack.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        LIB.hybrid_pack.restype = ctypes.c_int
        LIB.hybrid_pack_register_pairs.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        LIB.hybrid_pack_register_pairs.restype = ctypes.c_int
    return LIB


def mode_for_tokens(tokens):
    if tokens == 2048:
        return 0
    if tokens == 4096:
        return 2
    raise ValueError("Wide hot prefill requires an eligible whole batch")


class WideHot:
    def __init__(self, mode):
        if mode not in (0, 2):
            raise ValueError("Only qualified L1 variants may be dispatched")
        self.mode = mode
        self.groups = None

    def __call__(self, x, cold, hot, tensors, alpha, n, split, hot_parts):
        lib = library()
        slots, k = x.shape
        packed = torch.empty((slots, 4, k // 8), device=x.device, dtype=torch.int32)
        scales = torch.empty((slots, 4, k // 16), device=x.device, dtype=torch.uint8)
        partial = torch.empty((slots, n, split), device=x.device, dtype=torch.float32)
        output = torch.empty((slots, n), device=x.device, dtype=torch.float32)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        ptr = lambda t: ctypes.c_void_p(t.data_ptr())
        if self.groups is None:
            self.groups = torch.empty(
                ((slots + 31) // 32, 7), device=x.device, dtype=torch.int32
            )
            err = lib.hybrid_pack_register_pairs(
                *map(ptr, [x, packed, scales, cold, hot, self.groups]),
                k,
                slots,
                4,
                stream,
            )
        else:
            err = lib.hybrid_pack(*map(ptr, [x, packed, scales]), k, slots, 4, stream)
        if err:
            raise RuntimeError(f"ARVQ hot prefill CUDA launch failed: {err}")
        err = lib.wide_launch_register(
            *map(
                ptr,
                [
                    *tensors[3:6],
                    packed,
                    scales,
                    cold,
                    hot,
                    self.groups,
                    partial,
                    output,
                ],
            ),
            n,
            k,
            slots,
            split,
            hot_parts,
            self.mode,
            stream,
        )
        if err:
            raise RuntimeError(f"ARVQ hot prefill CUDA launch failed: {err}")
        return output
