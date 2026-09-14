# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local hot-only use of the bitwise-qualified routed P4 pairing prototype."""

import ctypes
from pathlib import Path

import torch

LIB = None


def library():
    global LIB
    if LIB is None:
        LIB = ctypes.CDLL(str(Path(__file__).with_name("arvq") / "prefill_pairs.so"))
        LIB.hybrid_launch.argtypes = (
            [ctypes.c_void_p] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 6
            + [ctypes.c_void_p]
        )
        LIB.hybrid_launch.restype = ctypes.c_int
        LIB.hybrid_pack.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        LIB.hybrid_pack.restype = ctypes.c_int
        LIB.hybrid_pack_pairs.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        LIB.hybrid_pack_pairs.restype = ctypes.c_int
    return LIB


class PairedHot:
    def __init__(self):
        self.partners = None

    def __call__(self, x, cold, hot, tensors, alpha, n, split, hot_parts):
        lib = library()
        slots, k = x.shape
        packed = torch.empty((slots, 4, k // 8), device=x.device, dtype=torch.int32)
        scales = torch.empty((slots, 4, k // 16), device=x.device, dtype=torch.uint8)
        partial = torch.empty((slots, n, split), device=x.device, dtype=torch.float32)
        output = torch.empty((slots, n), device=x.device, dtype=torch.float32)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        ptr = lambda t: ctypes.c_void_p(t.data_ptr())
        if self.partners is None:
            self.partners = torch.empty_like(cold)
            err = lib.hybrid_pack_pairs(
                *map(ptr, [x, packed, scales, cold, hot, self.partners]),
                k,
                slots,
                4,
                stream,
            )
        else:
            err = lib.hybrid_pack(*map(ptr, [x, packed, scales]), k, slots, 4, stream)
        if err:
            raise RuntimeError(f"ARVQ hot prefill CUDA launch failed: {err}")
        err = lib.hybrid_launch(
            *map(
                ptr,
                [*tensors, packed, scales, cold, hot, partial, output, self.partners],
            ),
            alpha,
            n,
            k,
            slots,
            split,
            4,
            hot_parts,
            stream,
        )
        if err:
            raise RuntimeError(f"ARVQ hot prefill CUDA launch failed: {err}")
        return output
