# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact grouped-cold row copies; callers supply unique in-range route slots."""

import ctypes
import os
from functools import cache
from pathlib import Path

import torch


def batch_eligible(x, routes, dst):
    return (
        os.environ.get("VLLM_ARVQ_FUSED_COLD_SCATTER") == "1"
        and x.is_cuda
        and routes.is_cuda
        and dst.is_cuda
        and x.device == routes.device == dst.device
        and x.dtype == torch.bfloat16
        and dst.dtype == torch.float32
        and routes.dtype == torch.int64
        and x.ndim == dst.ndim == 2
        and x.shape[0] in (2048, 4096)
        and routes.ndim == 1
        and x.shape[1] == dst.shape[1] == 6144
        and 0 < routes.numel() <= dst.shape[0]
        and routes.is_contiguous()
        and dst.is_contiguous()
        and dst.data_ptr() % 16 == 0
    )


def prepare(x, sorted_slots, dst):
    """Validate once; bind invariant destination, stream, and CUDA callable.

    The private caller supplies contiguous FP32 torch.mm outputs of width6144
    and contiguous slices of sorted_slots. Slots are unique/in-range because
    they come from nonzero + a permutation. No per-expert scan or metadata gate.
    """
    if not batch_eligible(x, sorted_slots, dst):
        return None
    fn = launch()
    destination = dst.data_ptr()
    stream = torch.cuda.current_stream(dst.device).cuda_stream

    def copy_rows(src, routes):
        error = fn(
            src.data_ptr(), routes.data_ptr(), destination, src.shape[0], 6144, stream
        )
        if error:
            raise RuntimeError(f"ARVQ cold scatter CUDA error: {error}")

    return copy_rows


@cache
def launch():
    path = os.environ.get(
        "VLLM_ARVQ_PREFILL_LIB", str(Path(__file__).with_name("arvq") / "prefill.so")
    )
    lib = ctypes.CDLL(path)
    try:
        fn = lib.arvq_route_scatter
    except AttributeError as error:
        raise RuntimeError(
            "Rebuild arvq/build_prefill.sh for fused cold scatter"
        ) from error
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn
