# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA nearest assignment and AQLM-code-to-RVQ packing; caller owns scaling."""

import ctypes
import os
from pathlib import Path

import torch

_lib = ctypes.CDLL(
    os.environ.get("ARVQ_ASSIGN_LIBRARY", str(Path(__file__).with_name("assign.so")))
)
_assign = _lib.assign_launch
_assign.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_translate = _lib.translate_pack_launch
_translate.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
_pack = _lib.pack_launch
_pack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(err):
    if err:
        raise RuntimeError(f"CUDA launch error {err}")


def assign(x, c0, c1, refine=2, threads=256):
    """x:[V,8] float16/32; c0:[256,8], c1:[128,8] float32. Returns uint8 a,b."""
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2 and x.shape[1] == 8
    assert x.dtype in (torch.float16, torch.float32)
    assert c0.shape == (256, 8) and c1.shape == (128, 8)
    assert all(
        c.is_cuda
        and c.is_contiguous()
        and c.dtype == torch.float32
        and c.device == x.device
        for c in (c0, c1)
    )
    assert threads in (128, 256) and refine >= 0
    a = torch.empty(x.shape[0], device=x.device, dtype=torch.uint8)
    b = torch.empty_like(a)
    if len(x):
        with torch.cuda.device(x.device):
            _check(
                _assign(
                    _p(x),
                    _p(c0),
                    _p(c1),
                    _p(a),
                    _p(b),
                    len(x),
                    refine,
                    int(x.dtype == torch.float16),
                    threads,
                    _stream(),
                )
            )
    return a, b


def translate_pack(old, mapping):
    """old:[E,N,K/8] int16/uint16 contiguous, mapping:[65536] uint16 values<32768.
    Returns flat uint32 native RVQ stream with one final guard word.
    No scale translation is performed. N%16 and K%64 must be zero.
    """
    assert (
        old.is_cuda
        and old.is_contiguous()
        and old.ndim == 3
        and old.dtype in (torch.int16, torch.uint16)
    )
    assert (
        mapping.is_cuda
        and mapping.is_contiguous()
        and mapping.shape == (65536,)
        and mapping.dtype == torch.uint16
        and mapping.device == old.device
    )
    E, N, K8 = old.shape
    K = K8 * 8
    assert N % 16 == 0 and K % 64 == 0 and E > 0
    out = torch.empty(
        E * (N // 16) * (K // 64) * 60 + 1, device=old.device, dtype=torch.uint32
    )
    with torch.cuda.device(old.device):
        _check(_translate(_p(old), _p(mapping), _p(out), E, N, K, _stream()))
    return out


def pack(a, b, N, K):
    """Pack one expert's natural uint8 indices [N,K/8] into RVQ fragment stream."""
    assert N % 16 == 0 and K % 64 == 0
    assert (
        all(
            t.is_cuda
            and t.is_contiguous()
            and t.dtype == torch.uint8
            and t.numel() == N * K // 8
            for t in (a, b)
        )
        and a.device == b.device
    )
    out = torch.empty(
        (N // 16) * (K // 64) * 60 + 1, device=a.device, dtype=torch.uint32
    )
    with torch.cuda.device(a.device):
        _check(_pack(_p(a), _p(b), _p(out), N, K, _stream()))
    return out
