# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility for exact fused hot gate gather/half/P4 packing."""

import os

import torch


def eligible(x, route_slots, top_k, use_pairs):
    return (
        os.environ.get("VLLM_ARVQ_FUSED_GATE_PACK") == "1"
        and os.environ.get("VLLM_ARVQ_WIDE_HOT_PREFILL") == "1"
        and use_pairs
        and top_k == 8
        and x.is_cuda
        and route_slots.is_cuda
        and x.device == route_slots.device
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == 6144
        and x.is_contiguous()
        and route_slots.dtype == torch.int64
        and route_slots.ndim == 1
        and route_slots.is_contiguous()
        and 0 < route_slots.numel() <= 1024
    )
