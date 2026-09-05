# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer by exchanging
the partial output and LSE in a single packed All-to-All payload.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _dcp_a2a_send_recv_buffers(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Don't use the shared WorkspaceManager here. A FULL cudagraph bakes in the
    # buffer address at capture, but the workspace is growable and sized only to
    # the largest *captured* batch (the cudagraph capture cap). Any eager a2a
    # with a bigger batch regrows it, freeing that address and poisoning every
    # captured graph -> illegal memory access on replay. This bites the very
    # first request: the post-capture warmup runs an eager decode at
    # max_num_seqs (> the cap), so the graphs are already dangling before the
    # server is ready. torch.empty buffers instead live in the graph's private
    # pool and stay valid for its lifetime (as _dcp_a2a_unpack_combine and the
    # AG+RS combine path already rely on).
    return (
        torch.empty(shape, device=device, dtype=dtype),
        torch.empty(shape, device=device, dtype=dtype),
    )


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )

        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        )
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        acc += (
            tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(tl.float32)
            * weight
        )

    final_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
) -> None:
    grid = (cp_attn_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    return_lse: bool,
    is_lse_base_on_e: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
    out = torch.empty(
        (num_tokens, h_per_rank, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    out_lse = torch.empty(
        (num_tokens, h_per_rank) if return_lse else (1, 1),
        device=recv_buffer.device,
        dtype=torch.float32 if return_lse else recv_buffer.dtype,
    )
    grid = (num_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        out,
        out_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
        LSE_PACK_DIM=lse_pack_dim,
    )
    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    The output and fp32 LSE are packed into a single output-dtype buffer, sent
    with one All-to-All, then unpacked and combined with exact LSE weighting.

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )

    _dcp_a2a_pack_send(
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
    )

    work = dist.all_to_all_single(
        recv_buffer.view(-1),
        send_buffer.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work.wait()

    return _dcp_a2a_unpack_combine(
        recv_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
    )


# =============================================================================
# Exact (bit-identical to AG+RS) All-to-All combine -- VLLM_DCP_A2A_EXACT
# =============================================================================
#
# The default a2a combine above accumulates the LSE-weighted outputs in fp32,
# which is *more* accurate than the ag_rs baseline but not bit-identical to it
# (the baseline rounds each rank's corrected output to the output dtype and
# then lets NCCL ReduceScatter sum those values in ring order with a rounding
# after every hop). The exact variant below reproduces the ag_rs numerics
# bit-for-bit:
#
#   1. The global LSE and each rank's correction factor are computed with the
#      exact op sequence of _correct_attn_cp_out_kernel (common.py), on the
#      exact same inputs (raw bf16 partial outputs + raw fp32 LSEs carried
#      losslessly through the packed a2a payload).
#   2. Each rank's corrected output is rounded to the output dtype -- the
#      same rounding the baseline's in-place Triton store performs.
#   3. The rounded values are summed in the *NCCL ReduceScatter reduction
#      order*, rounding to the output dtype after every partial sum (NCCL
#      reduces in the wire dtype, one rounding per ring hop).
#
# The reduction order is discovered at runtime by probing the very same
# GroupCoordinator.reduce_scatter path with the real tensor shape/dtype:
# random wide-exponent values make every distinct reduction tree produce a
# distinct bit pattern somewhere in the buffer, so the matching tree is
# unique. The probe then self-validates by comparing the full exact-a2a
# epilogue against the AG+RS epilogue bitwise on several random trials and
# refuses to enable itself on any mismatch. Probing (and validation) runs
# once per (group, shape, dtype) outside CUDA-graph capture; the captured
# graph replays only the pack kernel + all_to_all_single + combine kernel.

_EXACT_TREES: list[tuple[int, tuple[int, int, int, int]]] = []


def _enumerate_exact_trees() -> list[tuple[int, tuple[int, int, int, int]]]:
    """All distinct 4-leaf reduction trees modulo add commutativity.

    Encoding: (bal, (p0, p1, p2, p3)).
      bal == 0: ((p0 + p1) + p2) + p3   (sequential / ring)
      bal == 1: (p0 + p1) + (p2 + p3)   (balanced / tree)
    Commutativity dedupe: p0 < p1 (both kinds); bal==1 also p2 < p3 and
    p0 == min over both pairs.
    """
    global _EXACT_TREES
    if _EXACT_TREES:
        return _EXACT_TREES
    import itertools

    trees: list[tuple[int, tuple[int, int, int, int]]] = []
    for perm in itertools.permutations(range(4)):
        p0, p1, p2, p3 = perm
        if p0 < p1:
            trees.append((0, perm))  # 12 sequential classes
    for perm in itertools.permutations(range(4)):
        p0, p1, p2, p3 = perm
        if p0 < p1 and p2 < p3 and p0 < p2:
            trees.append((1, perm))  # 3 balanced classes
    _EXACT_TREES = trees
    return trees


def _apply_tree_fp(
    vals: list[torch.Tensor], tree: tuple[int, tuple[int, int, int, int]]
) -> torch.Tensor:
    """Reference (torch) evaluation of a reduction tree with per-node rounding
    to the value dtype. ``vals`` are output-dtype tensors."""
    bal, (p0, p1, p2, p3) = tree
    dt = vals[0].dtype

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(dt)

    if bal == 0:
        return add(add(add(vals[p0], vals[p1]), vals[p2]), vals[p3])
    return add(add(vals[p0], vals[p1]), add(vals[p2], vals[p3]))


# (group_name, world, B, H, D, dtype) -> (bal, (p0, p1, p2, p3))
_EXACT_ORDER_CACHE: dict[tuple, tuple[int, tuple[int, int, int, int]]] = {}


@triton.jit
def _dcp_a2a_exact_load_lse(
    recv_ptr,
    recv_offset,
    recv_stride_D,
    HEAD_DIM: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """Load a raw fp32 LSE carried in the packed a2a payload (bit-lossless)."""
    if LSE_PACK_DIM == 1:
        lse_val = tl.load(recv_ptr + recv_offset + HEAD_DIM * recv_stride_D).to(
            tl.float32, bitcast=True
        )
    else:
        lo_raw = tl.load(recv_ptr + recv_offset + HEAD_DIM * recv_stride_D)
        hi_raw = tl.load(recv_ptr + recv_offset + (HEAD_DIM + 1) * recv_stride_D)
        lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
    return lse_val


@triton.jit
def _dcp_a2a_exact_corrected(
    recv_ptr,
    recv_base,
    recv_stride_D,
    d_offsets,
    global_lse,
    HEAD_DIM: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """One rank's corrected partial output, rounded to the output dtype --
    replicates _correct_attn_cp_out_kernel's tail (load raw lse, subtract the
    global lse, inf/nan guard, exp, multiply, zero-fill, store-rounding)."""
    lse_tmp = _dcp_a2a_exact_load_lse(
        recv_ptr, recv_base, recv_stride_D, HEAD_DIM, LSE_PACK_DIM
    )
    lse_finally = lse_tmp - global_lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)
    # The baseline stores to the (bf16/fp16) output tensor here; replicate the
    # store rounding, then return as fp32 for the wire-order summation.
    return output.to(recv_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _dcp_a2a_unpack_combine_exact_kernel(
    recv_ptr,
    out_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    TREE_BAL: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    # ---- global LSE: exact op sequence of _correct_attn_cp_out_kernel ----
    num_n_offsets = tl.arange(0, N)
    lse_bases = (
        num_n_offsets * recv_stride_N
        + batch_idx * recv_stride_B
        + head_idx * recv_stride_H
    )
    if LSE_PACK_DIM == 1:
        lse = tl.load(recv_ptr + lse_bases + HEAD_DIM * recv_stride_D).to(
            tl.float32, bitcast=True
        )
    else:
        lo_raw = tl.load(recv_ptr + lse_bases + HEAD_DIM * recv_stride_D)
        hi_raw = tl.load(recv_ptr + lse_bases + (HEAD_DIM + 1) * recv_stride_D)
        lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse = (lo | (hi << 16)).to(tl.float32, bitcast=True)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        glse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        glse = tl.log2(lse_acc)
    glse += lse_max

    # ---- per-rank corrected partials, rounded to the output dtype ----
    base_p0 = P0 * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
    base_p1 = P1 * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
    base_p2 = P2 * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
    base_p3 = P3 * recv_stride_N + batch_idx * recv_stride_B + head_idx * recv_stride_H
    v0 = _dcp_a2a_exact_corrected(
        recv_ptr, base_p0, recv_stride_D, d_offsets, glse,
        HEAD_DIM, LSE_PACK_DIM, IS_BASE_E,
    )
    v1 = _dcp_a2a_exact_corrected(
        recv_ptr, base_p1, recv_stride_D, d_offsets, glse,
        HEAD_DIM, LSE_PACK_DIM, IS_BASE_E,
    )
    v2 = _dcp_a2a_exact_corrected(
        recv_ptr, base_p2, recv_stride_D, d_offsets, glse,
        HEAD_DIM, LSE_PACK_DIM, IS_BASE_E,
    )
    v3 = _dcp_a2a_exact_corrected(
        recv_ptr, base_p3, recv_stride_D, d_offsets, glse,
        HEAD_DIM, LSE_PACK_DIM, IS_BASE_E,
    )

    # ---- wire-order summation: one output-dtype rounding per partial sum ----
    out_dt = recv_ptr.dtype.element_ty
    if TREE_BAL == 1:
        sa = (v0 + v1).to(out_dt).to(tl.float32)
        sb = (v2 + v3).to(out_dt).to(tl.float32)
        acc = sa + sb
    else:
        acc = (v0 + v1).to(out_dt).to(tl.float32)
        acc = (acc + v2).to(out_dt).to(tl.float32)
        acc = acc + v3

    final_offsets = (
        batch_idx * out_stride_B
        + head_idx * out_stride_H
        + d_offsets * out_stride_D
    )
    # Final store rounds once -- same as the last NCCL hop's wire rounding.
    tl.store(out_ptr + final_offsets, acc)


def _dcp_a2a_unpack_combine_exact(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    is_lse_base_on_e: bool,
    tree: tuple[int, tuple[int, int, int, int]],
) -> torch.Tensor:
    world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
    bal, (p0, p1, p2, p3) = tree
    out = torch.empty(
        (num_tokens, h_per_rank, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    grid = (num_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_exact_kernel[grid](
        recv_buffer,
        out,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        LSE_PACK_DIM=lse_pack_dim,
        TREE_BAL=bal,
        P0=p0,
        P1=p1,
        P2=p2,
        P3=p3,
    )
    return out


def _probe_rs_reduction_order(
    cp_group: GroupCoordinator,
    B: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0x5EED,
) -> tuple[int, tuple[int, int, int, int]]:
    """Discover this rank's NCCL ReduceScatter reduction order.

    Runs the *production* GroupCoordinator.reduce_scatter path on a
    same-shape/dtype buffer of wide-exponent random values, gathers every
    rank's raw input, and finds the unique 4-leaf reduction tree whose
    per-node-rounded evaluation matches the received chunk everywhere.
    """
    world = cp_group.world_size
    assert world == 4, "VLLM_DCP_A2A_EXACT currently supports DCP world size 4"
    rank = cp_group.rank_in_group

    gen = torch.Generator(device="cpu").manual_seed(seed + 7919 * rank)
    mant = torch.rand((B, H, D), generator=gen, dtype=torch.float32) * 1.9 + 0.05
    sign = torch.where(
        torch.rand((B, H, D), generator=gen) < 0.5, -1.0, 1.0
    )
    expo = torch.randint(-6, 7, (B, H, D), generator=gen).to(torch.float32)
    x = (mant * sign * torch.pow(torch.tensor(2.0), expo)).to(dtype).to(device)

    # Gather every rank's raw input (order fidelity of this gather is
    # irrelevant -- values are exact).
    xs = torch.empty((world, B, H, D), dtype=dtype, device=device)
    dist.all_gather_into_tensor(
        xs.view(world, -1), x.view(-1).contiguous(), group=cp_group.device_group
    )

    # The very collective the ag_rs epilogue issues.
    got = cp_group.reduce_scatter(x, dim=1)  # [B, H/world, D]

    h_lo = rank * (H // world)
    h_hi = (rank + 1) * (H // world)
    vals = [xs[r, :, h_lo:h_hi, :] for r in range(world)]

    matches = []
    for tree in _enumerate_exact_trees():
        ref = _apply_tree_fp(vals, tree)
        if torch.equal(ref, got):
            matches.append(tree)
    if len(matches) != 1:
        raise RuntimeError(
            "VLLM_DCP_A2A_EXACT: could not identify a unique NCCL "
            f"ReduceScatter reduction order (rank {rank}, shape "
            f"[{B},{H},{D}], dtype {dtype}): {len(matches)} of "
            f"{len(_enumerate_exact_trees())} candidate trees matched. "
            "The collective is likely split across channels with different "
            "ring orders at this message size; the exact a2a path cannot "
            "guarantee bit-exactness here. Unset VLLM_DCP_A2A_EXACT."
        )
    return matches[0]


def _validate_exact_a2a(
    cp_group: GroupCoordinator,
    B: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    is_lse_base_on_e: bool,
    tree: tuple[int, tuple[int, int, int, int]],
    trials: int = 4,
) -> None:
    """Bitwise self-gate: full exact-a2a epilogue vs the AG+RS epilogue."""
    from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

    rank = cp_group.rank_in_group
    for t in range(trials):
        gen = torch.Generator(device="cpu").manual_seed(0xA2A + 104729 * rank + t)
        out = (
            torch.randn((B, H, D), generator=gen, dtype=torch.float32)
            * torch.pow(
                torch.tensor(2.0),
                torch.randint(-4, 5, (B, H, 1), generator=gen).to(torch.float32),
            )
        ).to(dtype).to(device)
        lse = (
            torch.randn((B, H), generator=gen, dtype=torch.float32) * 4.0
        ).to(device)
        # A handful of adversarial LSEs: +/-inf and NaN rows.
        if t == 0 and B > 0:
            lse.view(-1)[0] = float("inf")
            if lse.numel() > 1:
                lse.view(-1)[1] = float("-inf")
            if lse.numel() > 2:
                lse.view(-1)[2] = float("nan")

        ref = cp_lse_ag_out_rs(
            out.clone(), lse.clone(), cp_group, is_lse_base_on_e=is_lse_base_on_e
        )
        test = _dcp_a2a_lse_reduce_exact_impl(
            out.clone(), lse.clone(), cp_group, is_lse_base_on_e, tree
        )
        if not torch.equal(ref, test):
            diff = (ref.float() - test.float()).abs()
            raise RuntimeError(
                "VLLM_DCP_A2A_EXACT: bitwise self-validation FAILED on trial "
                f"{t} (rank {rank}, shape [{B},{H},{D}], dtype {dtype}): "
                f"{(ref != test).sum().item()} mismatched elements, max |d| "
                f"{diff.max().item():.3e}. Refusing to enable the exact a2a "
                "path. Unset VLLM_DCP_A2A_EXACT."
            )


def _dcp_a2a_lse_reduce_exact_impl(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    is_lse_base_on_e: bool,
    tree: tuple[int, tuple[int, int, int, int]],
) -> torch.Tensor:
    world_size = cp_group.world_size
    B, H, D = cp_attn_out.shape
    H_per_rank = H // world_size
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )
    _dcp_a2a_pack_send(
        cp_attn_out,
        cp_attn_lse.contiguous(),
        send_buffer,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
    )
    work = dist.all_to_all_single(
        recv_buffer.view(-1),
        send_buffer.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work.wait()
    return _dcp_a2a_unpack_combine_exact(
        recv_buffer, D, lse_pack_dim, is_lse_base_on_e, tree
    )


def dcp_a2a_exact_max_tokens() -> int:
    """Token cap for the exact-a2a epilogue. Decode batches (padded to the
    cudagraph capture sizes, <= 16 in prod) stay far below it; large sparse-MLA
    prefill batches fall back to the baseline ag_rs path (trivially lossless)
    so the probe never allocates gather buffers at prefill sizes."""
    return int(os.getenv("VLLM_DCP_A2A_EXACT_MAX_TOKENS", "64"))


def dcp_a2a_lse_reduce_exact(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """AG(lse)+ReduceScatter(out) replacement: one packed All-to-All whose
    combine is bit-identical to the ag_rs epilogue. See module comment above.

    Lazily probes + self-validates the NCCL reduction order per
    (group, shape, dtype) the first time each shape is seen; must first be
    called outside CUDA-graph capture (vLLM's eager warmup satisfies this).

    Batches larger than VLLM_DCP_A2A_EXACT_MAX_TOKENS (prefill via the sparse
    MLA path) use the baseline ag_rs epilogue unchanged.
    """
    world_size = cp_group.world_size
    if world_size == 1:
        return cp_attn_out

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")

    if B > dcp_a2a_exact_max_tokens():
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        return cp_lse_ag_out_rs(
            cp_attn_out, cp_attn_lse, cp_group, ctx, is_lse_base_on_e=is_lse_base_on_e
        )

    key = (cp_group.unique_name, world_size, B, H, D, cp_attn_out.dtype)
    tree = _EXACT_ORDER_CACHE.get(key)
    if tree is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VLLM_DCP_A2A_EXACT: reduction-order probe for shape "
                f"[{B},{H},{D}] requested during CUDA-graph capture. The "
                "shape must be seen once in eager mode first (run an eager "
                "warmup for every capture size)."
            )
        tree = _probe_rs_reduction_order(
            cp_group, B, H, D, cp_attn_out.dtype, cp_attn_out.device
        )
        _validate_exact_a2a(
            cp_group,
            B,
            H,
            D,
            cp_attn_out.dtype,
            cp_attn_out.device,
            is_lse_base_on_e,
            tree,
        )
        _EXACT_ORDER_CACHE[key] = tree
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "VLLM_DCP_A2A_EXACT enabled: NCCL RS order probed and bitwise "
            "self-validation passed (first shape [%d,%d,%d], tree %s).",
            B,
            H,
            D,
            str(tree),
        )

    return _dcp_a2a_lse_reduce_exact_impl(
        cp_attn_out, cp_attn_lse, cp_group, is_lse_base_on_e, tree
    )
