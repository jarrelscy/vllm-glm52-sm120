# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental exact fixed-size DCP transfer; default OFF."""

import os

import torch
import torch.distributed as dist

from vllm.v1.attention.ops.dcp_bytepack_kernel import pack
from vllm.v1.attention.ops.dcp_bytepack_selector import select_packed

_shadow_seen = set()
_shadow_failed = False
_logged = set()


def width_for(
    logits,
    ids,
    starts,
    rank,
    world,
    interleave,
    c1_prefill,
    context,
    raw_mode,
    canonical,
    query_split,
):
    if _shadow_failed or os.getenv("VLLM_EXPERIMENT_DCP_BYTEPACK", "0") != "1":
        return 0
    if not (
        c1_prefill
        and 4096 <= context <= 524288
        and world == 4
        and 0 <= rank < 4
        and interleave == 1
        and raw_mode
        and canonical
        and not query_split
        and starts is not None
        and 128 <= ids.shape[0] <= 4096
        and ids.shape[1:] == (2048,)
        and ids.dtype == torch.int32
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] == ids.shape[0]
        and 0 < logits.shape[1] <= 131072
        and starts.shape == (ids.shape[0],)
        and starts.dtype == torch.int32
        and starts.is_contiguous()
        and ids.device == logits.device == starts.device
        and os.getenv("NCCL_MAX_NCHANNELS") == "4"
        and os.getenv("NCCL_BUFFSIZE") == "1048576"
        and torch.cuda.get_device_capability(logits.device) == (12, 0)
        and not torch.cuda.is_current_stream_capturing()
    ):
        return 0
    # Global metadata chooses wire width identically on every rank.
    width = 6 if context <= 131072 else 7
    if width == 6 and logits.shape[1] > 65535:
        return 0
    return width


def _reference(logits, ids, starts, rank, group):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl as baseline_pack,
    )
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl as baseline_select,
    )

    output = ids.clone()
    local = torch.empty((*ids.shape, 2), device=ids.device, dtype=torch.float32)
    baseline_pack(logits, ids, local, rank, 4, 1, starts)
    gathered = torch.empty((4, *local.shape), device=ids.device, dtype=local.dtype)
    dist.all_gather_into_tensor(
        gathered.view(4 * ids.shape[0], 2048, 2), local, group=group.device_group
    )
    baseline_select(gathered, 2048, out=output, canonical=True)
    return output


def _try_merge_ag(
    logits,
    ids,
    starts,
    rank,
    world,
    interleave,
    layer,
    c1_prefill,
    context,
    raw_mode,
    canonical,
    query_split,
    group,
):
    global _shadow_failed
    width = width_for(
        logits,
        ids,
        starts,
        rank,
        world,
        interleave,
        c1_prefill,
        context,
        raw_mode,
        canonical,
        query_split,
    )
    if not width:
        return False
    rows = ids.shape[0]
    shadow_key = (layer, width)
    shadow = (
        os.getenv("VLLM_EXPERIMENT_DCP_BYTEPACK_SHADOW", "0") == "1"
        and shadow_key not in _shadow_seen
    )
    reference = _reference(logits, ids, starts, rank, group) if shadow else None
    local = torch.empty((rows, width * 2048), device=ids.device, dtype=torch.uint8)
    pack[(rows, 4)](
        logits,
        ids,
        starts,
        local,
        logits.stride(0),
        logits.stride(1),
        ids.stride(0),
        ids.stride(1),
        logits.shape[1],
        rank,
        width,
        512,
        num_warps=8,
    )
    gathered = torch.empty(
        (4, rows, width * 2048), device=ids.device, dtype=torch.uint8
    )
    dist.all_gather_into_tensor(
        gathered.view(4 * rows, width * 2048), local, group=group.device_group
    )
    select_packed(gathered, ids)
    if shadow:
        local_equal = torch.equal(ids, reference)
        verdict = torch.tensor(int(local_equal), device=ids.device, dtype=torch.int32)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.device_group)
        passed = bool(verdict.item())
        _shadow_failed = not passed
        _shadow_seen.add(shadow_key)
        ids.copy_(reference)
        print(
            f"DCP_BYTEPACK_SHADOW layer={layer} width={width} rows={rows} "
            f"rank={rank} local_pass={local_equal} all_ranks_pass={passed} "
            "returned_original=True",
            flush=True,
        )
    if rank == 0 and width not in _logged:
        _logged.add(width)
        print(f"DCP_BYTEPACK width={width} rows={rows} context={context}", flush=True)
    return True


_owner_states = {}
_owner_shadow_seen = set()
_owner_shadow_failed = False


def _owner_vote(ok, group):
    # Host-only state: no persistent or temporary CUDA allocation.
    verdict = torch.tensor(int(ok), dtype=torch.int32, device="cpu")
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    return bool(verdict.item())


def _owner_channel(tp, dcp, rank, device):
    from vllm.v1.attention.ops.dcp_bytepack_owner import (
        MAX_GENERATIONS,
        BorrowedOwner,
    )

    key = id(tp)
    if key not in _owner_states:
        candidate = None
        try:
            comm = tp.device_communicator.b12x_ar_comm
            dma = comm._dma
            if (
                tuple(tp.ranks) != tuple(dcp.ranks)
                or len(tp.ranks) != 4
                or tp.rank_in_group != rank
                or dma.rank != rank
                or comm.disabled
                or dma.device != device
            ):
                raise ValueError("TP/DCP channel mismatch")
            candidate = BorrowedOwner(dma)
        except (AttributeError, ImportError, ValueError, TypeError):
            pass
        _owner_states[key] = (
            candidate if _owner_vote(candidate is not None, dcp) else None
        )
    owner = _owner_states[key]
    ok = owner is not None
    if owner is not None:
        dma = tp.device_communicator.b12x_ar_comm._dma
        stream = torch.cuda.current_stream(device).cuda_stream
        ok = (
            dma is owner.dma
            and not dma._closed
            and not owner.busy
            and not owner.poisoned
            and owner.generations < MAX_GENERATIONS
            and (owner.stream is None or owner.stream == stream)
            and not torch.cuda.is_current_stream_capturing()
        )
    # Local-only readiness; the caller combines it with width validity.
    return owner if ok else None


def try_merge(
    logits,
    ids,
    starts,
    rank,
    world,
    interleave,
    layer,
    c1_prefill,
    context,
    raw_mode,
    canonical,
    query_split,
    group,
):
    args = (
        logits,
        ids,
        starts,
        rank,
        world,
        interleave,
        layer,
        c1_prefill,
        context,
        raw_mode,
        canonical,
        query_split,
        group,
    )
    global _owner_shadow_failed
    rows = ids.shape[0]
    # These values come from identical scheduler/model metadata on every rank.
    if not (
        os.getenv("VLLM_EXPERIMENT_DCP_BYTEPACK_OWNER", "0") == "1"
        and not _owner_shadow_failed
        and c1_prefill
        and world == 4
        and 512 <= rows <= 4096
        and 4096 <= context <= 524288
        and interleave == 1
        and raw_mode
        and canonical
        and not query_split
    ):
        return _try_merge_ag(*args)
    # Graph execution mode is uniform across TP ranks. Never perform host
    # collectives while capturing; the outer path retains its original behavior.
    if torch.cuda.is_current_stream_capturing():
        return False
    from vllm.distributed.parallel_state import get_tp_group

    width = width_for(
        logits,
        ids,
        starts,
        rank,
        world,
        interleave,
        c1_prefill,
        context,
        raw_mode,
        canonical,
        query_split,
    )
    owner = _owner_channel(get_tp_group(), group, rank, logits.device)
    # A single MIN vote selects the same branch on every rank:
    # 0 = outer original path; 1 = compressed AG; 2 = owner DMA.
    # Width itself is chosen solely from uniform global scheduler context.
    local_status = 0 if not width else 1 if owner is None else 2
    status = torch.tensor(local_status, dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group.cpu_group)
    decision = int(status.item())
    if decision == 0:
        return False
    if decision == 1:
        return _try_merge_ag(*args)
    assert owner is not None
    shadow_key = (layer, width)
    shadow = os.getenv("VLLM_EXPERIMENT_DCP_BYTEPACK_OWNER_SHADOW", "0") == "1"
    reference = ids.clone() if shadow else None
    if shadow:
        assert _try_merge_ag(
            logits,
            reference,
            starts,
            rank,
            world,
            interleave,
            layer,
            c1_prefill,
            context,
            raw_mode,
            canonical,
            query_split,
            group,
        )
    local = torch.empty((rows, width * 2048), device=ids.device, dtype=torch.uint8)
    pack[(rows, 4)](
        logits,
        ids,
        starts,
        local,
        logits.stride(0),
        logits.stride(1),
        ids.stride(0),
        ids.stride(1),
        logits.shape[1],
        rank,
        width,
        512,
        num_warps=8,
    )
    padded = (rows + 3) // 4

    def select(received):
        result = torch.empty((padded, 2048), device=ids.device, dtype=torch.int32)
        select_packed(received, result[: received.shape[1]])
        return result

    selected = owner.run(local, width, select)
    del local
    gathered = torch.empty((4 * padded, 2048), device=ids.device, dtype=torch.int32)
    dist.all_gather_into_tensor(gathered, selected, group=group.device_group)
    offset = 0
    for peer in range(4):
        count = rows // 4 + int(peer < rows % 4)
        ids[offset : offset + count].copy_(
            gathered[peer * padded : peer * padded + count]
        )
        offset += count
    if shadow:
        local_pass = torch.equal(ids, reference)
        passed = _owner_vote(local_pass, group)
        _owner_shadow_failed = not passed
        ids.copy_(reference)
        if shadow_key not in _owner_shadow_seen or not passed:
            _owner_shadow_seen.add(shadow_key)
            print(
                f"DCP_OWNER_SHADOW layer={layer} rank={rank} rows={rows} width={width} "
                f"local_pass={local_pass} all_ranks_pass={passed} "
                "returned_original=True",
                flush=True,
            )
    return True
