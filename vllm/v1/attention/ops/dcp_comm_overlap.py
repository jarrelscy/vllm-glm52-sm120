# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async DCP collective scheduling for GLM/DSA sparse-MLA decode.

Motivation (rank0 decode trace, GLM-5.3-hybrid TP4+DCP4+MTP ns=3, batch=1):
~440 NCCL RING_LL kernels per decode step (~5.7 ms) execute with ZERO
measured overlap against compute -- every collective is fully exposed on
the critical path. Most of the chain is dependency-locked, but two windows
around the per-layer DCP all-gathers contain genuinely independent work:

  1. AG(mqa_q): the top-k -> physical-slot index conversion
     (``triton_filter_and_convert_dcp_index`` + its fills) and the
     ``empty_rows`` mask reduction depend only on attention metadata and the
     (already final) ``topk_indices_buffer`` -- NOT on the gathered query.
     Today they run after the AG; they can run while it is in flight.

  2. AG(indexer top-k candidates): the CuteDSL stable-topk merge needs the
     gathered candidates, but the query-side work between the indexer and
     the attention core (q up-proj bmm, rope concat, q AG launch) does not
     need the merge. Deferring the merge to just before the index
     conversion lets AG(idx) overlap the q-side compute and lets AG(q)
     overlap the ~21us merge kernel.

Both are pure scheduling changes: the collectives move the same bytes, and
every kernel runs with identical inputs, so results are bit-exact vs the
serial order (validated in tests/kernels/attention/test_dcp_comm_overlap.py).

CUDA-graph capture: the async collective is the exact enqueue the sync path
uses (``dist.all_gather_into_tensor`` on the process group), with the
caller-stream sync (``work.wait()``) issued after the independent compute
instead of immediately. Under FULL/FULL_AND_PIECEWISE capture the deferred
wait records the same kind of cross-stream event edge the sync path records,
so the schedule is baked into the graph and replays correctly. The
stash/consume protocol below runs at capture time (or every step in eager
mode); a deferred merge is always consumed by the attention op of the same
layer, inside the same graph.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def dcp_comm_overlap_enabled() -> bool:
    return envs.VLLM_GLM_COMM_OVERLAP


def dcp_comm_coalesce_enabled() -> bool:
    """VLLM_GLM_COMM_COALESCE: launch the per-layer DCP AG(idx candidates) and
    AG(mqa q) in ONE NCCL group call instead of two separate enqueues.

    The indexer's candidates all-gather (fp32 packed payload -- the only fp32
    AsyncAllGather in the stack, asserted below) is *deferred* at construction
    and flushed together with the query all-gather inside a single
    ``dist._coalescing_manager`` group. All-gathers move bytes without any
    reduction, so grouping cannot change results: every output buffer is
    bit-identical to the two-launch schedule. Requires VLLM_GLM_COMM_OVERLAP.
    """
    return envs.VLLM_GLM_COMM_COALESCE and envs.VLLM_GLM_COMM_OVERLAP


# The single deferred (not-yet-launched) AsyncAllGather, if any. The very next
# AsyncAllGather construction with flush_coalesce=True (the same layer's query
# gather) launches both in one NCCL group. wait() on a still-deferred gather
# self-heals by launching it solo.
_PENDING_COALESCE: list["AsyncAllGather"] = []


class AsyncAllGather:
    """Concat-style all-gather with a deferred wait.

    Replicates ``DeviceCommunicatorBase.all_gather`` (the path taken by
    ``get_dcp_group().all_gather`` for dim != 0) exactly -- same
    ``dist.all_gather_into_tensor`` call, same output allocation, same
    reshape/movedim epilogue -- but returns before blocking the current
    stream, so independent kernels can be issued while NCCL runs.

    ``wait()`` blocks the current stream on the collective (event edge under
    graph capture) and applies the identical reshape epilogue, producing a
    tensor bit-identical to the sync path's.

    With VLLM_GLM_COMM_COALESCE, fp32 payloads (the DCP indexer's packed
    top-k candidates) are deferred and launched inside one NCCL group
    together with the next ``flush_coalesce=True`` gather (the query).
    """

    def __init__(
        self,
        group,
        input_: torch.Tensor,
        dim: int,
        flush_coalesce: bool = False,
    ):
        world_size = group.world_size
        assert world_size > 1
        if dim < 0:
            dim += input_.dim()
        # dim == 0 may be routed through NCCL symmetric memory by
        # CudaCommunicator.all_gather; this helper only mirrors the base ring
        # path. All DCP-decode gathers use dim=1.
        assert dim != 0, "AsyncAllGather only supports dim != 0"
        self._world_size = world_size
        self._dim = dim
        self._input_size = input_.size()
        output_size = (self._input_size[0] * world_size,) + self._input_size[1:]
        self._output = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        self._group = group
        self._work = None
        self._cm = None
        # Kept alive only while the launch is deferred (NCCL must read the
        # input when the group is eventually launched).
        self._deferred_input: torch.Tensor | None = None

        if (
            dcp_comm_coalesce_enabled()
            and not flush_coalesce
            and input_.dtype == torch.float32
        ):
            # Indexer packed-candidates gather (fp32 by contract): defer.
            if _PENDING_COALESCE:
                # Unexpected: no query gather consumed the previous deferred
                # candidates gather. Self-heal by launching it solo.
                logger.error(
                    "dcp_comm_coalesce: stale deferred all-gather at defer "
                    "time; launching it solo. Unexpected op ordering."
                )
                _PENDING_COALESCE.pop()._launch_solo()
            self._deferred_input = input_
            _PENDING_COALESCE.append(self)
            return

        if flush_coalesce and _PENDING_COALESCE:
            # Launch the deferred candidates gather and this query gather in
            # a single NCCL group call. Identical collectives, one launch.
            other = _PENDING_COALESCE.pop()
            with dist._coalescing_manager(
                group=group.device_group, async_ops=True
            ) as cm:
                dist.all_gather_into_tensor(
                    other._output,
                    other._deferred_input,
                    group=other._group.device_group,
                )
                dist.all_gather_into_tensor(
                    self._output, input_, group=group.device_group
                )
            other._deferred_input = None
            other._cm = cm
            self._cm = cm
            return

        self._work = dist.all_gather_into_tensor(
            self._output,
            input_,
            group=group.device_group,
            async_op=True,
        )

    def _launch_solo(self) -> None:
        assert self._deferred_input is not None and self._work is None
        self._work = dist.all_gather_into_tensor(
            self._output,
            self._deferred_input,
            group=self._group.device_group,
            async_op=True,
        )
        self._deferred_input = None

    def wait(self) -> torch.Tensor:
        if self._deferred_input is not None:
            # Deferred but never coalesced (e.g. exception between indexer
            # and attention op): self-heal with a solo launch.
            if self in _PENDING_COALESCE:
                _PENDING_COALESCE.remove(self)
            self._launch_solo()
        if self._cm is not None:
            self._cm.wait()
            self._cm = None
        if self._work is not None:
            self._work.wait()
            self._work = None
        input_size = self._input_size
        dim = self._dim
        output_tensor = self._output.reshape((self._world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self._world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor


# ---------------------------------------------------------------------------
# Pending deferred DCP top-k merge (indexer -> attention handoff).
#
# The sparse indexer custom op stashes a zero-arg closure that (a) waits on
# the candidates all-gather and (b) runs the CuteDSL stable-topk merge,
# writing the final global top-k into topk_indices_buffer. The MLA attention
# op of the same layer consumes it before converting indices. A single slot
# suffices: the very next attention op always consumes the pending merge
# before the next indexer op can stash a new one.
# ---------------------------------------------------------------------------

_PENDING_DCP_MERGE: list[Callable[[], None]] = []


def stash_pending_dcp_merge(finish: Callable[[], None]) -> None:
    if _PENDING_DCP_MERGE:
        # Should be unreachable: every stash is consumed by the same layer's
        # attention op. Self-heal by completing the stale merge (lossless --
        # it is the same kernels, just later than intended) and log loudly.
        logger.error(
            "dcp_comm_overlap: stale pending DCP top-k merge at stash time; "
            "completing it now. This indicates an unexpected op ordering."
        )
        consume_pending_dcp_merge()
    _PENDING_DCP_MERGE.append(finish)


def consume_pending_dcp_merge() -> None:
    """Complete a deferred indexer top-k merge, if any. No-op otherwise."""
    if _PENDING_DCP_MERGE:
        finish = _PENDING_DCP_MERGE.pop()
        finish()
