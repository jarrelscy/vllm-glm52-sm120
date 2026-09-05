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
    """

    def __init__(self, group, input_: torch.Tensor, dim: int):
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
        self._work = dist.all_gather_into_tensor(
            self._output,
            input_,
            group=group.device_group,
            async_op=True,
        )

    def wait(self) -> torch.Tensor:
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
