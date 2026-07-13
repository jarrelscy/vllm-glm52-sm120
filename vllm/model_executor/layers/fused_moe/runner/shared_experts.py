# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)

# --- shared-experts slot diagnostics (2026-07-13, see forward()) ---
import collections as _collections
import itertools as _itertools
import os as _os

_SE_DEBUG = int(_os.environ.get("GLM_SHARED_EXPERTS_DEBUG", "0"))
_SE_HEAL = int(_os.environ.get("GLM_SHARED_EXPERTS_HEAL", "1"))
_SE_EVENTS: "_collections.deque" = _collections.deque(maxlen=2048)
_SE_DUMPS = 0
_SE_IDS = _itertools.count()


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._layer = layer
        self._moe_config = moe_config
        self._se_id = next(_SE_IDS)  # instance id for slot diagnostics

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    # ------------------------------------------------------------------ #
    # DIAGNOSTIC instrumentation (2026-07-13): the _output slot is being
    # left undrained under TP4+DCP4+MTP+hybrid modular-MoE, so a later
    # forward() hits a dirty slot (upstream: fatal assert -> engine death).
    # Every slot event (SET / DRAIN / SKIP / DIRTY) is recorded in a global
    # ring buffer with full context; on DIRTY we dump the recent history +
    # current stack. GLM_SHARED_EXPERTS_DEBUG>=2 additionally captures the
    # python stack at every SET (costly; only for deep tracing).
    # GLM_SHARED_EXPERTS_HEAL=1 (default) clears the stale slot and
    # recomputes after dumping — the output is a pure function of the
    # current input, so this is lossless and keeps the engine alive to
    # collect MANY leak samples per run; set =0 for the original hard
    # assert (one sample, then crash).
    # ------------------------------------------------------------------ #

    def _se_ctx(self, event: str, order=None, experts_order=None, shape0=None):
        import time as _t
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = "?"
        stream = None
        try:
            stream = torch.cuda.current_stream().stream_id
        except Exception:
            pass
        fwd = None
        try:
            from vllm.forward_context import get_forward_context
            fc = get_forward_context()
            am = getattr(fc, "attn_metadata", None)
            fwd = {
                "num_tokens": getattr(fc, "num_tokens", None),
                "cudagraph_runtime_mode": str(
                    getattr(fc, "cudagraph_runtime_mode", None)),
                "attn_meta": type(am).__name__ if am is not None else None,
            }
        except Exception:
            pass
        ev = {
            "t": round(_t.monotonic(), 6), "ev": event, "inst": self._se_id,
            "idx": self._output_idx, "dbo": self.enable_dbo,
            "order": None if order is None else int(order),
            "det": None if experts_order is None else int(experts_order),
            "shape0": shape0, "capturing": capturing, "stream": stream,
            "fwd": fwd,
            "slots": [x is not None for x in self._output],
        }
        if _SE_DEBUG >= 2 and event == "SET":
            import traceback as _tb
            ev["stack"] = "".join(_tb.format_stack(limit=10)[:-1])
        _SE_EVENTS.append(ev)
        return ev

    def _se_dump(self, reason: str, current_ev):
        import traceback as _tb
        global _SE_DUMPS
        _SE_DUMPS += 1
        if _SE_DUMPS > 20:  # cap full dumps; keep counting
            logger.error("shared_experts %s #%d (dump suppressed after 20)",
                         reason, _SE_DUMPS)
            return
        lines = [f"shared_experts {reason} #{_SE_DUMPS} — current={current_ev}"]
        lines.append("---- last slot events (most recent last) ----")
        for e in list(_SE_EVENTS)[-80:]:
            lines.append(repr(e))
        lines.append("---- current python stack ----")
        lines.append("".join(_tb.format_stack(limit=24)[:-2]))
        logger.error("\n".join(lines))

    @property
    def output(self) -> torch.Tensor:
        if _SE_DEBUG:
            ev = self._se_ctx("DRAIN")
            if self._output[self._output_idx] is None:
                self._se_dump("DRAIN-OF-EMPTY-SLOT", ev)
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            if _SE_DEBUG:
                self._se_ctx("SKIP", order, experts_order,
                             shared_experts_input.shape[0])
            return None

        if self._output[self._output_idx] is not None:
            ev = self._se_ctx("DIRTY", order, experts_order,
                              shared_experts_input.shape[0]) \
                if _SE_DEBUG else None
            if _SE_DEBUG:
                self._se_dump("DIRTY-SLOT-AT-SET", ev)
            if not _SE_HEAL:
                assert self._output[self._output_idx] is None
            self._output[self._output_idx] = None

        if _SE_DEBUG:
            self._se_ctx("SET", order, experts_order,
                         shared_experts_input.shape[0])

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
