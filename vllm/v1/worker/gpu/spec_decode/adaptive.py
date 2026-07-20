# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive per-request speculative-decoding suspension policy.

Lossless by construction: the policy only decides WHETHER a request
drafts; every emitted token is still verifier-approved, so generated
text is identical with the policy on or off (temp-0 equivalence).

Motivation (Inkling MTP ns=2 on 4x RTX PRO 6000): a draft+verify round
costs ~2x a plain decode step (~50ms vs ~24.5ms), so it only pays off
when the mean accepted-per-round (incl. the bonus token, out of ns+1)
exceeds ~2.05. Draft-friendly content (counting ~3.0, code ~2.6-2.7)
wins; draft-hostile prose (~1.7-2.1) decodes SLOWER than the no-MTP
baseline (~33-42 vs ~40.8 tok/s). This policy tracks a per-request EMA
of acceptance and suspends drafting for requests that do not pay for
their drafts. Suspended requests take plain 1-token decode
steps -- still within the FULL_DECODE_ONLY captured batch shapes.

Mechanics:
  - ``decide()`` runs on the host right before ``speculator.propose``.
    It consumes the previous round's acceptance counts (async-copied,
    so it never stalls on the GPU) and returns:
      * ``skip_propose``: True when EVERY request in the batch is
        suspended and the batch is pure 1-token decode -- the entire
        drafter forward (the expensive part) is skipped.
      * ``valid_counts``: per-request number of draft tokens to expose
        to the scheduler (0 for suspended requests in mixed batches;
        the drafter still runs for the batch in that case).
  - ``observe()`` snapshots this round's per-request acceptance
    (``num_sampled`` = accepted drafts + 1 bonus) with a non-blocking
    D2H copy, consumed by the NEXT ``decide()`` (one round of lag).
  - After ``suspend_len`` suspended rounds the request is re-probed for
    ``probe_len`` rounds; a failed probe multiplies ``suspend_len`` by
    ``backoff`` (capped), so probe overhead is geometrically bounded.

Caveat (drafter KV holes): the Inkling MTP drafter maintains its own KV
cache via per-round re-prefill of the current query window only. Rounds
where propose is skipped leave draft-KV holes at those positions, which
degrades draft quality after resume, so probe acceptance after a long
suspension is pessimistic and suspension is, in practice, close to
one-way per request. That is the intended trade: hostile requests never
decode materially slower than the no-MTP baseline, while requests that
never trigger suspension keep the full MTP win. A preemption/re-prefill
fully heals the drafter KV. All decisions are pure functions of
sampler outputs that are identical on every TP rank, so ranks stay in
lockstep (propose is a TP collective and must be skipped by all ranks
or none).

Trigger statistic: an exponential moving average of accepted-per-round
rather than a short rolling-window mean. Measured on the 3-workload gate
(2026-07-19), a 10-round window mean spuriously dips below threshold on
code (locally draft-hostile docstring stretches inside globally friendly
content) in 3 of 5 passes; the EMA (halflife ~32 rounds) tracks the
long-run rate (code ~2.66) and only crosses the threshold for content
that is hostile for ~1.5x the halflife or longer.

Enable with INKLING_ADAPTIVE_SPEC=1 (default OFF). Tunables (env):
  INKLING_ADAPTIVE_SPEC_MIN_ROUNDS=10    drafted rounds before any decision
  INKLING_ADAPTIVE_SPEC_EMA_HALFLIFE=32  EMA halflife (drafted rounds)
  INKLING_ADAPTIVE_SPEC_THRESHOLD        suspend when EMA accepted/round < T;
                                         default MARGIN*(1 + RATIO*ns)
                                         (~2.06 at ns=2)
  INKLING_ADAPTIVE_SPEC_DRAFT_COST_RATIO=0.48  per-draft-pass cost / plain step
  INKLING_ADAPTIVE_SPEC_MARGIN=1.05      threshold safety margin
  INKLING_ADAPTIVE_SPEC_RESUME_THRESHOLD  probe pass bar; default T*1.1
  INKLING_ADAPTIVE_SPEC_SUSPEND=1024     initial suspension length S0 (rounds)
  INKLING_ADAPTIVE_SPEC_SUSPEND_MAX=16384  suspension length cap
  INKLING_ADAPTIVE_SPEC_BACKOFF=2.0      suspension backoff multiplier
  INKLING_ADAPTIVE_SPEC_PROBE=4          probe length P (rounds); 0 = one-way
"""

import os
from collections import deque

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ACTIVE = 0
_SUSPENDED = 1
_PROBING = 2


class _ReqPolicyState:
    __slots__ = (
        "mode",
        "ema",
        "rounds_observed",
        "probe_window",
        "rounds_suspended",
        "suspend_len",
        "probe_left",
        "last_seen_step",
    )

    def __init__(self, suspend_len: int):
        self.mode = _ACTIVE
        self.ema: float | None = None
        self.rounds_observed = 0
        self.probe_window: deque[int] = deque()
        self.rounds_suspended = 0
        self.suspend_len = suspend_len
        self.probe_left = 0
        self.last_seen_step = 0


class AdaptiveSpecPolicy:
    """Per-request rolling-acceptance suspend/resume policy.

    Host-side only; no GPU state beyond one pinned staging buffer for the
    async acceptance-count copy.
    """

    def __init__(
        self,
        num_spec_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        use_cuda_staging: bool = True,
    ):
        self.num_spec_tokens = num_spec_tokens
        self.max_num_reqs = max_num_reqs
        self.device = device
        # use_cuda_staging=False (host-only unit tests): synchronous copies,
        # no pinned memory / CUDA events.
        self.use_cuda_staging = use_cuda_staging

        def _env_float(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        def _env_int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        self.min_rounds = max(1, _env_int("INKLING_ADAPTIVE_SPEC_MIN_ROUNDS", 10))
        self.ema_halflife = max(
            1.0, _env_float("INKLING_ADAPTIVE_SPEC_EMA_HALFLIFE", 32.0)
        )
        # Per-round decay factor for the acceptance EMA.
        self.ema_decay = 0.5 ** (1.0 / self.ema_halflife)

        # Break-even model (task #139 ns sweep): a draft+verify round costs
        # roughly verify (~ a plain decode step p, weight-bound) plus ns
        # sequential drafter passes of ~RATIO*p each, so drafting pays off
        # when accepted-per-round A > round/p ~= 1 + RATIO*ns. Measured at
        # ns=2 @32K (2026-07-19): p ~24.5ms, round ~48ms -> RATIO ~0.48; the
        # default threshold adds a MARGIN so marginal content suspends
        # (ns=2 default: 1.05 * (1 + 0.96) ~= 2.06, matching the gated 2.05).
        # INKLING_ADAPTIVE_SPEC_THRESHOLD overrides absolutely.
        self.draft_cost_ratio = _env_float(
            "INKLING_ADAPTIVE_SPEC_DRAFT_COST_RATIO", 0.48
        )
        self.threshold_margin = _env_float("INKLING_ADAPTIVE_SPEC_MARGIN", 1.05)
        default_threshold = self.threshold_margin * (
            1.0 + self.draft_cost_ratio * num_spec_tokens
        )
        self.threshold = _env_float(
            "INKLING_ADAPTIVE_SPEC_THRESHOLD", default_threshold
        )
        self.resume_threshold = _env_float(
            "INKLING_ADAPTIVE_SPEC_RESUME_THRESHOLD", self.threshold * 1.1
        )
        self.suspend_len0 = max(1, _env_int("INKLING_ADAPTIVE_SPEC_SUSPEND", 1024))
        self.suspend_len_max = max(
            self.suspend_len0, _env_int("INKLING_ADAPTIVE_SPEC_SUSPEND_MAX", 16384)
        )
        self.backoff = max(1.0, _env_float("INKLING_ADAPTIVE_SPEC_BACKOFF", 2.0))
        self.probe_len = max(0, _env_int("INKLING_ADAPTIVE_SPEC_PROBE", 4))

        # Pinned staging buffer + event for the async num_sampled copy.
        self._staged_num_sampled = torch.empty(
            max_num_reqs, dtype=torch.int32, pin_memory=use_cuda_staging
        )
        self._staged_event = torch.cuda.Event() if use_cuda_staging else None
        # Metadata snapshot matching the staged copy (None = nothing staged).
        self._staged_req_ids: list[str] | None = None
        self._staged_drafted: np.ndarray | None = None

        self._states: dict[str, _ReqPolicyState] = {}
        self._step = 0
        self._transitions_logged = 0

        logger.info(
            "Adaptive speculative decoding enabled: min_rounds=%d "
            "ema_halflife=%.0f threshold=%.2f resume_threshold=%.2f "
            "suspend=%d suspend_max=%d backoff=%.1f probe=%d "
            "(num_spec_tokens=%d)",
            self.min_rounds,
            self.ema_halflife,
            self.threshold,
            self.resume_threshold,
            self.suspend_len0,
            self.suspend_len_max,
            self.backoff,
            self.probe_len,
            num_spec_tokens,
        )

    # ------------------------------------------------------------------
    # Observation (called once per step, after sampling).
    # ------------------------------------------------------------------

    def observe(self, input_batch, num_sampled: torch.Tensor) -> None:
        """Stage this round's per-request acceptance for the next decide().

        ``num_sampled[i]`` = accepted draft tokens + 1 (bonus) for spec
        rounds; 1 for plain decode; 0/1 for prefill chunks. Non-blocking
        D2H copy on the current stream; consumed next round after an
        event sync (by then long complete).
        """
        drafted = input_batch.num_draft_tokens_per_req
        if input_batch.num_draft_tokens == 0 or drafted is None:
            # No drafts were verified this round; nothing to learn.
            self._staged_req_ids = None
            self._staged_drafted = None
            return
        num_reqs = input_batch.num_reqs
        self._staged_num_sampled[:num_reqs].copy_(
            num_sampled[:num_reqs], non_blocking=self.use_cuda_staging
        )
        if self._staged_event is not None:
            self._staged_event.record()
        self._staged_req_ids = list(input_batch.req_ids)
        self._staged_drafted = drafted.copy()

    def _consume_staged(self) -> None:
        req_ids = self._staged_req_ids
        if req_ids is None:
            return
        drafted = self._staged_drafted
        assert drafted is not None
        if self._staged_event is not None:
            self._staged_event.synchronize()
        counts = self._staged_num_sampled[: len(req_ids)].tolist()
        self._staged_req_ids = None
        self._staged_drafted = None

        for i, req_id in enumerate(req_ids):
            if drafted[i] <= 0 or counts[i] < 1:
                continue
            state = self._states.get(req_id)
            if state is None:
                continue
            accepted = int(counts[i])
            if state.mode == _PROBING:
                state.probe_window.append(accepted)
                state.probe_left -= 1
                if state.probe_left <= 0:
                    probe_mean = sum(state.probe_window) / max(
                        1, len(state.probe_window)
                    )
                    if probe_mean >= self.resume_threshold:
                        state.mode = _ACTIVE
                        state.ema = probe_mean
                        state.rounds_observed = len(state.probe_window)
                        state.suspend_len = self.suspend_len0
                        self._log_transition(
                            req_id, "probe PASS -> resume drafting", probe_mean
                        )
                    else:
                        state.mode = _SUSPENDED
                        state.rounds_suspended = 0
                        state.suspend_len = min(
                            int(state.suspend_len * self.backoff),
                            self.suspend_len_max,
                        )
                        self._log_transition(
                            req_id,
                            f"probe FAIL -> suspend {state.suspend_len} rounds",
                            probe_mean,
                        )
                    state.probe_window.clear()
            else:
                if state.ema is None:
                    state.ema = float(accepted)
                else:
                    state.ema = (
                        state.ema * self.ema_decay
                        + float(accepted) * (1.0 - self.ema_decay)
                    )
                state.rounds_observed += 1
                if (
                    state.mode == _ACTIVE
                    and state.rounds_observed >= self.min_rounds
                    and state.ema < self.threshold
                ):
                    ema = state.ema
                    state.mode = _SUSPENDED
                    state.rounds_suspended = 0
                    state.suspend_len = self.suspend_len0
                    self._log_transition(
                        req_id,
                        f"suspend drafting {state.suspend_len} rounds",
                        ema,
                    )

    def _log_transition(self, req_id: str, event: str, mean: float) -> None:
        # INFO for the first transitions (gate visibility), DEBUG after.
        self._transitions_logged += 1
        if self._transitions_logged <= 50:
            logger.info(
                "Adaptive spec: req %s %s (mean accepted/round %.2f)",
                req_id,
                event,
                mean,
            )
        else:
            logger.debug(
                "Adaptive spec: req %s %s (mean accepted/round %.2f)",
                req_id,
                event,
                mean,
            )

    # ------------------------------------------------------------------
    # Decision (called once per step, right before propose).
    # ------------------------------------------------------------------

    def decide(self, input_batch) -> tuple[bool, np.ndarray | None]:
        """Return (skip_propose, valid_counts).

        skip_propose: skip the drafter forward entirely (all requests
            suspended AND the batch is pure 1-token decode -- so the
            drafter's per-round KV re-prefill is not needed for any
            request that will draft again this round).
        valid_counts: per-request number of draft tokens to expose to
            the scheduler (int32 [num_reqs]), or None for no masking.
        """
        self._step += 1
        self._consume_staged()

        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        is_prefilling = input_batch.is_prefilling_np
        num_scheduled = input_batch.num_scheduled_tokens

        any_masked = False
        all_suspended_decode = True
        valid_counts = np.full(num_reqs, self.num_spec_tokens, dtype=np.int32)

        for i in range(num_reqs):
            state = self._states.get(req_ids[i])
            if state is None:
                state = _ReqPolicyState(self.suspend_len0)
                self._states[req_ids[i]] = state
            state.last_seen_step = self._step

            if is_prefilling[i]:
                # Never suspend the drafter during (re-)prefill: propose
                # syncs the drafter KV over the prompt (and heals holes
                # after preemption). Scheduler ignores drafts for
                # non-final prefill chunks anyway.
                all_suspended_decode = False
                continue

            if state.mode == _SUSPENDED:
                state.rounds_suspended += 1
                if (
                    self.probe_len > 0
                    and state.rounds_suspended >= state.suspend_len
                ):
                    state.mode = _PROBING
                    state.probe_left = self.probe_len
                    state.probe_window.clear()
                else:
                    valid_counts[i] = 0
                    any_masked = True
                    continue

            # ACTIVE or PROBING: this request drafts.
            all_suspended_decode = False

        skip_propose = (
            num_reqs > 0
            and all_suspended_decode
            and bool((num_scheduled[:num_reqs] == 1).all())
        )

        self._maybe_prune()
        if not any_masked:
            return skip_propose, None
        return skip_propose, valid_counts

    def _maybe_prune(self) -> None:
        # Request ids are unique per request; drop states not seen for a
        # long time (finished/aborted requests).
        if len(self._states) <= 4 * self.max_num_reqs:
            return
        horizon = self._step - 8 * self.suspend_len_max
        stale = [
            req_id
            for req_id, state in self._states.items()
            if state.last_seen_step < horizon
        ]
        for req_id in stale:
            del self._states[req_id]
        if len(self._states) > 64 * self.max_num_reqs:
            # Hard cap: evict oldest.
            by_age = sorted(
                self._states.items(), key=lambda kv: kv[1].last_seen_step
            )
            for req_id, _ in by_age[: len(by_age) // 2]:
                del self._states[req_id]


def maybe_init_adaptive_spec_policy(
    vllm_config, device: torch.device
) -> AdaptiveSpecPolicy | None:
    """Create the policy iff INKLING_ADAPTIVE_SPEC=1 and the serving
    configuration is one where skipping the drafter collective is safe
    (single DP rank, no PP draft broadcast, sync scheduling, and an
    Eagle/MTP-style speculator that runs a model forward)."""
    if os.getenv("INKLING_ADAPTIVE_SPEC", "0") != "1":
        return None
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        logger.warning(
            "INKLING_ADAPTIVE_SPEC=1 ignored: no Eagle/MTP-style "
            "speculative config."
        )
        return None
    parallel = vllm_config.parallel_config
    if parallel.data_parallel_size > 1 or parallel.pipeline_parallel_size > 1:
        logger.warning(
            "INKLING_ADAPTIVE_SPEC=1 ignored: DP/PP > 1 (skipping the "
            "drafter forward is only coordinated across TP ranks)."
        )
        return None
    if getattr(spec_config, "draft_tp_over_pp", False):
        logger.warning(
            "INKLING_ADAPTIVE_SPEC=1 ignored: draft-TP-over-PP path."
        )
        return None
    if vllm_config.scheduler_config.async_scheduling:
        logger.warning(
            "INKLING_ADAPTIVE_SPEC=1 ignored: async scheduling not "
            "supported (draft-token counts must round-trip through the "
            "scheduler each step)."
        )
        return None
    return AdaptiveSpecPolicy(
        num_spec_tokens=spec_config.num_speculative_tokens,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        device=device,
    )
