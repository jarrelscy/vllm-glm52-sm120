# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only unit tests for the adaptive speculation policy (laneC).

No GPU required: the policy is exercised in CPU staging mode and the
DraftTokensHandler slicing is tested on the sync (placeholder) path.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.adaptive import AdaptiveSpecPolicy

NS = 2  # num_speculative_tokens (production ns=2)


def make_batch(
    req_ids,
    *,
    drafted=None,
    prefilling=None,
    num_scheduled=None,
):
    n = len(req_ids)
    drafted = np.array(drafted if drafted is not None else [0] * n, dtype=np.int32)
    if num_scheduled is None:
        num_scheduled = drafted + 1
    return SimpleNamespace(
        num_reqs=n,
        req_ids=list(req_ids),
        is_prefilling_np=np.array(
            prefilling if prefilling is not None else [False] * n, dtype=bool
        ),
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int32),
        num_draft_tokens=int(drafted.sum()),
        num_draft_tokens_per_req=drafted,
    )


def make_policy(monkeypatch, **env):
    defaults = {
        "INKLING_ADAPTIVE_SPEC_WINDOW": "4",
        "INKLING_ADAPTIVE_SPEC_THRESHOLD": "2.1",
        "INKLING_ADAPTIVE_SPEC_RESUME_THRESHOLD": "2.25",
        "INKLING_ADAPTIVE_SPEC_SUSPEND": "6",
        "INKLING_ADAPTIVE_SPEC_SUSPEND_MAX": "24",
        "INKLING_ADAPTIVE_SPEC_BACKOFF": "2.0",
        "INKLING_ADAPTIVE_SPEC_PROBE": "2",
    }
    defaults.update({k: str(v) for k, v in env.items()})
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)
    return AdaptiveSpecPolicy(
        num_spec_tokens=NS,
        max_num_reqs=2,
        device=torch.device("cpu"),
        use_cuda_staging=False,
    )


def spec_round(policy, req_ids, accepted, *, expect_skip=False, expect_mask=None):
    """One decide+observe cycle for a drafting round (qlen = 1 + ns)."""
    batch = make_batch(req_ids, drafted=[NS] * len(req_ids))
    skip, counts = policy.decide(batch)
    assert skip == expect_skip
    if expect_mask is None:
        assert counts is None
    else:
        assert counts is not None and counts.tolist() == expect_mask
    policy.observe(batch, torch.tensor(accepted, dtype=torch.int32))
    return skip, counts


def plain_round(policy, req_ids):
    """One decide+observe cycle for a pure 1-token decode round."""
    batch = make_batch(req_ids, drafted=[0] * len(req_ids))
    skip, counts = policy.decide(batch)
    policy.observe(batch, torch.tensor([1] * len(req_ids), dtype=torch.int32))
    return skip, counts


def test_new_request_drafts_by_default(monkeypatch):
    policy = make_policy(monkeypatch)
    skip, counts = policy.decide(make_batch(["a"], drafted=[NS]))
    assert not skip and counts is None


def test_low_acceptance_suspends_then_skips(monkeypatch):
    policy = make_policy(monkeypatch)
    # Window of 4 rounds at acceptance 1 (all drafts rejected): mean 1 < 2.1.
    for _ in range(4):
        spec_round(policy, ["a"], [1])
    # Next round: suspended. Nothing scheduled drafts anymore, pure decode
    # batch -> the whole propose is skipped.
    skip, counts = plain_round(policy, ["a"])
    assert skip
    # Stays skipped until the probe fires (SUSPEND=6 rounds).
    for _ in range(4):
        skip, _ = plain_round(policy, ["a"])
        assert skip
    # Round 6 of suspension flips to PROBING: drafts again, no skip.
    skip, counts = plain_round(policy, ["a"])
    assert not skip and counts is None


def test_high_acceptance_stays_active(monkeypatch):
    policy = make_policy(monkeypatch)
    for _ in range(20):
        skip, counts = spec_round(policy, ["a"], [3])
        assert not skip and counts is None


def test_marginal_acceptance_code_like_stays_active(monkeypatch):
    policy = make_policy(monkeypatch)
    # Code-like acceptance ~2.6 out of 3 stays above threshold 2.1.
    pattern = [3, 3, 2, 3, 2, 3, 3, 2]
    for accepted in pattern * 3:
        skip, counts = spec_round(policy, ["a"], [accepted])
        assert not skip and counts is None


def drive_spec(policy, accepted):
    """decide+observe for a drafting round; no expectations."""
    batch = make_batch(["a"], drafted=[NS])
    skip, counts = policy.decide(batch)
    policy.observe(batch, torch.tensor([accepted], dtype=torch.int32))
    return skip, counts


def drive_until_suspended(policy, max_rounds=20):
    """Feed hostile rounds until the policy suspends request 'a'."""
    for _ in range(max_rounds):
        drive_spec(policy, 1)
        if policy._states["a"].mode == 1:  # SUSPENDED
            return
    raise AssertionError("policy never suspended")


def drive_until_probing(policy, max_rounds=50):
    for _ in range(max_rounds):
        skip, _ = plain_round(policy, ["a"])
        if policy._states["a"].mode == 2:  # PROBING
            assert not skip
            return
        assert skip
    raise AssertionError("policy never probed")


def settle(policy, accepted):
    """Run probe rounds with the given acceptance until the probe resolves."""
    for _ in range(20):
        drive_spec(policy, accepted)
        # One extra decide consumes the lagging observation.
        if policy._states["a"].mode != 2:
            return
        if policy._states["a"].probe_left <= 0:
            policy.decide(make_batch(["a"], drafted=[0]))
            return
    raise AssertionError("probe never resolved")


def test_probe_pass_resumes_and_probe_fail_backs_off(monkeypatch):
    policy = make_policy(monkeypatch)

    drive_until_suspended(policy)
    drive_until_probing(policy)
    # Probe rounds with high acceptance -> ACTIVE again.
    settle(policy, 3)
    state = policy._states["a"]
    assert state.mode == 0  # ACTIVE
    assert state.suspend_len == 6  # reset on successful probe

    # Suspend again, fail the probe -> backoff doubles suspend_len.
    drive_until_suspended(policy)
    drive_until_probing(policy)
    settle(policy, 1)
    state = policy._states["a"]
    assert state.mode == 1  # SUSPENDED
    assert state.suspend_len == 12  # 6 * backoff 2.0


def test_probe_zero_is_one_way(monkeypatch):
    policy = make_policy(monkeypatch, INKLING_ADAPTIVE_SPEC_PROBE="0")
    for _ in range(4):
        spec_round(policy, ["a"], [1])
    for _ in range(50):
        assert plain_round(policy, ["a"])[0]


def test_prefill_never_suspended_or_skipped(monkeypatch):
    policy = make_policy(monkeypatch)
    for _ in range(4):
        spec_round(policy, ["a"], [1])
    assert plain_round(policy, ["a"])[0]  # suspended
    # A (re-)prefill chunk for the same request must run propose (drafter
    # KV sync) and must not be masked.
    batch = make_batch(["a"], drafted=[0], prefilling=[True], num_scheduled=[512])
    skip, counts = policy.decide(batch)
    assert not skip and counts is None


def test_mixed_batch_masks_only_suspended(monkeypatch):
    policy = make_policy(monkeypatch)
    # Suspend "a" while "b" stays hot.
    for _ in range(4):
        spec_round(policy, ["a", "b"], [1, 3])
    batch = make_batch(["a", "b"], drafted=[0, NS])
    skip, counts = policy.decide(batch)
    assert not skip  # "b" still drafts -> propose must run
    assert counts is not None and counts.tolist() == [0, NS]


def test_handler_valid_counts_slicing():
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    handler = DraftTokensHandler.__new__(DraftTokensHandler)
    handler.req_ids = ["a", "b"]
    handler.draft_tokens_np = None
    handler.num_draft_tokens = NS
    handler.valid_counts = np.array([0, NS], dtype=np.int32)
    out = handler.get_draft_tokens()
    assert out.draft_token_ids == [[], [-1] * NS]

    handler.valid_counts = None
    out = handler.get_draft_tokens()
    assert out.draft_token_ids == [[-1] * NS, [-1] * NS]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
