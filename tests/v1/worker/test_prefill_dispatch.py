# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only request-state eligibility tests; no CUDA/backend import required."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm/v1/worker/gpu/prefill_dispatch.py"
spec = importlib.util.spec_from_file_location("prefill_dispatch_test", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
choose = module.decode_only_uniform_token_count


@pytest.mark.parametrize("uniform", [1, 4, 8])
@pytest.mark.parametrize(
    "computed",
    [[8192, 8192, 8192, 8192], [8191, 8192, 8192, 8192], [8191, 8191, 8191, 8191]],
)
def test_completed_mixed_and_all_new(uniform, computed):
    mapping = {"new": 3, "old": 0, "other": 2, "last": 1}
    scheduled = ["old", "other", "last", "new"]
    result = choose(uniform, scheduled, mapping, np.array(computed), np.full(4, 8192))
    assert result == (uniform if min(computed) >= 8192 else None)


def test_slot_reuse_and_only_scheduled_requests():
    # Slot 0 belongs to a fresh request; only completed slot 5 is scheduled.
    mapping = {"fresh_reused": 0, "old": 5}
    computed = np.array([8191, 0, 0, 0, 0, 8192])
    lengths = np.array([8192, 1, 1, 1, 1, 8192])
    assert choose(4, ["old"], mapping, computed, lengths) == 4
    assert choose(4, ["old", "fresh_reused"], mapping, computed, lengths) is None


def test_resumed_prefill_and_completed_decode():
    # prefill_len may include partial output for a resumed request.
    mapping = {"resumed": 1}
    assert choose(4, mapping, mapping, np.array([0, 8200]), np.array([0, 8201])) is None
    assert choose(4, mapping, mapping, np.array([0, 8201]), np.array([0, 8201])) == 4


def test_dummy_capture_does_not_dereference_request_state():
    assert choose(4, ["dummy"], {}, np.array([]), np.array([]), dummy_run=True) == 4


def test_nonuniform_batch_needs_no_state_lookup():
    assert choose(None, ["not_registered"], {}, np.array([]), np.array([])) is None


def test_empty_dp_rank_retains_existing_descriptor():
    assert choose(None, [], {}, np.array([]), np.array([])) is None
    assert choose(4, [], {}, np.array([]), np.array([])) == 4


def test_eligibility_is_applied_before_dp_sync_and_preparation():
    source = (ROOT / "vllm/v1/worker/gpu/model_runner.py").read_text()
    tree = ast.parse(source)
    calls: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None)
            )
            if name in (
                "decode_only_uniform_token_count",
                "dispatch_cg_and_sync_dp",
                "prepare_inputs",
            ):
                calls.setdefault(name, []).append(node.lineno)
    eligibility = calls["decode_only_uniform_token_count"][0]
    dispatch = next(
        line for line in sorted(calls["dispatch_cg_and_sync_dp"]) if line > eligibility
    )
    prepare = next(
        line for line in sorted(calls["prepare_inputs"]) if line > eligibility
    )
    assert eligibility < dispatch < prepare
