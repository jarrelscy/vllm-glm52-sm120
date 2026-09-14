# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for allocation-only bounds, without importing GPU backends."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/attention/backends/mla/workspace_limits.py"
)
spec = importlib.util.spec_from_file_location("workspace_limits_test", SOURCE)
assert spec is not None and spec.loader is not None
limits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limits)


@pytest.mark.parametrize("heads", [1, 16, 31, 32, 64, 128])
def test_default_preserves_existing_reservations(monkeypatch, heads):
    monkeypatch.delenv("VLLM_SM120_COMPACT_WORKSPACE", raising=False)
    assert limits.needs_bf16_prefill_workspace("fp8_ds_mla", heads, 32)
    assert limits.indexer_allocation_tokens(40 * 1024, 1024, 8, 3) == 40 * 1024


@pytest.mark.parametrize("heads", [1, 16, 31, 32, 64, 128])
def test_only_unused_prefill_branch_is_removed(monkeypatch, heads):
    monkeypatch.setenv("VLLM_SM120_COMPACT_WORKSPACE", "1")
    assert limits.needs_bf16_prefill_workspace("fp8_ds_mla", heads, 32) == (heads >= 32)
    assert not limits.needs_bf16_prefill_workspace("bfloat16", heads, 32)


@pytest.mark.parametrize("num_seqs", [1, 4, 8, 40, 64])
@pytest.mark.parametrize("lookahead", [0, 3, 5])
def test_every_legal_partial_chunk_fits(monkeypatch, num_seqs, lookahead):
    monkeypatch.setenv("VLLM_SM120_COMPACT_WORKSPACE", "1")
    length = 1024
    original = 40 * length
    allocation = limits.indexer_allocation_tokens(original, length, num_seqs, lookahead)
    assert allocation <= original
    # Enumerate partial request counts and prefix lengths, including empty
    # profiling metadata and all-full contexts. Existing chunks never exceed
    # the original scheduling budget. Allocation must cover every such chunk.
    for requests in range(num_seqs + 1):
        for context in [0, 1, length // 2, length, length + lookahead + 1]:
            chunk_tokens = min(original, requests * context)
            assert chunk_tokens <= allocation
            for dcp in [1, 2, 4, 8]:
                assert (chunk_tokens + dcp - 1) // dcp <= allocation


def test_current_profile_exact_bound(monkeypatch):
    monkeypatch.setenv("VLLM_SM120_COMPACT_WORKSPACE", "1")
    assert limits.indexer_allocation_tokens(40 * 1048576, 1048576, 8, 3) == 8388640
