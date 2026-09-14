# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scratch allocation bounds; attention scheduling is unchanged."""

import os


def compact_workspace_enabled() -> bool:
    return os.environ.get("VLLM_SM120_COMPACT_WORKSPACE", "0") == "1"


def needs_bf16_prefill_workspace(
    cache_dtype: str, num_heads: int, minimum_prefill_heads: int
) -> bool:
    return cache_dtype == "fp8_ds_mla" and (
        not compact_workspace_enabled() or num_heads >= minimum_prefill_heads
    )


def indexer_allocation_tokens(
    original_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    speculative_tokens: int = 0,
) -> int:
    if not compact_workspace_enabled():
        return original_tokens
    # Prefill metadata contains at most max_num_seqs requests. Each request
    # has at most max_model_len context tokens. Keep a conservative extra
    # verification token plus the complete draft lookahead per request.
    # DCP local lengths cannot exceed these global lengths. This only limits
    # storage, not the independent metadata chunking/logits budgets.
    feasible_tokens = max_num_seqs * (max_model_len + speculative_tokens + 1)
    return min(original_tokens, feasible_tokens)
