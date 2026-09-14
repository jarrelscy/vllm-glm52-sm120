# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only eligibility for uniform decode CUDA graphs."""

from collections.abc import Iterable, Mapping

import numpy as np


def decode_only_uniform_token_count(
    uniform_token_count: int | None,
    scheduled_req_ids: Iterable[str],
    req_id_to_index: Mapping[str, int],
    num_computed_prefill_tokens: np.ndarray,
    prefill_lens: np.ndarray,
    *,
    dummy_run: bool = False,
) -> int | None:
    """Exclude prefilling requests from graphs captured for uniform decode.

    Use request-state indices, which can differ from batch order after request
    admission or slot reuse. Dummy runs retain their capture descriptors.
    """
    if dummy_run or uniform_token_count is None:
        return uniform_token_count
    for req_id in scheduled_req_ids:
        index = req_id_to_index[req_id]
        if num_computed_prefill_tokens[index] < prefill_lens[index]:
            return None
    return uniform_token_count
