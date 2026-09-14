# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from prepare_compact_workspace_overlay import prepare

SOURCE = """from vllm.v1.worker.workspace import current_workspace_manager
from vllm.config import get_current_vllm_config

QUERY_SPLIT = 37  # Keep the host's independently tuned split unchanged.

class Indexer:
    def __init__(self, max_total_seq_len, max_model_len):
        self.max_total_seq_len = max_total_seq_len

    def gather(self, chunks, total_seq_lens):
        for chunk in chunks:
            assert chunk.local_cu_seq_lens is not None
            use_split(QUERY_SPLIT, chunk)
"""


def test_preserves_host_query_split_and_is_idempotent():
    result = prepare(SOURCE)
    assert (
        "QUERY_SPLIT = 37  # Keep the host's independently tuned split unchanged."
        in result
    )
    assert "            use_split(QUERY_SPLIT, chunk)" in result
    assert "self.max_total_seq_len = indexer_allocation_tokens(" in result
    assert "assert chunk.max_local_total_seq_lens <= total_seq_lens" in result
    assert prepare(result) == result
    compile(result, "overlay", "exec")


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_rejects_unexpected_anchor(change):
    anchor = "            assert chunk.local_cu_seq_lens is not None"
    source = SOURCE.replace(
        anchor, "            pass" if change == "missing" else anchor + "\n" + anchor
    )
    with pytest.raises(ValueError, match="expected one match"):
        prepare(source)


def test_rejects_partial_preexisting_overlay():
    source = prepare(SOURCE).replace(
        "from vllm.v1.attention.backends.mla.workspace_limits "
        "import indexer_allocation_tokens\n",
        "",
    )
    with pytest.raises(ValueError, match="partial or altered"):
        prepare(source)


def test_revalidates_syntax_on_second_run():
    with pytest.raises(SyntaxError):
        prepare(prepare(SOURCE) + "\nthis is invalid syntax !\n")
