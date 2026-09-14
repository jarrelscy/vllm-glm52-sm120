# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add allocation bounds to the host's indexer without editing its source."""

import argparse
from pathlib import Path


def prepare(source: str) -> str:
    additions = (
        (
            "from vllm.v1.worker.workspace import current_workspace_manager",
            "from vllm.v1.attention.backends.mla.workspace_limits import "
            "indexer_allocation_tokens\n"
            "from vllm.v1.worker.workspace import current_workspace_manager",
        ),
        (
            "        self.max_total_seq_len = max_total_seq_len",
            "        config = get_current_vllm_config()\n"
            "        speculative_tokens = (\n"
            "            (config.speculative_config.num_speculative_tokens or 0)\n"
            "            if config.speculative_config is not None else 0\n"
            "        )\n"
            "        self.max_total_seq_len = indexer_allocation_tokens(\n"
            "            max_total_seq_len, max_model_len,\n"
            "            config.scheduler_config.max_num_seqs, speculative_tokens\n"
            "        )",
        ),
        (
            "            assert chunk.local_cu_seq_lens is not None",
            "            assert chunk.local_cu_seq_lens is not None\n"
            "            assert chunk.max_local_total_seq_lens <= total_seq_lens, (\n"
            '                "Indexer local gather exceeds allocation bound"\n'
            "            )",
        ),
    )
    if "self.max_total_seq_len = indexer_allocation_tokens(" in source:
        if any(source.count(replacement) != 1 for _, replacement in additions):
            raise ValueError("Host indexer contains a partial or altered overlay")
        if additions[1][0] in source:
            raise ValueError("Host indexer contains both old and bounded allocations")
        compile(source, "indexer_overlay", "exec")
        return source
    for original, replacement in additions:
        if source.count(original) != 1:
            raise ValueError(f"Host indexer changed; expected one match: {original}")
        source = source.replace(original, replacement, 1)
    compile(source, "indexer_overlay", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = prepare(args.source.read_text())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(".tmp")
    temporary.write_text(result)
    temporary.replace(args.destination)


if __name__ == "__main__":
    main()
