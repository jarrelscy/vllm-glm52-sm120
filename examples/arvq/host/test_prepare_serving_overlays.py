# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest
from prepare_compact_workspace_overlay import prepare as prepare_compact
from prepare_serving_overlays import (
    CALL,
    ENV_ANCHOR,
    ENV_ENTRY,
    indexer_replacements,
    prepare_envs,
    prepare_indexer,
)
from test_prepare_compact_workspace_overlay import SOURCE as COMPACT_SOURCE

REPOSITORY = (
    Path(__file__).parents[3] / "vllm/model_executor/layers/sparse_attn_indexer.py"
).read_text()


def host_source():
    tree = ast.parse(REPOSITORY)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_dcp_topk_global"
    )
    lines = REPOSITORY.splitlines(keepends=True)
    body = "".join(lines[function.lineno - 1 : function.end_lineno])
    for original, replacement in indexer_replacements(REPOSITORY)[:2]:
        assert body.count(replacement) == 1
        body = body.replace(replacement, original, 1)
    return (
        COMPACT_SOURCE
        + "\n"
        + body
        + "\n\ndef foreign_query_split(x):\n    return x[::4]  # Preserve me.\n"
        + "\ndef prefill_call():\n    result = unknown(\n"
        + CALL
        + "    )\n    return result\n"
    )


def test_preserves_all_unrelated_source_and_is_idempotent():
    original = host_source()
    result = prepare_indexer(original, REPOSITORY)
    assert prepare_indexer(result, REPOSITORY) == result
    restored = result
    for before, after in indexer_replacements(REPOSITORY):
        restored = restored.replace(after, before, 1)
    assert restored == prepare_compact(original)
    assert "return x[::4]  # Preserve me." in result
    assert "QUERY_SPLIT = 37" in result
    compile(result, "indexer_overlay", "exec")


@pytest.mark.parametrize("partial", ["signature", "body", "call", "altered"])
def test_rejects_partial_or_altered_patch(partial):
    source = prepare_indexer(host_source(), REPOSITORY)
    replacements = indexer_replacements(REPOSITORY)
    if partial == "altered":
        source = source.replace(
            "bytepack_context=attn_metadata_narrowed.max_seq_len", "bytepack_context=42"
        )
    else:
        original, replacement = replacements[
            ("signature", "body", "call").index(partial)
        ]
        source = source.replace(replacement, original, 1)
    with pytest.raises(ValueError, match="Partial or altered"):
        prepare_indexer(source, REPOSITORY)


@pytest.mark.parametrize("duplicate", [False, True])
def test_unknown_anchor_rejected(duplicate):
    source = host_source()
    if duplicate:
        source += "\ndef duplicate():\n    unknown(\n" + CALL + "    )\n"
    else:
        source = source.replace(CALL, "")
    with pytest.raises(ValueError, match="Expected one host anchor"):
        prepare_indexer(source, REPOSITORY)


def test_env_default_off_preserves_other_flags_and_is_idempotent(monkeypatch):
    monkeypatch.delenv("VLLM_GLM_RAW_KV_GATHER", raising=False)
    source = (
        "import os\nflags = {\n"
        + ENV_ANCHOR
        + """
        int(os.getenv("VLLM_GLM_DCP_AG_RAW_TOPK", "0"))
    ),
    "FOREIGN_QUERY_SPLIT": lambda: 37,
}
"""
    )
    result = prepare_envs(source)
    assert result.replace(ENV_ENTRY, "", 1) == source
    assert prepare_envs(result) == result
    scope = {}
    exec(compile(result, "envs", "exec"), scope)
    assert scope["flags"]["FOREIGN_QUERY_SPLIT"]() == 37
    assert scope["flags"]["VLLM_GLM_RAW_KV_GATHER"]() is False


@pytest.mark.parametrize(
    "source",
    [
        "flags = {}\n",
        "VLLM_GLM_RAW_KV_GATHER = True\n",
        "flags = {\n" + ENV_ENTRY.replace('"0"', '"1"') + ENV_ANCHOR + "False),\n}\n",
    ],
)
def test_env_rejects_unknown_or_partial_source(source):
    with pytest.raises(ValueError):
        prepare_envs(source)
