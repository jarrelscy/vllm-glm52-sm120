# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm/v1/attention/ops/raw_kv_reuse.py"
spec = importlib.util.spec_from_file_location("raw_kv_reuse_test", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stash_consumed = module.stash_consumed


@pytest.mark.parametrize("peer_bytes", [32, 64, 96, 128, 192])
def test_only_consumed_bytes_are_overwritten(peer_bytes):
    gathered = torch.arange(4 * peer_bytes).remainder(256).to(torch.uint8)
    gathered = gathered.view(4, peer_bytes)
    original = gathered.clone()
    parts = []
    expected = []
    stashed = 0
    for peer in range(4):
        out = (torch.arange(64) + peer * 100).to(torch.bfloat16).view(8, 8)
        expected.append(out.clone())
        parts.append(out)
        del out
        stashed = stash_consumed(gathered, parts, peer + 1, stashed)
        assert torch.equal(gathered[peer + 1 :], original[peer + 1 :])
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(parts, expected)
        )
    assert stashed == min(4, 4 * peer_bytes // 128)


def test_rejects_empty_parts_and_wrong_dtype():
    gathered = torch.zeros((4, 128), dtype=torch.uint8)
    with pytest.raises(AssertionError):
        stash_consumed(gathered, [], 1, 0)
    with pytest.raises(AssertionError):
        stash_consumed(gathered, [torch.zeros(8, dtype=torch.float32)], 1, 0)


def test_repeated_call_does_not_recopy_stashed_outputs():
    gathered = torch.zeros((4, 256), dtype=torch.uint8)
    parts = [torch.ones(64, dtype=torch.bfloat16)]
    stashed = stash_consumed(gathered, parts, 1, 0)
    assert stashed == 1
    pointer = parts[0].data_ptr()
    parts[0].add_(1)
    assert stash_consumed(gathered, parts, 1, stashed) == 1
    assert parts[0].data_ptr() == pointer
    assert torch.equal(parts[0], torch.full((64,), 2, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "length,called", [(131072, False), (262144, False), (262145, True), (524288, True)]
)
def test_production_stash_only_after_qualified_boundary(length, called):
    import ast

    source = SOURCE.with_name("raw_kv_gather.py")
    tree = ast.parse(source.read_text())
    block = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "length > 262144"
    )
    calls = []

    def stash(*args):
        calls.append(args)
        return 1

    scope = dict(
        length=length,
        output=object(),
        allkv=object(),
        parts=[],
        peer=0,
        stashed=0,
        stash_consumed=stash,
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
    assert bool(calls) == called
    assert ("output" not in scope) == called
