# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure CPU proof of bounded descriptors and unchanged warp-local token pairs."""

import importlib.util
import unittest
from pathlib import Path

import pytest
import torch


def descriptors(ids):
    result = []
    for start in range(0, len(ids), 32):
        window = ids[start : start + 32]
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(window):
            end = i + 1
            while end < len(window) and window[end] == window[i]:
                end += 1
            groups.extend((start + j, min(16, end - j)) for j in range(i, end, 16))
            i = end
        assert len(groups) <= 3
        result.extend(groups)
    return result


def original_pairs(ids):
    pairs: list[tuple[int, ...]] = []
    for start in range(0, len(ids), 32):
        for expert in sorted(set(ids[start : start + 32])):
            slots = [
                i for i in range(start, min(start + 32, len(ids))) if ids[i] == expert
            ]
            pairs.extend(tuple(slots[j : j + 2]) for j in range(0, len(slots), 2))
    return sorted(pairs)


class RouteProof(unittest.TestCase):
    def test_all_offsets_and_odd_runs(self):
        for a in (32, 33, 47, 63, 64, 95):
            ids = sum(
                (
                    [expert] * length
                    for expert, length in enumerate([a, 35, 64, 49, 97] * 8)
                ),
                [],
            )
            for offset in range(64):
                for size in (1, 7, 31, 32, 33, 63, 127, 1024):
                    chunk = ids[offset : offset + size]
                    groups = descriptors(chunk)
                    covered = [
                        i
                        for start, count in groups
                        for i in range(start, start + count)
                    ]
                    self.assertEqual(covered, list(range(len(chunk))))
                    pairs: list[tuple[int, ...]] = []
                    for start, count in groups:
                        self.assertEqual(len(set(chunk[start : start + count])), 1)
                        pairs.extend(
                            tuple(range(j, min(j + 2, start + count)))
                            for j in range(start, start + count, 2)
                        )
                    self.assertEqual(sorted(pairs), original_pairs(chunk))

    def test_tile_coverage(self):
        for n in (1024, 6144):
            old = sorted(
                block * 4 + warp for block in range(n // 64) for warp in range(4)
            )
            self.assertEqual(old, list(range(n // 16)))
            for row_tiles in (2, 4):
                for count in range(1, 17):
                    outputs = [
                        (block * row_tiles + warp % row_tiles, position)
                        for block in range(n // (16 * row_tiles))
                        for warp in range(8 * row_tiles)
                        for position in range(
                            2 * (warp // row_tiles),
                            min(2 * (warp // row_tiles) + 2, count),
                        )
                    ]
                    self.assertEqual(
                        sorted(outputs),
                        [
                            (tile, slot)
                            for tile in range(n // 16)
                            for slot in range(count)
                        ],
                    )


if __name__ == "__main__":
    unittest.main()


def test_eight_pairs_per_warp_cover_sixteen_positions():
    for count in range(1, 17):
        positions = [
            (warp, token)
            for warp in range(4)
            for pair in range(8)
            for token in range(pair * 2, min(pair * 2 + 2, count))
        ]
        assert sorted(positions) == [
            (tile, token) for tile in range(4) for token in range(count)
        ]


@pytest.mark.parametrize("mode", [0, 2])
def test_register8_runtime_descriptor_abi(monkeypatch, mode):
    from types import SimpleNamespace

    path = (
        Path(__file__).resolve().parents[2]
        / "vllm/model_executor/layers/quantization/nvfp4_arvq_wide_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("register8_runtime_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def record(name):
        def invoke(*args):
            calls.append((name, args))
            return 0

        return invoke

    library = SimpleNamespace(
        hybrid_pack_register_pairs=record("pack_register_pairs"),
        hybrid_pack=record("pack"),
        wide_launch_register=record("register"),
    )
    monkeypatch.setattr(module, "library", lambda: library)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    fn = module.WideHot(mode)
    x = torch.zeros((33, 128), dtype=torch.float16)
    hot = torch.zeros(33, dtype=torch.int32)
    cold = torch.full_like(hot, -1)
    tensors = [torch.zeros(1)] * 6
    fn(x, cold, hot, tensors, 1.0, 64, 8, 2)
    groups = fn.groups
    fn(x, cold, hot, tensors, 1.0, 64, 2, 1)
    assert groups.shape == (2, 7) and fn.groups is groups
    assert [n for n, _ in calls] == [
        "pack_register_pairs",
        "register",
        "pack",
        "register",
    ]
    assert calls[1][1][10:16] == (64, 128, 33, 8, 2, mode)
    assert calls[3][1][10:16] == (64, 128, 33, 2, 1, mode)
