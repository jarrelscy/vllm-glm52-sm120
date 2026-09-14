# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU proof that ineligible calls do not query CUDA capture state."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock


class GuardTests(unittest.TestCase):
    def function(self, capture, flag="1"):
        tree = ast.parse(
            (
                Path(__file__).parents[3] / "vllm/v1/attention/ops/dcp_bytepack.py"
            ).read_text()
        )
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "try_merge"
        )
        original = Mock(return_value="original")
        ns = dict(
            _try_merge_ag=original,
            os=SimpleNamespace(getenv=lambda key, default: flag),
            torch=SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=capture)
            ),
            _owner_shadow_failed=False,
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "guard", "exec"), ns)
        return ns["try_merge"], original

    def test_decode_and_small_prefill_never_call_cuda(self):
        capture = Mock(side_effect=AssertionError("unexpected CUDA call"))
        run, original = self.function(capture)
        for c1, rows in ((False, 4096), (False, 1), (True, 128)):
            result = run(
                None,
                SimpleNamespace(shape=(rows, 2048)),
                None,
                0,
                4,
                1,
                "test",
                c1,
                131072,
                True,
                True,
                False,
                None,
            )
            self.assertEqual(result, "original")
        capture.assert_not_called()
        self.assertEqual(original.call_count, 3)

    def test_eligible_capture_returns_before_host_collectives(self):
        capture = Mock(return_value=True)
        run, original = self.function(capture)
        result = run(
            None,
            SimpleNamespace(shape=(4096, 2048)),
            None,
            0,
            4,
            1,
            "test",
            True,
            131072,
            True,
            True,
            False,
            None,
        )
        self.assertFalse(result)
        capture.assert_called_once()
        original.assert_not_called()

    def test_default_off_never_calls_cuda(self):
        capture = Mock(side_effect=AssertionError("unexpected CUDA call"))
        run, original = self.function(capture, "0")
        result = run(
            None,
            SimpleNamespace(shape=(4096, 2048)),
            None,
            0,
            4,
            1,
            "test",
            True,
            131072,
            True,
            True,
            False,
            None,
        )
        self.assertEqual(result, "original")
        capture.assert_not_called()
        original.assert_called_once()


class Protocol(unittest.TestCase):
    def test_shapes_and_sender_destinations(self):
        tree = ast.parse(
            (
                Path(__file__).parents[3]
                / "vllm/v1/attention/ops/dcp_bytepack_owner.py"
            ).read_text()
        )
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "layout"
        )
        scope: dict[str, Any] = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "layout", "exec"), scope)
        for rows in (128, 129, 496, 512, 1008, 2048, 2177, 4096):
            for width in (6, 7):
                for owner in range(4):
                    splits, offsets, n, stride = scope["layout"](rows, width, owner)
                    self.assertEqual(sum(splits), rows)
                    self.assertLessEqual(4 * n * stride, 56 * 1024 * 1024)
                    destinations = [
                        (sender * n * stride, (sender + 1) * n * stride)
                        for sender in range(4)
                    ]
                    self.assertEqual(destinations[-1][1], 4 * n * stride)
                    self.assertEqual(offsets[owner], sum(splits[:owner]))

    def test_flag_slots_do_not_alias_ring_or_each_other(self):
        channels: dict[tuple[int, int], tuple[int, int]] = {}
        for phase in (192, 196, 200):
            for sender in range(4):
                for dest in range(4):
                    if sender != dest:
                        flag = (dest, phase + sender)
                        counter = (sender, phase + dest)
                        self.assertNotIn(flag, channels)
                        channels[flag] = counter
                        self.assertGreaterEqual(flag[1], 192)
                        self.assertLess(flag[1], 256)
        self.assertEqual(len(channels), 36)
        self.assertEqual(len(set(channels.values())), 36)


if __name__ == "__main__":
    unittest.main()
