# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for rank-uniform wire width and conservative fallback."""

import ast
import os
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch


class Gates(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(
            (
                Path(__file__).parents[3] / "vllm/v1/attention/ops/dcp_bytepack.py"
            ).read_text()
        )
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "width_for"
        )
        scope: dict[str, Any] = dict(
            os=os,
            _shadow_failed=False,
            torch=NS(
                int32="i32",
                float32="f32",
                cuda=NS(
                    is_current_stream_capturing=lambda: False,
                    get_device_capability=lambda device: (12, 0),
                ),
            ),
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<gate>", "exec"), scope)
        self.fn = scope["width_for"]
        self.torch = scope["torch"]
        self.args: dict[str, Any] = dict(
            logits=NS(shape=(512, 32768), ndim=2, dtype="f32", device="gpu"),
            ids=NS(shape=(512, 2048), dtype="i32", device="gpu"),
            starts=NS(
                shape=(512,), dtype="i32", device="gpu", is_contiguous=lambda: True
            ),
            rank=0,
            world=4,
            interleave=1,
            c1_prefill=True,
            context=131072,
            raw_mode=True,
            canonical=True,
            query_split=False,
        )
        self.env = {
            "VLLM_EXPERIMENT_DCP_BYTEPACK": "1",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "1048576",
        }

    def test_width_uniform_at_boundaries(self):
        for context, width in [
            (4096, 6),
            (131072, 6),
            (131073, 7),
            (524288, 7),
            (524289, 0),
        ]:
            with patch.dict(os.environ, self.env, clear=True):
                for rank in range(4):
                    self.assertEqual(
                        self.fn(**{**self.args, "context": context, "rank": rank}),
                        width,
                    )

    def test_default_off_and_unsupported(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.fn(**{k: None for k in self.args}), 0)
        for key, value in [
            ("c1_prefill", False),
            ("world", 2),
            ("interleave", 4),
            ("raw_mode", False),
            ("canonical", False),
            ("query_split", True),
        ]:
            with patch.dict(os.environ, self.env, clear=True):
                self.assertEqual(self.fn(**{**self.args, key: value}), 0)

    def test_capture_shape_and_local_id_bound(self):
        with patch.dict(os.environ, self.env, clear=True):
            self.torch.cuda.is_current_stream_capturing = lambda: True
            self.assertEqual(self.fn(**self.args), 0)
            self.torch.cuda.is_current_stream_capturing = lambda: False
            self.args["logits"].shape = (512, 65536)
            self.assertEqual(self.fn(**self.args), 0)
            self.args["context"] = 131073
            self.assertEqual(self.fn(**self.args), 7)
            self.args["ids"].shape = (513, 2048)
            self.assertEqual(self.fn(**self.args), 0)

    def test_outer_topk_guard_does_not_import_for_other_configs(self):
        source = (
            Path(__file__).parents[3]
            / "vllm/model_executor/layers/sparse_attn_indexer.py"
        )
        tree = ast.parse(source.read_text())
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "topk_tokens == 2048" in ast.unparse(n.test)
            and "VLLM_EXPERIMENT_DCP_BYTEPACK" in ast.unparse(n.test)
        )
        with patch.dict(os.environ, self.env, clear=True):
            function = ast.parse("def guarded():\n    pass\n").body[0]
            assert isinstance(function, ast.FunctionDef)
            function.body = [branch]
            scope: dict[str, Any] = dict(os=os, topk_tokens=1024)
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[function], type_ignores=[])
                    ),
                    str(source),
                    "exec",
                ),
                scope,
            )
            self.assertIsNone(scope["guarded"]())

    def test_generic_row_boundaries_and_remainders(self):
        for rows in (127, 128, 129, 196, 252, 260, 496, 1008, 2177, 4032, 4096, 4097):
            self.args["ids"].shape = (rows, 2048)
            self.args["logits"].shape = (rows, 32768)
            self.args["starts"].shape = (rows,)
            with self.subTest(rows=rows), patch.dict(os.environ, self.env, clear=True):
                for rank in range(4):
                    self.assertEqual(
                        self.fn(**{**self.args, "rank": rank}),
                        6 if 128 <= rows <= 4096 else 0,
                    )


class Wire(unittest.TestCase):
    def test_raw_bits_and_ids(self):
        patterns = [
            0,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0xFFC04321,
        ]
        for width in (6, 7):
            for rank in range(4):
                n = 2048
                payload = bytearray(width * n)
                values = []
                for j in range(n):
                    local = (
                        -1 if j % 7 == 0 else (65534 - j if width == 6 else 262143 - j)
                    )
                    token = local if width == 6 else local * 4 + rank
                    encoded = (
                        (0xFFFF if width == 6 else 0xFFFFFF) if local < 0 else token
                    )
                    bits = patterns[j % len(patterns)]
                    struct.pack_into("<I", payload, j * 4, bits)
                    for plane in range(width - 4):
                        payload[4 * n + plane * n + j] = (encoded >> (8 * plane)) & 255
                    values.append((bits, -1 if local < 0 else local * 4 + rank))
                for j, (bits, token) in enumerate(values):
                    decoded = sum(
                        payload[4 * n + plane * n + j] << (8 * plane)
                        for plane in range(width - 4)
                    )
                    if decoded == (0xFFFF if width == 6 else 0xFFFFFF):
                        decoded = -1
                    elif width == 6:
                        decoded = decoded * 4 + rank
                    self.assertEqual(
                        (struct.unpack_from("<I", payload, j * 4)[0], decoded),
                        (bits, token),
                    )

    def test_canonical_selector_methods_unchanged(self):
        root = Path(__file__).parents[3]

        def methods(path):
            tree = ast.parse(path.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            return {
                n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)
            }

        old = methods(
            root / "vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py"
        )
        new = methods(root / "vllm/v1/attention/ops/dcp_bytepack_selector.py")
        for name in old:
            if name not in ("__init__", "kernel", "compile"):
                self.assertEqual(old[name], new[name], name)


if __name__ == "__main__":
    unittest.main()
