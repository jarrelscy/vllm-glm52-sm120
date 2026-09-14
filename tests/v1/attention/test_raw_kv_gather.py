# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only production-shaped gate fixtures; imports no CUDA runtime."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm/v1/attention/ops/raw_kv_gather.py"


class GateTest(unittest.TestCase):
    def setUp(self):
        source = SOURCE
        fn = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_eligible"
        )
        self.copyfree = True
        scope: dict[str, Any] = {
            "os": os,
            "torch": NS(
                bfloat16="bf16",
                uint8="u8",
                cuda=NS(
                    is_current_stream_capturing=lambda: False,
                    get_device_capability=lambda _: (12, 0),
                ),
            ),
            "_dcp_rs_copyfree_supported": lambda _: self.copyfree,
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope
        )
        self.gate = scope["_eligible"]
        impl = type("FlashInferMLASparseSM120Impl", (), {})()
        impl.__dict__.update(
            dcp_world_size=4,
            kv_scale_format="arbitrary_fp32",
            lse_base_on_e=False,
            need_to_return_lse_for_decode=True,
            supports_quant_query_input=False,
        )
        self.layer = NS(
            impl=impl,
            q_pad_num_heads=None,
            is_aiter_triton_fp4_bmm_enabled=False,
            is_aiter_triton_fp8_bmm_enabled=False,
            dcp_a2a=False,
            dcp_a2a_exact=False,
            W_UK_T=NS(
                shape=(16, 192, 512), dtype="bf16", stride=lambda: (229376, 512, 1)
            ),
        )
        self.query = NS(
            shape=(4096, 16, 256),
            dtype="bf16",
            stride=lambda: (4096, 256, 1),
            device="cuda",
        )
        self.cache = NS(
            shape=(512, 64, 656),
            ndim=3,
            dtype="u8",
            stride=lambda n: (41984, 656, 1)[n],
        )
        self.meta = NS(
            num_reqs=1,
            num_actual_tokens=4096,
            max_seq_len=131072,
            topk_tokens=2048,
            block_size=64,
            cp_kv_cache_interleave_size=1,
        )
        self.group = NS(world_size=4)
        self.env = {
            "NCCL_ALGO": "RING,TREE",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "1048576",
            "NCCL_P2P_LEVEL": "SYS",
            "VLLM_DSA_CANONICAL_TOPK": "inkernel",
            "VLLM_GLM_DCP_RS_STAGED": "1",
            "VLLM_GLM_DCP_RS_VIEW": "1",
        }

    def eligible(self):
        return self.gate(
            self.layer, self.query, self.cache, self.meta, True, self.group
        )

    def test_actual_uppercase_and_spacing(self):
        for algo in ("RING,TREE", "Ring,Tree", " ring , TREE "):
            with (
                self.subTest(algo=algo),
                patch.dict(os.environ, {**self.env, "NCCL_ALGO": algo}, clear=True),
            ):
                self.assertTrue(self.eligible())

    def test_unsupported_collectives_and_caps(self):
        for key, value in [
            ("NCCL_ALGO", "TREE"),
            ("NCCL_MAX_NCHANNELS", "8"),
            ("NCCL_BUFFSIZE", "4194304"),
            ("VLLM_GLM_DCP_RS_STAGED", "0"),
            ("VLLM_GLM_DCP_RS_VIEW", "0"),
        ]:
            with (
                self.subTest(key=key),
                patch.dict(os.environ, {**self.env, key: value}, clear=True),
            ):
                self.assertFalse(self.eligible())
        with patch.dict(os.environ, self.env, clear=True):
            self.copyfree = False
            self.assertFalse(self.eligible())
            self.copyfree = True
            self.group.world_size = 2
            self.assertFalse(self.eligible())
            self.group.world_size = 4
            for name in ("dcp_a2a", "dcp_a2a_exact"):
                setattr(self.layer, name, True)
                self.assertFalse(self.eligible())
                setattr(self.layer, name, False)
            self.layer.impl.need_to_return_lse_for_decode = False
            self.assertFalse(self.eligible())

    def test_shape_context_concurrency_fallback(self):
        with patch.dict(os.environ, self.env, clear=True):
            self.meta.max_seq_len = 524289
            self.assertFalse(self.eligible())
            self.meta.max_seq_len = 131072
            self.meta.num_reqs = 4
            self.assertFalse(self.eligible())
            self.meta.num_reqs = 1
            self.query.shape = (4, 16, 256)
            self.assertFalse(self.eligible())

    def test_layout_and_backend_fallback(self):
        with patch.dict(os.environ, self.env, clear=True):
            for length in (4096, 4097, 8193, 131072, 262144, 262145, 524288):
                self.meta.max_seq_len = length
                self.assertTrue(self.eligible())
            for length in (4095, 524289):
                self.meta.max_seq_len = length
                self.assertFalse(self.eligible())
            self.meta.max_seq_len = 8192
            self.query.stride = lambda: (4097, 256, 1)
            self.assertFalse(self.eligible())
            self.query.stride = lambda: (4096, 256, 1)
            self.layer.W_UK_T.stride = lambda: (98304, 512, 1)
            self.assertFalse(self.eligible())
            self.layer.W_UK_T.stride = lambda: (229376, 512, 1)
            self.cache.stride = lambda n: (43008, 656, 1)[n]
            self.assertTrue(self.eligible())  # physical page padding is supported
            self.cache.stride = lambda n: (43008, 657, 1)[n]
            self.assertFalse(self.eligible())
            self.cache.stride = lambda n: (41984, 656, 1)[n]
            self.layer.impl.lse_base_on_e = True
            self.assertFalse(self.eligible())

    def test_default_off_returns_before_collectives(self):
        fn = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "try_raw_kv_gather"
        )
        scope: dict[str, Any] = {"envs": NS(VLLM_GLM_RAW_KV_GATHER=False)}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), scope
        )
        self.assertIsNone(scope["try_raw_kv_gather"](None, None, None, None, None))

    def test_registered_flag_default_off(self):
        tree = ast.parse((SOURCE.parents[3] / "envs.py").read_text())
        node = next(
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "VLLM_GLM_RAW_KV_GATHER"
        )
        scope: dict[str, Any] = {"os": os}
        fn = eval(compile(ast.Expression(node), "<flag>", "eval"), scope)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(fn())
        with patch.dict(os.environ, {"VLLM_GLM_RAW_KV_GATHER": "1"}, clear=True):
            self.assertTrue(fn())
