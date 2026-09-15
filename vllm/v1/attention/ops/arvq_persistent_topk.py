# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Corrected SM120 selector for images retaining prebuilt vLLM extensions."""

import functools
from pathlib import Path

import torch


@functools.cache
def _operator():
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid

    library = Path(nvfp4_arvq_hybrid.__file__).with_name("arvq") / "topk.so"
    if not library.is_file():
        raise RuntimeError(
            "VLLM_DSA_FIXED_PERSISTENT_TOPK requires arvq/topk.so; "
            "rebuild the ARVQ image with build_topk.sh"
        )
    torch.ops.load_library(str(library))
    return torch.ops.arvq_indexer.persistent_topk


def persistent_topk(logits, lengths, output, workspace, k, max_seq_len):
    _operator()(logits, lengths, output, workspace, k, max_seq_len)
