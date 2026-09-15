# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Avoid repeatedly decoding the same expert weights within reference prefill."""

import os


def install():
    from vllm.model_executor.layers.quantization import nvfp4_arvq_hybrid as arvq

    if getattr(arvq, "_task44_reference_chunk", False):
        return
    original = arvq.arvq_mlp
    chunk = int(os.environ["VLLM_TASK44_REFERENCE_CHUNK"])
    assert 1 <= chunk <= 1024

    def call(x, topk_weights, topk_ids, lookups, tensors, alphas, chunk_tokens):
        return original(x, topk_weights, topk_ids, lookups, tensors, alphas, chunk)

    arvq.arvq_mlp = call
    arvq._task44_reference_chunk = True
    print("TASK44 reference expert chunk", chunk, flush=True)
