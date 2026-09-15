# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate installed production indexer against per-request isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_indexer_ragged_repro import DEV, D, H, build_cache, mqa_logits

import vllm.model_executor.layers.sparse_attn_indexer as sparse
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
)

for lengths in ([1, 2], [2, 1], [2, 2], [1, 2, 1, 2]):
    B, n, width = len(lengths), sum(lengths), max(lengths)
    contexts = [500 + 100 * i for i in range(B)]
    cache, bt, _ = build_cache(contexts, 16)
    q = (torch.randn(n, H, D, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    w = torch.rand(n + 8, H, device=DEV)
    for i in range(n + 8):
        w[i, (5 * i + 3) % H] += 3
    dl_cpu = torch.tensor(lengths, dtype=torch.int32)
    dl = dl_cpu.to(DEV)
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_seq_lens_buffer = torch.zeros(64, dtype=torch.int32, device=DEV)
    builder.offsets_buffer = torch.arange(4, dtype=torch.int32, device=DEV)
    seq, _, _, _, padded, _ = builder._prepare_decode_tensors(
        torch.tensor(contexts, dtype=torch.int32, device=DEV),
        bt,
        dl,
        dl_cpu,
        dl.cumsum(0) - dl,
        B,
        n,
        True,
        2,
        width,
    )
    decode = SimpleNamespace(
        decode_lens=dl,
        requires_padding=padded,
        seq_lens=seq,
        block_table=bt,
        schedule_metadata=get_paged_mqa_logits_metadata(
            seq, 64, torch.cuda.get_device_properties(0).multi_processor_count
        ),
        global_seq_lens=None,
    )
    meta = DeepseekV32IndexerMetadata(
        seq_lens=seq,
        max_seq_len=max(contexts),
        slot_mapping=torch.arange(n, device=DEV),
        num_decodes=B,
        num_decode_tokens=n,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    original = sparse.fp8_fp4_paged_mqa_logits
    captured = []

    def recording(*args, original=original, captured=captured, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out.clone())
        return out

    with (
        patch.object(
            sparse,
            "get_forward_context",
            lambda meta=meta: SimpleNamespace(attn_metadata={"test": meta}),
        ),
        patch.object(sparse, "fp8_fp4_paged_mqa_logits", recording),
    ):
        topk = sparse.sparse_attn_indexer(
            torch.empty(n, 1, device=DEV),
            "test",
            cache.squeeze(-2),
            q,
            None,
            None,
            w,
            128,
            None,
            256,
            D,
            4096,
            4096,
            torch.empty(B * width, 256, device=DEV, dtype=torch.int32),
            True,
        )
    start, max_delta = 0, 0.0
    for b, count in enumerate(lengths):
        ctx = torch.tensor(
            [[contexts[b] - count + 1 + j for j in range(count)]],
            dtype=torch.int32,
            device=DEV,
        )
        ref = mqa_logits(
            q[start : start + count].reshape(1, count, H, D),
            cache,
            w[start : start + count],
            ctx,
            bt[b : b + 1],
        )
        for j in range(count):
            actual_ctx = contexts[b] - count + 1 + j
            assert int(seq[b, j]) == actual_ctx
            delta = (
                (captured[0][b * width + j, :actual_ctx] - ref[j, :actual_ctx])
                .abs()
                .max()
                .item()
            )
            max_delta = max(max_delta, delta)
            assert delta == 0, (lengths, b, j, delta)
            expected = set(ref[j, :actual_ctx].topk(256).indices.tolist())
            actual = set(topk[start + j].tolist())
            assert expected == actual, (lengths, b, j, len(expected ^ actual))
        start += count
    print(
        "PASS",
        lengths,
        "installed logits exact; topk sets exact; max_delta",
        max_delta,
        flush=True,
    )
