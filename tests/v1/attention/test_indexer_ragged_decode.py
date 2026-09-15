# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
)


@pytest.mark.parametrize("lengths", [[1, 2], [2, 1], [2, 2], [1, 2, 1, 2]])
def test_native_ragged_context_lengths(lengths):
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_seq_lens_buffer = torch.zeros(64, dtype=torch.int32)
    builder.offsets_buffer = torch.arange(4, dtype=torch.int32)
    dl = torch.tensor(lengths, dtype=torch.int32)
    contexts = torch.arange(len(lengths), dtype=torch.int32) * 100 + 500
    starts = dl.cumsum(0) - dl
    seq, _, _, _, padded, _ = builder._prepare_decode_tensors(
        contexts,
        torch.zeros(len(lengths), 16, dtype=torch.int32),
        dl,
        dl,
        starts,
        len(lengths),
        sum(lengths),
        True,
        2,
        max(lengths),
    )
    expected = contexts[:, None].expand(len(lengths), max(lengths)).clone()
    for b, n in enumerate(lengths):
        expected[b, :n] = torch.arange(contexts[b] - n + 1, contexts[b] + 1)
    assert torch.equal(seq, expected)
    assert padded == (min(lengths) != max(lengths))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("lengths", [[1, 2], [2, 1], [2, 2], [1, 2, 1, 2]])
@pytest.mark.parametrize("fp4", [False, True])
def test_decode_passes_matching_query_and_weight_rows(monkeypatch, lengths, fp4):
    """Exercise the production indexer up to the paged-logits call boundary."""
    device = "cuda"
    n, batch, width = sum(lengths), len(lengths), max(lengths)
    dl = torch.tensor(lengths, device=device, dtype=torch.int32)
    weights = torch.arange((n + 8) * 32, device=device).reshape(n + 8, 32).float()
    q = torch.arange(n, device=device).reshape(n, 1, 1).expand(n, 32, 128)
    q = q.to(torch.uint8 if fp4 else torch.float8_e4m3fn).contiguous()
    qscale = torch.ones(n, 32, 4, device=device, dtype=torch.uint8) if fp4 else None
    meta = object.__new__(DeepseekV32IndexerMetadata)
    meta.slot_mapping = torch.arange(n, device=device)
    meta.num_decodes, meta.num_prefills, meta.num_decode_tokens = batch, 0, n
    meta.decode = SimpleNamespace(
        decode_lens=dl,
        requires_padding=min(lengths) != max(lengths),
        seq_lens=torch.ones(batch, width, dtype=torch.int32, device=device),
        block_table=None,
        schedule_metadata=None,
    )
    monkeypatch.setattr(
        sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test": meta}),
    )
    monkeypatch.setattr(sparse, "kv_cache_as_quant_view", lambda cache, *args: cache)
    monkeypatch.setattr(sparse, "_MTP_DRAFT_REUSE_TOPK", False)

    class ReachedLogits(Exception):
        pass

    def check_rows(query, cache, got_weights, *args, **kwargs):
        packed_q, packed_scales = query
        got_q = packed_q.view(torch.uint8) if fp4 else packed_q.float()
        start = 0
        for b, count in enumerate(lengths):
            for t in range(width):
                row = b * width + t
                if t < count:
                    assert torch.equal(got_weights[row], weights[start + t])
                    assert (got_q[b, t] == start + t).all()
                    if fp4:
                        assert (packed_scales[b, t] == 1).all()
                else:
                    assert (got_weights[row] == 0).all()
                    assert (got_q[b, t] == 0).all()
            start += count
        raise ReachedLogits

    monkeypatch.setattr(sparse, "fp8_fp4_paged_mqa_logits", check_rows)
    with pytest.raises(ReachedLogits):
        sparse.sparse_attn_indexer(
            torch.empty(n, 1, device=device),
            "test",
            torch.empty(0, device=device),
            q,
            qscale,
            None,
            weights,
            128,
            None,
            256,
            128,
            4096,
            4096,
            torch.empty(batch * width, 256, dtype=torch.int32, device=device),
            True,
            use_fp4_cache=fp4,
        )
