# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 MTP must find the head inside a multimodal language-model wrapper."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.v1.worker.gpu.spec_decode.eagle import utils


@pytest.mark.parametrize("own_distinct_head", [False, True])
def test_multimodal_mtp_head_sharing(monkeypatch, own_distinct_head):
    target = nn.Module()
    language = nn.Module()
    language.model = nn.Module()
    language.model.embed_tokens = nn.Embedding(8, 4)
    language.lm_head = nn.Linear(4, 8, bias=False)
    target.language_model = language
    target.get_language_model = lambda: language
    assert not hasattr(target, "lm_head")

    draft = nn.Module()
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(8, 4)
    layer = nn.Module()
    layer.shared_head = nn.Module()
    layer.shared_head.norm = nn.LayerNorm(4)
    layer.shared_head.head = nn.Linear(4, 8, bias=False)
    # GLM's MTP loader remaps the same checkpoint lm_head.weight here.
    layer.shared_head.head.weight.data.copy_(language.lm_head.weight)
    draft.model.layers = nn.ModuleDict({"78": layer})
    original_head = layer.shared_head.head
    original_norm = layer.shared_head.norm
    original_bytes = original_head.weight.detach().clone().view(torch.uint8)
    if own_distinct_head:
        draft.has_own_lm_head = True
        draft.lm_head = original_head
        draft.lm_head.weight.data.add_(1)

    monkeypatch.setattr(utils, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )
    result = utils.load_eagle_model(target, config)
    assert result is draft
    assert draft.model.embed_tokens is language.model.embed_tokens
    assert layer.shared_head.norm is original_norm
    if own_distinct_head:
        assert draft.lm_head is original_head
        assert layer.shared_head.head is original_head
        assert original_head.weight.data_ptr() != language.lm_head.weight.data_ptr()
    else:
        assert draft.lm_head is language.lm_head
        assert layer.shared_head.head is language.lm_head
        assert layer.shared_head.head.weight is language.lm_head.weight
        assert torch.equal(
            original_bytes, language.lm_head.weight.detach().view(torch.uint8)
        )
