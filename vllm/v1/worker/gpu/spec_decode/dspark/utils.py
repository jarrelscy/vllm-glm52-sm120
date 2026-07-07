# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag

    # DSpark uses non-causal attention.
    causal = False
    # The DSpark draft is a small dense-attention model (e.g. Qwen3DSparkModel),
    # so it cannot use the target's MLA-specific KV cache dtype (e.g.
    # ``fp8_ds_mla``). Give the draft its own bf16 ("auto") KV cache — it is tiny
    # (few layers, head_size 64) so the memory cost is negligible.
    draft_cache_config = vllm_config.cache_config
    if draft_cache_config is not None and draft_cache_config.cache_dtype not in (
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ):
        draft_cache_config = replace(draft_cache_config, cache_dtype="auto")
    draft_vllm_config = replace(
        vllm_config,
        cache_config=draft_cache_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # DSpark is built only on the last PP rank (init_speculator gates on
    # is_last_pp_rank) and its draft layers are a plain ModuleList (not PP
    # partitioned), so the drafter is fully local. The target propagates the
    # required aux hidden states down the pipeline, so PP is supported.

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    # Under pipeline parallelism the target's embed_tokens lives only on the
    # first stage (a PPMissingLayer elsewhere) and its lm_head only on the last
    # stage. The drafter runs on the last stage, so the target embedding is not
    # available to share here -- fall back to the drafter's own weights (the
    # RedHat DSpark checkpoint ships embed_tokens.weight and lm_head.weight).
    def _shareable(module) -> bool:
        return module is not None and hasattr(module, "weight")

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if _shareable(target_embed) and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if _shareable(target_lm_head) and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
