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

    # DIAGNOSTIC (draft-TP-over-PP spike): report the draft attention head
    # sharding so we can confirm the draft KV shards N-way across the PP ranks.
    # Only in the experimental mode, to keep normal runs quiet.
    if getattr(speculative_config, "draft_tp_over_pp", False):
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            import logging as _logging

            _log = _logging.getLogger("vllm")
            _tpw = get_tensor_model_parallel_world_size()
            _tpr = get_tensor_model_parallel_rank()
            for _n, _m in draft_model.named_modules():
                if hasattr(_m, "num_kv_heads") and hasattr(_m, "num_heads"):
                    _log.info(
                        "[TPPP-DIAG] draft attn %s: tp_world=%d tp_rank=%d "
                        "num_kv_heads(per-rank)=%s num_heads(per-rank)=%s "
                        "total_kv=%s kv_size=%s",
                        _n, _tpw, _tpr, getattr(_m, "num_kv_heads", "?"),
                        getattr(_m, "num_heads", "?"),
                        getattr(_m, "total_num_kv_heads", "?"),
                        getattr(_m, "kv_size", "?"),
                    )
                    break
        except Exception:  # pragma: no cover
            pass

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

    # draft-TP-over-PP: the draft is TP-sharded across the PP ranks while the
    # target is TP=1 (unsharded). Sharing embed_tokens / lm_head would splice a
    # full (tp=1) tensor into the draft's sharded (tp=N) layers -> shape/rank
    # mismatch. Force the draft to use its own (sharded) weights. See
    # TP_DRAFT_PP_FINDINGS.md.
    draft_tp_over_pp = getattr(speculative_config, "draft_tp_over_pp", False)

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if not draft_tp_over_pp and _shareable(target_embed) and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if not draft_tp_over_pp and _shareable(target_lm_head) and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
