# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling MTP (multi-token prediction) draft model.

Semantics restored from the reference implementation (upstream commit
5624be7aa "[Model] Add Inkling LoRA and MTP support", which mirrors the
``mtp_model.py`` shipped with the original checkpoint):

* Each MTP depth ``i`` owns ``hidden_norm`` / ``embed_norm`` RMSNorms, an
  ``input_proj`` (``2H -> H``, consuming ``cat([hidden_norm(hidden),
  embed_norm(embed)])`` — hidden first) and a full Inkling transformer block
  (dense MLP, sliding-window or full attention per depth via
  ``mtp_config.local_layer_ids``).
* The block output is returned raw: with ``chain_hidden_post_norm=False``
  (this checkpoint) there is NO chain norm, NO per-depth final norm and NO
  extra residual. The same raw value is both the logits input (through the
  shared muP-scaled LM head) and the previous hidden fed to the next depth.
* The depth layers consume the *backbone-normed* embedding
  (``embed_norm(embed(ids))``, weight ``model.llm.embed_norm.weight``), not
  the raw one: the per-depth ``embed_norm`` weights are near-identity trims,
  unlike the backbone's whitening ``embed_norm``. Feeding raw embeddings
  costs real acceptance (upstream: MTP1 ~0.85 -> ~0.70).
* The draft shares the target's token embedding table and LM head
  (``load_eagle_model`` attaches both; neither is materialized here).

Scheduling (see ``spec_decode/autoregressive/speculator.py``): the first
``min(n_predict, num_speculative_tokens)`` draft tokens are produced by
re-running the draft prefill window once per depth module
(``spec_step_idx`` selects the depth), chaining the full-window hidden
module-to-module and shifting input ids left by one per step.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from .layernorm import InklingRMSNorm
from .logits_processor import InklingLogitsProcessor
from .model import InklingDecoderLayer, _TmlForCausalLMBase, _sconv_add_norm

logger = init_logger(__name__)

# Matches the real checkpoint's per-layer MTP tensor names, e.g.
# "model.mtp.layers.3.transformer_block.attn.wq_du.weight".
_MTP_LAYER_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.")
# The backbone's whitening embed_norm, applied to the shared embedding
# table's rows before the per-depth (near-identity) embed_norm.
_BACKBONE_EMBED_NORM_NAME = "model.llm.embed_norm.weight"


class InklingMTPLayer(nn.Module):
    """One MTP depth: norm both inputs, fuse (2H->H), run an Inkling block.

    ``transformer_block`` reuses the main model's own ``InklingDecoderLayer``
    unmodified -- checkpoint weight names/shapes for it are identical to a
    main-model decoder layer. All 8 MTP layers use dense MLP only (verified:
    no ``mlp.experts.*`` keys anywhere under ``model.mtp.*``), so
    ``dense_mlp_idx`` is forced high on a shallow config copy regardless of
    the shared text_config's own value.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        mtp_layer_idx: int,
        is_local: bool,
    ) -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config.text_config
        quant_config = None  # MTP layers are unquantized BF16 on the release.

        self.embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # input_proj.weight: [hidden_size, 2*hidden_size], plain Linear over
        # concat(hidden_norm_out, embed_norm_out) -- verified shape on the
        # real release ([6144, 12288] for hidden_size=6144).
        self.input_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_proj",
        )

        block_config = copy.copy(config)
        # Force dense MLP for every MTP layer regardless of the main model's
        # dense_mlp_idx (see class docstring).
        block_config.dense_mlp_idx = 10**9
        self.transformer_block = InklingDecoderLayer(
            block_config,
            layer_id=mtp_layer_idx,
            is_local=is_local,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer_block",
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Checkpoint semantics: hidden-first, embed-second (the REVERSE of
        # DeepSeek MTP's enorm/hnorm cat order; established empirically in
        # task #88 and confirmed by the reference implementation's
        # embed_dual_rmsnorm_cat ordering).
        hidden = self.hidden_norm(previous_hidden_states)
        embed = self.embed_norm(inputs_embeds)
        combined = torch.cat([hidden, embed], dim=-1)

        # return_bias=False: ReplicatedLinear.forward() returns the output
        # tensor directly, NOT a (tensor, bias) tuple -- do not unpack.
        proj_out = self.input_proj(combined)

        hidden_states, pending = self.transformer_block(positions, proj_out)
        # Resolve the transformer_block's deferred MLP delta within this same
        # call (MTP layers run standalone, not chained into a "next layer"
        # the way the main model's stack is). norm=None: with
        # chain_hidden_post_norm=False there is no chain norm -- the raw
        # residual-stream value is both the logits input and the next
        # depth's hidden.
        return _sconv_add_norm(
            pending[0], hidden_states, pending[1], None, positions
        )[1]


class InklingMTPPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_hf_config = speculative_config.draft_model_config.hf_config
        text_config = draft_hf_config.text_config
        mtp_config = getattr(draft_hf_config, "mtp_config", {}) or {}
        # The checkpoint ships num_nextn_predict_layers depth blocks, but only
        # the first num_speculative_tokens are exercised (step i uses depth
        # i). Build only those to save memory -- each depth is a full Inkling
        # block with its own sconv caches and KV cache.
        n_predict = mtp_config.get("num_nextn_predict_layers", 1)
        num_spec = speculative_config.num_speculative_tokens
        self.num_mtp_layers = min(n_predict, num_spec) if num_spec else n_predict
        local_layer_ids = set(mtp_config.get("local_layer_ids", []))

        self.layers = nn.ModuleDict(
            {
                str(idx): InklingMTPLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    mtp_layer_idx=idx,
                    is_local=idx in local_layer_ids,
                )
                for idx in range(self.num_mtp_layers)
            }
        )
        # The target's raw token embedding table, attached post-load by
        # load_eagle_model (never materialized here: a replicated copy would
        # transiently double the ~2.3 GiB table).
        self.embed_tokens = None  # type: ignore[assignment]
        # The depth layers consume the *backbone-normed* embedding
        # (embed_norm(embed(ids))), not the raw one: mtp embed_norm weights
        # are near-identity (trained on already-normalized inputs). Weight
        # loaded from the target's model.llm.embed_norm.weight; gated like
        # the target's InklingModel.embed_norm.
        self.backbone_embed_norm = (
            InklingRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
            if text_config.use_embed_norm
            else None
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draft-prefill embedding: gather + backbone embed_norm, then the
        target's tower embeddings scattered in unnormed (the backbone
        convention -- MM embeds are merged after embed_norm)."""
        embeds = self.embed_tokens(input_ids)
        if self.backbone_embed_norm is not None:
            embeds = self.backbone_embed_norm(embeds)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:  # type: ignore[arg-type]
            return embeds
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            # inputs_embeds from the speculator's MM path are already
            # backbone-normed (embed_input_ids); this raw-ids path norms here.
            inputs_embeds = self.embed_tokens(input_ids)
            if self.backbone_embed_norm is not None:
                inputs_embeds = self.backbone_embed_norm(inputs_embeds)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            positions, previous_hidden_states, inputs_embeds
        )


class InklingMTP(nn.Module):
    """Top-level registered class (``InklingMTPModel`` -> ``InklingMTP``)."""

    # Reuse the main model's substr/stacked/suffix weight-name rules
    # (w13_dn/w2_md, wq_du/wk_dv/wv_dv/wr_du -> qkvr stacking) layered with
    # an MTP-specific prefix rule.
    hf_to_vllm_mapper = _TmlForCausalLMBase.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.mtp.layers.": "model.layers.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = self.config.text_config
        self.model = InklingMTPPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # The target's (vocab-sharded) LM head, attached by load_eagle_model;
        # never materialized here (same reasoning as model.embed_tokens).
        self.lm_head = None  # type: ignore[assignment]
        # The MTP shares the base model's LM head, which is trained on
        # ``hidden / mup``-scaled inputs -- apply the same muP scaling for a
        # matching logit scale (argmax-invariant for greedy draft sampling,
        # but it matters for gumbel sampling at temperature > 0). NO final
        # norm: the reference decodes the raw block output.
        self.logits_processor = InklingLogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
            logits_mup_width_multiplier=text_config.logits_mup_width_multiplier,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        local_layer_ids = set(
            getattr(self.config, "mtp_config", {}).get("local_layer_ids", [])
        )
        num_mtp_layers = self.model.num_mtp_layers

        def _iter_loadable_weights():
            for name, weight in weights:
                if name == _BACKBONE_EMBED_NORM_NAME:
                    yield "model.backbone_embed_norm.weight", weight
                    continue
                m = _MTP_LAYER_RE.match(name)
                if m is None:
                    # Not an MTP-layer tensor (main-model layers, audio/
                    # visual towers, embed/unembed, etc.) -- irrelevant to
                    # this module.
                    continue
                if int(m.group(1)) >= num_mtp_layers:
                    # Depth blocks beyond the ones we built
                    # (num_speculative_tokens < n_predict).
                    continue
                yield name, weight

        mapped = self.hf_to_vllm_mapper.apply(_iter_loadable_weights())

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, weight in mapped:
            shard_id = getattr(weight, "shard_id", None)
            # Replicate K/V conv-free GQA heads when tp_size > num_kv_heads,
            # mirroring _load_inkling_weights' identical handling for the
            # main model (same InklingAttention/qkvr structure is reused
            # here for transformer_block).
            if shard_id in (1, 2) and name.endswith(".attn.qkvr.weight"):
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )

                m = re.search(r"\.layers\.(\d+)\.", name)
                if m is not None:
                    lid = int(m.group(1))
                    is_local = lid in local_layer_ids
                    n_kv = (
                        self.config.text_config.swa_num_key_value_heads
                        if is_local
                        else self.config.text_config.num_key_value_heads
                    )
                    head_dim = (
                        self.config.text_config.swa_head_dim
                        if is_local
                        else self.config.text_config.head_dim
                    )
                    tp_size = get_tensor_model_parallel_world_size()
                    tp_rank = get_tensor_model_parallel_rank()
                    if tp_size > n_kv and weight.shape[0] == n_kv * head_dim:
                        kv_idx = (tp_rank * n_kv) // tp_size
                        weight = weight.narrow(0, kv_idx * head_dim, head_dim)
                        weight.shard_id = shard_id
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                weight_loader(param, weight, shard_id)
            else:
                weight_loader(param, weight)
            loaded_params.add(name)

        loaded_layers = {
            int(m.group(1))
            for name in loaded_params
            if (m := re.search(r"\.layers\.(\d+)\.", name)) is not None
        }
        missing = set(range(num_mtp_layers)) - loaded_layers
        if missing:
            raise ValueError(
                f"MTP speculative decoding layer(s) {sorted(missing)} weights "
                f"missing from checkpoint (expected {num_mtp_layers} layers "
                f"under model.mtp.layers.*)."
            )
        if (
            self.model.backbone_embed_norm is not None
            and "model.backbone_embed_norm.weight" not in loaded_params
        ):
            raise ValueError(
                "Inkling MTP requires the backbone embed_norm "
                f"({_BACKBONE_EMBED_NORM_NAME}) but it was not found in the "
                "checkpoint."
            )
        logger.info_once("Inkling MTP draft model loaded: %d params", len(loaded_params))
        return loaded_params
