# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP (multi-token-prediction) draft model for Inkling.

Task #88. Registered as ``InklingMTPModel`` / ``InklingMTP`` via
``vllm/config/speculative.py``'s ``hf_config_override`` (model_type
``inkling_mtp``) and ``vllm/model_executor/models/registry.py``.

Real checkpoint evidence (verified directly against the release
snapshot's ``model.safetensors.index.json`` / tensor shapes, NOT
assumed):
  * ``mtp_config`` = ``{"num_nextn_predict_layers": 8,
    "chain_hidden_post_norm": False, "local_layer_ids": [0, 2, 4, 5, 6, 7]}``
    (a plain dict on the top-level ``InklingConfig``, sibling of
    ``text_config``, NOT nested inside it).
  * Exactly 160 weight tensors under ``model.mtp.layers.{0..7}.*``, 20 per
    layer: ``embed_norm.weight``, ``hidden_norm.weight``,
    ``input_proj.weight`` (shape ``[hidden_size, 2*hidden_size]`` --
    confirmed ``[6144, 12288]`` on the real release, i.e. a plain
    concat(hidden_norm_out, embed_norm_out) -> hidden_size projection —
    NOTE: hidden-first, the REVERSE of DeepSeek's ``eh_proj`` order,
    established empirically via offline fp32 teacher-forcing, task #88),
    plus 16
    ``transformer_block.*`` keys identical in name/shape to a main-model
    ``InklingDecoderLayer`` (``attn_norm``, ``attn.{k_norm,k_sconv,q_norm,
    rel_logits_proj.proj,v_sconv,wk_dv,wo_ud,wq_du,wr_du,wv_dv}``,
    ``attn_sconv``, ``mlp.{global_scale,w13_dn,w2_md}``, ``mlp_norm``,
    ``mlp_sconv``). All BF16 (unquantized) on the real release, unlike the
    NVFP4/AQLM-hybrid main model -- confirmed via direct tensor dtype
    inspection of ``transformer_block.attn.wq_du.weight`` and
    ``transformer_block.mlp.w13_dn.weight``.
  * There is NO per-layer final-norm or lm_head weight anywhere under
    ``model.mtp.*`` (unlike DeepSeek's MTP layers, which do ship their own
    per-step ``.norm``/``.head``). The generic proposer-level sharing in
    ``vllm/v1/spec_decode/llm_base_proposer.py``
    (``_maybe_share_embeddings`` / ``_maybe_share_lm_head``) already
    auto-shares ``embed_tokens`` and each layer's ``shared_head.head``
    with the target model's own instances post-construction for any
    draft model that doesn't set ``has_own_embed_tokens`` /
    ``has_own_lm_head`` (the "MTP model" branch in both methods) -- so
    those two are handled for free and this module only needs to supply
    placeholder modules for them to attach to. ``shared_head.norm`` has
    no equivalent generic auto-share hook, so ``InklingMTP.load_weights``
    below explicitly broadcasts the target's own ``model.llm.norm``
    tensor into every layer's ``shared_head.norm.weight``.

No HF reference implementation exists for this module anywhere (grepped
``transformers/models/inkling/modeling_inkling.py`` across all local
venvs: zero MTP classes; the model's own ``_keys_to_ignore_on_load_unexpected
= [r"model\\.mtp\\..*"]`` shows even the public release discards these
weights). The combiner semantics below (plain concat-then-project, no
extra post-norm) are reverse-engineered from weight names/shapes only.

ASSUMPTION (explicitly flagged per coordinator sign-off, unverified against
any reference): with ``chain_hidden_post_norm=False`` in this checkpoint,
no additional norm is applied after the ``input_proj`` combiner -- the
simplest reading of the flag name, consistent with DeepSeek's simpler
``eh_proj`` pattern this module mirrors structurally. Real correctness
here can only be confirmed later via draft/target logprob agreement or
measured MTP acceptance rate once end-to-end TP4 wiring exists, not from
static checkpoint inspection alone.
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
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from .layernorm import InklingRMSNorm
from .model import InklingDecoderLayer, _TmlForCausalLMBase, _sconv_add_norm

logger = init_logger(__name__)

# Matches the real checkpoint's per-layer MTP tensor names, e.g.
# "model.mtp.layers.3.transformer_block.attn.wq_du.weight".
_MTP_LAYER_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.")
# The target model's own final norm (shared_head.norm has no counterpart
# in the checkpoint -- broadcast this single tensor into every MTP layer).
_TARGET_NORM_NAME = "model.llm.norm.weight"


class InklingMTPLayer(nn.Module):
    """One MTP prediction step.

    ``transformer_block`` reuses the main model's own ``InklingDecoderLayer``
    unmodified -- checkpoint weight names/shapes for it are identical to a
    main-model decoder layer. All 8 MTP layers use dense MLP only (verified:
    no ``mlp.experts.*`` keys anywhere under ``model.mtp.*``), so
    ``dense_mlp_idx`` is forced high on a shallow config copy regardless of
    the shared text_config's own value (which gates dense-vs-MoE for the
    much larger main model stack, not for MTP).
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
        # concat(embed_norm_out, hidden_norm_out) -- verified shape on the
        # real release ([6144, 12288] for hidden_size=6144).
        self.input_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_proj",
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=None
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed_norm(inputs_embeds)
        hidden = self.hidden_norm(previous_hidden_states)

        # Checkpoint semantics: hidden-first, embed-second — the REVERSE of
        # DeepSeek MTP's enorm/hnorm cat order. Determined empirically
        # (task #88): teacher-forcing this layer offline in fp32 over a real
        # committed stream gives median true-next-token rank ~6/201k with
        # [hidden, embed] vs ~130k/201k (and anchor-echo drafts, 0%
        # acceptance) with [embed, hidden].
        combined = torch.cat([hidden, embed], dim=-1)

        # return_bias=False (set at construction) means ReplicatedLinear.
        # forward() returns the output tensor directly, NOT a (tensor, bias)
        # tuple -- do not unpack.
        proj_out = self.input_proj(combined)

        hidden_states, pending = self.transformer_block(positions, proj_out)
        # Resolve the transformer_block's deferred MLP delta within this
        # same call (MTP layers run standalone, not chained into a "next
        # layer" the way the main model's stack is). norm=None here mirrors
        # the chain_hidden_post_norm=False assumption documented in the
        # module docstring: no extra norm on the recycled hidden state.
        hidden_states = _sconv_add_norm(
            pending[0], hidden_states, pending[1], None, positions
        )[1]
        # Second element: shared_head-normed hidden ready for logits.
        return hidden_states, self.shared_head(hidden_states)


class InklingMTPPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_hf_config = speculative_config.draft_model_config.hf_config
        text_config = draft_hf_config.text_config
        mtp_config = getattr(draft_hf_config, "mtp_config", {}) or {}
        self.num_mtp_layers = mtp_config.get("num_nextn_predict_layers", 1)
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
        # Own copy, auto-shared with the target's embed_tokens post-load by
        # llm_base_proposer.py's _maybe_share_embeddings (generic "MTP
        # model" branch) -- real weights never need to land here.
        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(text_config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            positions, previous_hidden_states, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        # hidden_states here is the RAW (pre-shared_head-norm) recycled
        # state returned by forward()'s first tuple element -- re-apply
        # shared_head's norm here, mirroring DeepSeekMultiTokenPredictor's
        # identical re-invocation-at-compute_logits-time pattern.
        normed = mtp_layer.shared_head(hidden_states)
        return self.logits_processor(mtp_layer.shared_head.head, normed)


class InklingMTP(nn.Module):
    """Top-level registered class (``InklingMTPModel`` -> ``InklingMTP``)."""

    # Reuse the main model's substr/stacked/suffix weight-name rules
    # (w13_dn/w2_md, wq_du/wk_dv/wv_dv/wr_du -> qkvr stacking, NVFP4/AQLM
    # scale suffixes -- unused here since MTP is unquantized, but harmless)
    # layered with an MTP-specific prefix rule.
    hf_to_vllm_mapper = _TmlForCausalLMBase.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.mtp.layers.": "model.layers.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = InklingMTPPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass the (pre-norm, post-norm) tuple straight through, matching
        # DeepSeekMTP.forward()'s identical pattern -- do NOT unpack/discard
        # here. model_returns_tuple() (llm_base_proposer.py) now recognizes
        # InklingMTPModel alongside DeepSeekMTPModel, so the harness splits
        # this correctly: compute_logits gets the pre-norm element (and
        # applies shared_head's norm itself), while the post-norm element
        # is what actually gets recycled as the next draft step's
        # previous_hidden_states. Previously this unpacked and discarded
        # the post-norm element, so the recycled state was silently never
        # normed -- the likely root cause of the flat 0% MTP acceptance
        # measured across all 8 draft positions.
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        local_layer_ids = set(
            getattr(self.config, "mtp_config", {}).get("local_layer_ids", [])
        )
        num_mtp_layers = self.model.num_mtp_layers

        def _iter_loadable_weights():
            for name, weight in weights:
                if name == _TARGET_NORM_NAME:
                    # Broadcast the target's own final norm into every MTP
                    # layer's shared_head.norm (no per-layer counterpart in
                    # the checkpoint -- see module docstring).
                    for idx in range(num_mtp_layers):
                        yield f"model.layers.{idx}.shared_head.norm.weight", weight
                    continue
                if not _MTP_LAYER_RE.match(name):
                    # Not an MTP-layer tensor (main-model layers, audio/
                    # visual towers, embed/unembed, etc.) -- irrelevant to
                    # this module, drop before it ever reaches the mapper.
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
        logger.info_once("Inkling MTP draft model loaded: %d params", len(loaded_params))
        return loaded_params
