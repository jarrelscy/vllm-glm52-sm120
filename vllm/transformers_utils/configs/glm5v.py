# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2-Vision (glm5v) model configuration.

glm5v is baseten/GLM-5.2-Vision: Kimi-K2.5's MoonViT vision tower + trained
K2VL PatchMerger projector grafted onto a GLM-5.2 (glm_moe_dsa) text decoder.
The config mirrors :class:`KimiK25Config` (``vision_config`` + ``text_config``
+ media placeholder fields) with the text model swapped to the
transformers-native ``glm_moe_dsa`` config and the image placeholder set to
GLM's ``<|image|>`` (154854).
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Glm5vVisionConfig(PretrainedConfig):
    """MoonViT vision tower + PatchMerger projector config.

    Field names/defaults mirror :class:`KimiK25VisionConfig`; the checkpoint's
    duplicated ``vt_*`` aliases arrive via ``**kwargs`` and are stored as
    plain attributes (vLLM's MoonViT code reads the unprefixed names).
    """

    model_type = "glm5v_vision"

    def __init__(
        self,
        # Vision tower
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        # MM projector
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        text_hidden_size: int = 6144,  # GLM-5.2 hidden (Kimi default is 7168)
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Vision tower
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        # MM projector
        self.mm_projector_type = mm_projector_type
        if mm_hidden_size is not None:
            self.mm_hidden_size = mm_hidden_size
        else:
            self.mm_hidden_size = hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size


class Glm5vConfig(PretrainedConfig):
    """glm5v top-level config: MoonViT ``vision_config`` + GLM-5.2
    ``text_config`` (glm_moe_dsa)."""

    model_type = "glm5v"

    def __init__(
        self,
        vision_config: dict | Glm5vVisionConfig | None = None,
        text_config: dict | PretrainedConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 154854,  # GLM <|image|>
        pad_token_id: int = 154820,
        use_unified_vision_chunk: bool = True,
        video_placeholder: str = "<|glm5v_video_placeholder|>",
        encoder_only: bool = False,
        language_only: bool = False,
        **kwargs,
    ):
        # Vision config (MoonViT)
        if vision_config is None:
            self.vision_config = Glm5vVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Glm5vVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Text config (GLM-5.2 / glm_moe_dsa), built via AutoConfig so the
        # transformers-native GlmMoeDsaConfig class is used — identical to
        # loading the bare GLM-5.2 checkpoint (text path must stay prod-exact).
        raw_text = dict(text_config) if isinstance(text_config, dict) else None
        if text_config is None:
            self.text_config = AutoConfig.for_model("glm_moe_dsa")
        elif isinstance(text_config, dict):
            tc = dict(text_config)
            tc.setdefault("model_type", "glm_moe_dsa")
            # Some upstream glm5v configs carry the legacy layer_types value
            # "deepseek_sparse_attention" which newer transformers rejects;
            # the DSA path is selected from model_type + DSA fields, not
            # layer_types, so drop it. (Our v2 text config does not set it.)
            tc.pop("layer_types", None)
            self.text_config = AutoConfig.for_model(**tc)
        else:
            self.text_config = text_config

        # Older transformers GlmMoeDsaConfig releases drop/clobber raw DSA
        # fields the sparse-attention path needs (fixed upstream in
        # transformers #46338); restore them from the raw dict.
        if raw_text is not None:
            for key in ("qk_rope_head_dim", "index_topk_freq"):
                if key in raw_text:
                    setattr(self.text_config, key, raw_text[key])
            if hasattr(self.text_config, "qk_nope_head_dim") and hasattr(
                self.text_config, "qk_rope_head_dim"
            ):
                self.text_config.qk_head_dim = (
                    self.text_config.qk_nope_head_dim
                    + self.text_config.qk_rope_head_dim
                )

        # Set mm_hidden_size (projector output dim) to the text hidden size if
        # not explicitly set — same rule as KimiK25Config.
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        self.encoder_only = encoder_only
        self.language_only = language_only

        # Propagate quantization config from the text model (Kimi pattern):
        # only the GLM text Linears are quantized; vision tower/projector stay
        # bf16 by construction in the model code.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        """Get hidden size from text config for compatibility."""
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocab size from text config for compatibility."""
        return self.text_config.vocab_size

    def __getattr__(self, name: str):
        """Read-delegate missing attributes to the text config.

        Several vLLM code paths read text-model fields off the TOP-LEVEL
        hf_config (e.g. the V2-runner MTP draft model reads
        vllm_config.model_config.hf_config.num_hidden_layers / n_group /
        rms_norm_eps / n_routed_experts...). For the standalone GLM-5.2
        checkpoint the top-level config IS the text config, so delegate the
        long tail here to keep those paths prod-identical. Only called when
        normal attribute lookup fails; explicit properties above take
        precedence. Writes still store on the wrapper as usual.
        """
        if name.startswith("_") or name == "text_config":
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        d = object.__getattribute__(self, "__dict__")
        tc = d.get("text_config")
        if tc is not None and hasattr(tc, name):
            return getattr(tc, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def _make_text_passthrough(name: str) -> property:
    def fget(self: Glm5vConfig):
        return getattr(self.text_config, name)

    def fset(self: Glm5vConfig, value) -> None:
        setattr(self.text_config, name, value)

    return property(fget, fset)


# DSA (DeepSeek Sparse Attention) fields that vLLM reads from the TOP-LEVEL
# hf_config at runtime (e.g. FlashInferMLASparseMetadataBuilder reads
# `vllm_config.model_config.hf_config.index_topk`, and the global vllm_config
# carries the multimodal wrapper config, not the promoted text config).
# This is the vLLM analog of the SGLang glm5v patch.py DSA/MLA arch wiring.
for _name in (
    "index_topk",
    "index_head_dim",
    "index_n_heads",
    "index_topk_freq",
    "index_topk_pattern",
    "index_skip_topk_offset",
    "indexer_types",
    "indexer_rope_interleave",
    "index_share_for_mtp_iteration",
    "first_k_dense_replace",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "v_head_dim",
    "num_nextn_predict_layers",
):
    setattr(Glm5vConfig, _name, _make_text_passthrough(_name))
del _name
