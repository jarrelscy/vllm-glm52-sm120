# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2-Vision (glm5v) for vLLM.

glm5v = baseten/GLM-5.2-Vision: Kimi-K2.5's MoonViT vision tower + the trained
K2VL PatchMerger projector grafted onto a GLM-5.2 (GlmMoeDsa) text decoder.
Only the projector was trained; tower and LLM are frozen — the LLM backbone is
byte-identical to the bare GLM-5.2 checkpoint.

Implementation: a thin subclass of the in-tree
:class:`KimiK25ForConditionalGeneration` with three swaps (mirroring the
SGLang ``sglang_glm5v`` plugin):

  1. text backbone ``DeepseekV2ForCausalLM`` -> ``GlmMoeDsaForCausalLM``;
  2. image placeholder -> GLM's ``<|image|>`` (``media_placeholder_token_id``
     154854), with the chat template wrapping each image as
     ``<|begin_of_image|><|image|><|end_of_image|>``;
  3. checkpoint weight prefixes: text weights are stored bare
     (``model.*`` / ``lm_head.*``, exactly the GLM-5.2 checkpoint) instead of
     under ``language_model.*``.

The vision tower and projector always stay unquantized (bf16): the checkpoint
quantization config (NVFP4 modelopt, optionally AQLM-hybrid) only describes
the GLM text Linears.
"""

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo, InputProcessingContext
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5v import Glm5vConfig
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor

from .utils import WeightsMapper, init_vllm_registered_model, maybe_prefix


class Glm5vProcessingInfo(KimiK25ProcessingInfo):
    """Kimi-K2.5 processing with GLM's ``<|image|>`` placeholder."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        # Replicate KimiK25ProcessingInfo.__init__ with the media token
        # swapped from Kimi's <|media_pad|> to GLM's <|image|>.
        BaseProcessingInfo.__init__(self, ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve the token ID from the tokenizer in case it disagrees with
        # config.json (same guard as KimiK25ProcessingInfo).
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            media_token_id = resolved_token_id
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK25Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            media_token_id=media_token_id,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_config(self):
        return self.ctx.get_hf_config(Glm5vConfig)


class Glm5vDummyInputsBuilder(KimiK25DummyInputsBuilder):
    pass


class Glm5vMultiModalProcessor(KimiK25MultiModalProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm5vMultiModalProcessor,
    info=Glm5vProcessingInfo,
    dummy_inputs=Glm5vDummyInputsBuilder,
)
class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    """glm5v: MoonViT vision tower + trained projector + GLM-5.2 text."""

    # IMPORTANT: hf_to_vllm_mapper must stay None at CLASS level.
    # SupportsQuant.__new__ applies a class-level mapper to the quant config
    # (quant_config.apply_vllm_mapper), which would rewrite the hybrid
    # checkpoint's ignore/AQLM layer lists from their bare prod names
    # ("model.layers.N...") to "language_model.model.layers.N..." — but the
    # language model is built with prefix="" (bare prod names), so the
    # rewritten lists would no longer match and e.g. the unquantized dense
    # layers would be built NVFP4-quantized (observed: layer-0 gate_up shape
    # AssertionError at load). The name remap is applied locally in
    # load_weights() instead.
    hf_to_vllm_mapper = None

    _checkpoint_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # GLM-5.2 text weights are stored bare in the checkpoint
            # (identical names to the standalone GLM-5.2 checkpoint).
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            # mm projector legacy naming (Kimi compat; our checkpoint already
            # ships pre_norm/linear_1/linear_2)
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    def load_weights(self, weights):
        from .utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self._checkpoint_to_vllm_mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|begin_of_image|><|image|><|end_of_image|>"
        elif modality == "video":
            # placeholder, to be replaced in the future (video untested)
            return "<|glm5v_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # Replicate KimiK25ForConditionalGeneration.__init__ with the text
        # backbone swapped to GlmMoeDsaForCausalLM (skip the parent __init__,
        # which hardcodes DeepseekV2ForCausalLM).
        import torch.nn as nn

        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config: Glm5vConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        # Vision tower + projector: ALWAYS unquantized bf16 (the checkpoint
        # quant config describes only the GLM text Linears; SGLang reference
        # passes quant_config=None to the tower and builds the projector
        # unquantized).
        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.vision_tower = self.vision_tower.to(
                device=self.device, dtype=model_config.dtype
            )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=None,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            # THE swap: GLM-5.2 (MLA + MoE + DSA sparse attention) backbone.
            #
            # prefix="" (NOT "language_model"): the hybrid checkpoint's
            # quantization_config (nvfp4_aqlm_hybrid) matches layers by their
            # BARE prod names ("model.layers.N...."), and KV-cache/attention
            # layer names must stay byte-identical to the standalone GLM-5.2
            # deployment (LMCache keys, DSA indexer wiring). SGLang's glm5v
            # reference does exactly this (prefix "" unless ModelSlim/Quark).
            # Weight loading is unaffected: AutoWeightsLoader navigates module
            # attributes, and hf_to_vllm_mapper maps checkpoint names to
            # "language_model.*".
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=prefix,
                architectures=["GlmMoeDsaForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # Vision tower / projector are never quantized in glm5v checkpoints.
        return None
