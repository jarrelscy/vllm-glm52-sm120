# GLM-5.2-Vision (glm5v) vLLM port — status

Branch: `vision-graft` (worktree /home/jarrelscy/glm52/vllm-vision, off cudagraphs-v2 e11d95cc9).
Mission: replicate baseten/GLM-5.2-Vision-NVFP4 (MoonViT tower + K2VL projector + GLM-5.2 text) in vLLM
as `Glm5vForConditionalGeneration`, serve with our NVFP4+AQLM hybrid text weights, verify image understanding.

**GPU policy (user directive 2026-07-23): do NOT stop prod / take GPUs until ALL code + dir assembly +
CPU-only sanity checks are done and committed.** Prod container: homeassistant-vllm-glm5.2-hybrid-1m-mtp-1.

## Architecture recon (done)

- vLLM fork has Kimi-K2.5 in-tree: `vllm/model_executor/models/kimi_k25.py` (+ `kimi_k25_vit.py`),
  config `vllm/transformers_utils/configs/kimi_k25.py`, processor `vllm/transformers_utils/processors/kimi_k25.py`.
  kimi_k25 builds its LM via `init_vllm_registered_model(..., architectures=["DeepseekV2ForCausalLM"])` — the
  glm5v swap point is that one string -> `GlmMoeDsaForCausalLM` (deepseek_v2.py).
- Image processor is loaded via `cached_get_image_processor` = AutoImageProcessor **with trust_remote_code**
  (that IS how in-tree kimi_k25 works). The vision-head repo's `kimi_k25_vision_processing.py` API matches
  vLLM's expectations exactly (`media_tokens_calculator`, `num_frames_per_chunk`, `preprocess(vision_chunks)`).
- MLA/DSA gating: `is_deepseek_mla` (model_arch_config_convertor.py) reads `hf_text_config.model_type` ->
  nested text_config works. Backend priorities sm120 = [TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120].
  **GOTCHA**: `FlashInferMLASparseMetadataBuilder` (shared by the SM120 sparse backend) reads
  `vllm_config.model_config.hf_config.index_topk` from the GLOBAL config at runtime (= the wrapper config)
  -> Glm5vConfig has passthrough properties for DSA fields (index_topk etc.).
- MTP: `SpeculativeConfig.hf_config_override` maps model_type glm_moe_dsa -> deepseek_mtp; added glm5v
  text_config promotion at the top (minimax_m3_vl precedent) so the draft maps to the text backbone.
  Draft-side `index_share_for_mtp_iteration` reads draft_model_config.hf_config = promoted text config -> OK.
- v2 tokenizer HAS `<|image|>` = 154854 (verified), plus begin/end_of_image extra special tokens.
- Chat template diff (v2 vs vision): vision adds ONLY an image branch emitting
  `<|begin_of_image|><|image|><|end_of_image|>` for items type image/image_url; everything else identical.
- deepseek_v2 `load_weights` skips spec-layer (78) weights in the target; MTP heads load in the draft. NEVER trim.

## Code changes on vision-graft (done, pending commit)

1. `vllm/transformers_utils/configs/glm5v.py` — Glm5vVisionConfig + Glm5vConfig (mirrors KimiK25Config;
   text_config built via AutoConfig.for_model glm_moe_dsa; propagates quantization_config to top level;
   DSA passthrough properties to text_config: index_topk, index_head_dim, index_n_heads, index_topk_freq,
   index_topk_pattern, index_skip_topk_offset, indexer_types, indexer_rope_interleave,
   index_share_for_mtp_iteration, first_k_dense_replace, kv_lora_rank, q_lora_rank, qk_nope/rope/qk_head_dim,
   v_head_dim, num_nextn_predict_layers).
2. Registered: configs/__init__.py (Glm5vConfig, Glm5vVisionConfig), config.py `_CONFIG_REGISTRY` glm5v=Glm5vConfig.
3. `vllm/model_executor/models/glm5v.py` — Glm5vProcessingInfo (media token `<|image|>`/154854, config class
   Glm5vConfig), Glm5vDummyInputsBuilder, Glm5vMultiModalProcessor (inherit Kimi), Glm5vForConditionalGeneration:
   - hf_to_vllm_mapper: "model."->"language_model.model.", "lm_head."->"language_model.lm_head." (text weights
     stored bare, byte-identical to standalone GLM-5.2), mm_projector.proj.{0,2} legacy remaps kept.
   - get_placeholder_str image -> `<|begin_of_image|><|image|><|end_of_image|>`.
   - __init__ = copy of kimi_k25's with architectures=["GlmMoeDsaForCausalLM"] and tower/projector forced
     quant_config=None (only text Linears are quantized; SGLang reference behavior).
4. registry.py: "Glm5vForConditionalGeneration": ("glm5v", ...).
5. `vllm/config/speculative.py` hf_config_override: glm5v -> promote text_config (+carry quantization_config).

## Next (in order)

1. Assemble /data/huggingface/glm52-models/v2-vision (symlinks to v2 files + vision safetensors; REAL files:
   merged config.json, merged model.safetensors.index.json, chat_template.jinja (v2 + image branch),
   preprocessor_config.json, kimi_k25_vision_processing.py, media_utils.py, kimi_k25_processor.py).
2. CPU-only sanity in glm52-sm120 image (docker run WITHOUT --gpus, bind-mount worktree vllm):
   - import vllm.model_executor.models.glm5v
   - vLLM get_config() on v2-vision -> Glm5vConfig, text_config glm_moe_dsa, quant config present
   - tokenizer id 154854; index keys == union(v2 keys, 335 vision keys); loader name-mapping dry-run
   - multimodal processor instantiation + dummy prompt expansion if feasible
3. Commit. THEN stop prod, boot text-only (PARALLEL=tp4-1m, MAXLEN 65536, no spec, ENABLE_LMCACHE=0),
   then vision requests, then step up (MTP -> full 950K -> LMCACHE).

## Boot plan (exact commands — run only after CPU checks pass)

```bash
docker stop homeassistant-vllm-glm5.2-hybrid-1m-mtp-1   # note in this file when done

# Boot A: text-only sanity (no spec, small window, no LMCache)
docker run --rm --name glm5v-dev --gpus all --ipc=host -p 8001:8001 \
  -v /home/jarrelscy/glm52/vllm-vision/vllm:/opt/vllm/vllm \
  -v /data/huggingface/glm52-models/v2-vision:/models/1m:ro \
  -e PARALLEL=tp4-1m -e MAXLEN=65536 -e ENABLE_LMCACHE=0 \
  glm52-sm120:latest --trust-remote-code
# expect: Glm5vForConditionalGeneration resolved, vision_tower/mm_projector weights loaded (not "unexpected"),
# server ready; curl /v1/chat/completions "The capital of France is" -> Paris.

# Boot B: same + image requests (PIL-generated red square w/ white circle; 2-image request too)

# Boot C: step up: PARALLEL=tp4-1m-mtp -e MAXLEN=65536 (MTP draft promotion path exercised)
# Boot D: full prod: PARALLEL=tp4-1m-mtp MAXLEN=950000, then ENABLE_LMCACHE=1
```

API key: if 401, read from prod compose env into a shell var; never print/commit.

## Failures + hypotheses

(none yet)
