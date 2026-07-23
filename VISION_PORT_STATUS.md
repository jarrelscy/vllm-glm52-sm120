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

## Dir assembly + CPU gates (ALL DONE, commit ab38e5991)

/data/huggingface/glm52-models/v2-vision assembled: symlinks to every v2 file + vision_tower/mm_projector
safetensors + kimi_k25_vision_processing.py/media_utils.py/kimi_k25_processor.py/preprocessor_config.json
(from glm52-vision-head); REAL files authored: config.json (text_config = v2's EXACT dict incl.
nvfp4_aqlm_hybrid quant config, mirrored at top level too), model.safetensors.index.json (5489 v2 + 335
vision = 5824), chat_template.jinja (v2's + image branch; byte-identical to the vision-head template).
NOTE: symlinks are absolute -> containers must ALSO mount -v /data/huggingface:/data/huggingface:ro.
NOTE: worktree vllm/ now carries the compiled *.so + _version.py copied from prod checkout (untracked,
needed for the bind-mount dev loop).

CPU checks (glm52-sm120:latest, no --gpus), ALL PASS:
- get_config -> Glm5vConfig, text glm_moe_dsa 78L/6144/154880, index_topk passthrough 2048, quant nvfp4_aqlm_hybrid
- registry resolves Glm5vForConditionalGeneration; tokenizer <|image|> == 154854
- weight-map dry run: all 5824 names bucket into language_model.model./language_model.lm_head./vision_tower./mm_projector.; MTP layer-78 = 3095 tensors present
- safetensors headers == index for both vision files; projector shapes pre_norm[1152] linear_1[4608,4608] linear_2[6144,4608]
- MoonViT tower + projector instantiated on CPU: param names EXACTLY match checkpoint (329 + 6, zero diff)
- ModelConfig(model=v2-vision): is_deepseek_mla True, use_mla True, is_moe True, quant nvfp4_aqlm_hybrid, multimodal True
- Glm5vMultiModalProcessor.apply(): 64x96 img -> 12 tokens @ correct placeholder range; two-image prompt -> (3,12)+(19,25); dummy profiling 3000x3000 -> 4225 tokens
- Processor round trip: 512x512 -> 361 tokens, KimiK25Processor expansion == calculator
- SpeculativeConfig.hf_config_override(glm5v cfg) -> model_type deepseek_mtp, arch DeepSeekMTPModel, n_predict 1, quant preserved
- chat template: text-only AND tools renders byte-identical to v2's template; image message renders <|begin_of_image|><|image|><|end_of_image|>

## Next: GPU boots (per boot plan below)

## GPU boot results

**Boot A/B (tp4-1m, MAXLEN=65536, no spec, no LMCache) — ALL PASS (2026-07-23 ~08:46)**
- Boot: ~8.5 min total (weights ~5 min, init engine 138.7s). KV cache 1,461,760 tokens @ 18.36 GiB free.
  Multi-modal warmup completed in 1.048s. No weight-loading warnings; no unexpected/missing keys.
- TEXT: "The capital of France is? one word" -> reasoned then "Paris". Text path coherent.
- IMAGE 1 (red square + centered white circle): "background is a solid, vibrant red ... a single, perfectly
  round, white circle ... centered" — EXACT content, no hallucination.
- IMAGE 2 (blue bg + yellow triangle): "The background is blue, and the shape is a yellow triangle."
- TWO IMAGES one request: "Image 1: solid red ... white circle. Image 2: solid blue ... yellow triangle."
  Correct per-image attribution and ordering.
- Correct descriptions prove the trained projector + tower weights loaded and are wired correctly
  (a random projector would produce garbage).

**Boot C (tp4-1m-mtp, MAXLEN=65536, MTP ns3, no LMCache) — ALL PASS (2026-07-23 ~09:00)**
- After the __getattr__ fix: boots clean; KV cache 1,407,488 tokens; "Resolved architecture: DeepSeekMTPModel";
  MTP heads loaded in the draft (never trimmed); V2 model runner handles the multimodal wrapper fine.
- TEXT: Paris (coherent). IMAGE (red/white circle): correct. TWO IMAGES: both correct per-image.
- SpecDecoding metrics: mean acceptance length 3.17-3.30, per-position 0.90/0.73/0.57, avg draft acceptance
  72-77% — in prod envelope (VLLM_MTP_INDEX_SHARE=1 default). 0 ERROR lines in the whole boot log.
- Boot time note: after "Building aqlm_moe extension" the engine goes silent ~5 min (nvcc, shm_broadcast
  60s warnings are normal); total boot ~8 min.

## Boot plan (exact commands — run only after CPU checks pass)

```bash
docker stop homeassistant-vllm-glm5.2-hybrid-1m-mtp-1   # note in this file when done

# Boot A: text-only sanity (no spec, small window, no LMCache)
docker run --rm --name glm5v-dev --gpus all --ipc=host -p 8001:8001 \
  -v /home/jarrelscy/glm52/vllm-vision/vllm:/opt/vllm/vllm \
  -v /data/huggingface/glm52-models/v2-vision:/models/1m:ro \
  -v /data/huggingface:/data/huggingface:ro \
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

1. Boot A #1: ImportError _vllm_fa2_C — bind mount shadows compiled ext subdirs. FIX: rsync ALL missing
   files (vllm_flash_attn/*.so, third_party/deep_gemm/*.so, _version.py) from prod checkout
   /home/jarrelscy/glm52/vllm/vllm into the worktree vllm/ (untracked); delete stale __pycache__.
2. Boot A #2: AssertionError load_merged_column_weight on layer-0 gate_up (first shard). ROOT CAUSE:
   SupportsQuant.__new__ applies the CLASS-level hf_to_vllm_mapper to quant_config
   (nvfp4_aqlm_hybrid -> modelopt apply_vllm_mapper), rewriting the ignore/AQLM lists from bare
   "model.layers.*" to "language_model.model.layers.*" while the language model is built with prefix=""
   (bare names) -> dense layers built quantized. FIX: class hf_to_vllm_mapper=None; name remap applied
   locally in load_weights() via _checkpoint_to_vllm_mapper.
3. Boot C #1 (tp4-1m-mtp): AttributeError 'Glm5vConfig' has no 'num_hidden_layers' in
   DeepSeekMultiTokenPredictor.__init__ — the V2-runner MTP draft path reads text fields off the TARGET's
   top-level hf_config (deepseek_mtp.py uses vllm_config.model_config.hf_config: num_hidden_layers,
   n_group, rms_norm_eps, n_routed_experts, n_shared_experts...). FIX: Glm5vConfig.__getattr__ read-delegates
   missing attrs to text_config (underscore + text_config excluded; explicit DSA properties take precedence).
   Verified MTP draft loader tolerates vision_tower./mm_projector. keys (spec_layer None -> continue).

## GPU state

- 2026-07-23 ~08:40 prod STOPPED (docker stop homeassistant-vllm-glm5.2-hybrid-1m-mtp-1); GPUs taken for glm5v boots. Orchestrator restores prod at the end.
