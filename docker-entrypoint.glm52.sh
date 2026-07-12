#!/usr/bin/env bash
# Entrypoint for the GLM-5.2 hybrid SM120 image. Selects a serving topology via
# $PARALLEL and launches vllm serve on :8001.
#
#   PARALLEL=tp4-1m-mtp  TP4+DCP4+MTP ns3 + PIECEWISE graphs + glm47 tools  (DEFAULT; coherent ~1M + lossless spec, ~57 tok/s @32K)
#   PARALLEL=pp4-1m      PP4, NO speculator, 1M window   (full-context, no spec)
#   PARALLEL=pp4-dspark  PP4 + DSpark, ~256K             (spec decode, reduced context)
#   PARALLEL=tp2pp2      TP2xPP2 + DSpark, ~200K         (spec decode, fastest single-stream decode)
#   PARALLEL=pp4-mtp     PP4 + native MTP self-spec, ~32K   (coherent+lossless; short-ctx ~1.17x @ns=2, net-neg >=100K)
#   PARALLEL=pp4-tpdraft PP4 target + DSpark draft sharded TP4, ~450K  (draft-TP-over-PP, longest spec-decode ctx)
#
# WHY 1M and DSpark are separate modes: on 4x96GB, the 754B hybrid weights (~272 GiB)
# plus a 1M-sized KV cache already fill VRAM. The DSpark drafter (its own embed/
# lm_head/layers + a 1M activation reservation) leaves too little KV for a 1M
# sequence, so speculative decode is only available at capped context.
set -euo pipefail
cd /opt/vllm && source .venv/bin/activate

# Python.h (JIT) + CUDA home (nvidia pip cu13/cu12) discovery
PYINC_DIR=$(find /root/.local/share/uv/python -maxdepth 4 -type d -path "*/include/python3.12" 2>/dev/null | head -1)
[ -n "${PYINC_DIR:-}" ] && export CPATH="$PYINC_DIR" C_INCLUDE_PATH="$PYINC_DIR"
# nvidia is a namespace package (nvidia.__file__ is None) -> use __path__.
NV=$(python -c "import nvidia;print(nvidia.__path__[0])" 2>/dev/null || true)
for c in cu13 cu12; do [ -n "${NV:-}" ] && [ -d "$NV/$c" ] && export CUDA_HOME="$NV/$c" && break; done

export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DISABLE_FP8_W8A16="${VLLM_DISABLE_FP8_W8A16:-1}"   # v2-only (bit-exact) default; set =0 to opt into v4 fp8 W8A16 (+6-8% base decode)

MODEL_DIR="${MODEL_DIR:-/models/1m}"
PARALLEL="${PARALLEL:-pp4-1m}"
NUM_SPEC_ENV="${NUM_SPEC:-}"   # user override only; per-mode default applied AFTER the case

UTIL_DEFAULT=0.95   # per-mode default; env UTIL overrides
DRAFT_TP="${DRAFT_TP:-}"   # non-empty -> draft_tensor_parallel_size in the spec config (env-overridable for speed hunt)
DCP=""              # non-empty -> --decode-context-parallel-size (shard MLA KV across TP ranks -> ~1M at TP4)
CGMODE=""           # per-mode cudagraph_mode override (empty -> FULL_AND_PIECEWISE); DCP+spec MUST use PIECEWISE
NSDEF=""            # per-mode spec-token default override (empty -> MTP=2 / DSpark=5); env NUM_SPEC always wins

case "$PARALLEL" in
  pp4-1m)      # full 1M window, NO speculator (verified: KV 1.27M tokens, ~22 tok/s)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=0; DEFLEN=1048576 ;;
  tp2pp2)      # TP2xPP2 + DSpark, ~223K, ~35 tok/s (best spec-decode single-stream)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=1; DEFLEN=200000 ;;
  tp4-dspark)  # TP4 + DSpark, ~200K, ~59.6 tok/s (fastest decode, no PP bubble)
    PAR="--tensor-parallel-size 4"; SPEC=1; DEFLEN=200000 ;;
  tp4-1m)      # TP4 + Decode Context Parallelism (DCP4) -> ~1M at TP speed, NO spec.
    # DCP sequence-shards the MLA latent KV across the 4 TP ranks (TP4 alone caps ~371K
    # because MLA KV replicates); reaches the same ~1.13M KV ceiling as PP4 but at TP speed
    # with no pipeline bubble. VALIDATED: KV 1,128,940 tok @ 950K; needle @ 749,035 tok
    # depth-0.5 = PASS (coherent at depth). DECODE with graphs (default-on for TP) = 24.9
    # tok/s @ ~700K — BEATS PP4-base@1M (18.7) by ~33%; eager is only 10.8 (DCP LSE
    # all-gather over PCIe is comm-bound -> graphs are ESSENTIAL here, they capture the
    # ag_rs comms). Needs the SM120 return-LSE + decode-out-size fix (on this branch).
    PAR="--tensor-parallel-size 4"; SPEC=0; DEFLEN=950000; DCP=4 ;;
  tp4-1m-mtp)  # TP4 + DCP4 + native MTP (ns3) -> coherent ~1M WITH lossless spec at TP speed.
    # THE default config. DCP shards KV (fits ~1M) + MTP spec verifies k+1 tok/step.
    # 2026-07-12 promoted stack (all 64K-golden/1M/acceptance gated, lossless):
    #   - VLLM_MTP_INDEX_SHARE=1: draft steps reuse the verify-anchored DSA top-k
    #     (fixes bad in-loop draft top-k; count accept 0.43->0.96; +24-47% count)
    #   - GLM_MOE_LANE_ROWS=1 + GLM_NVFP4_LUT256=1: bit-exact gemv lane repack (w2 -32%)
    #     + smem LUT for fp4 cvt (NV slice -26%); server +7.3% count @123K
    #   - DCP_BACKEND=ag_rs + NCCL_P2P_LEVEL=SYS: NCCL was SHM-bouncing (topology NODE);
    #     forced P2P needs ag_rs (a2a-over-P2P is pathological). +3-5% decode, +35% prefill
    #   - chunk 4096 + util 0.97: prefill +13%, still boots the 950K window (KV ~1.006M)
    # MEASURED (this stack): short 73/64/52 tok/s, @123K 44/41/32; fresh prefill ~1.2K tok/s.
    # PIECEWISE cudagraphs remain the default (FULL_AND_PIECEWISE now boots under ag_rs —
    # +2-9% more — but is opt-in pending soak: CUDAGRAPH_MODE=FULL_AND_PIECEWISE).
    # Every knob env-overridable; set VLLM_MTP_INDEX_SHARE=0 etc. to peel back.
    export VLLM_MTP_INDEX_SHARE="${VLLM_MTP_INDEX_SHARE:-1}"
    export GLM_MOE_LANE_ROWS="${GLM_MOE_LANE_ROWS:-1}"
    export GLM_NVFP4_LUT256="${GLM_NVFP4_LUT256:-1}"
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    DCP_BACKEND="${DCP_BACKEND:-ag_rs}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    PAR="--tensor-parallel-size 4"; SPEC=2; DEFLEN=950000; DCP=4; CGMODE=PIECEWISE; NSDEF=3; UTIL_DEFAULT=0.97 ;;
  pp4-dspark)  # PP4 + DSpark, ~130K (drafter co-locates on last rank; dominated by tp2pp2)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=131072 ;;
  pp4-mtp)     # PP4 + native MTP self-speculation (checkpoint layer 78, method=mtp), ~1M ctx.
    # Coherent + lossless. Routed through the V2 model runner (vllm/config/vllm.py force-V2 for
    # method in {dspark,mtp}). FITS ~1M (est ceiling 999,168 tok @ util 0.97; default 950K for
    # margin) — MTP's draft KV is MLA-shaped & tiny, unlike DSpark. PP split MUST be 20,20,20,18
    # (NOT 21,19,19,19): the MTP layer is a full 256-expert MoE (~7.5 GiB) that lands on the LAST
    # rank, so that rank carries fewer target layers or context caps. PERF (non-streamed, CORRECT
    # counting): MTP BEATS base on every workload at ns=2 — 1m @32K [easy 29.3/code 24.4/complex
    # 24.0] vs base 18.7 (1.26-1.57x); still wins @200K (easy 24.6 vs 18.4). ns=2 best all-round.
    # (Earlier "MTP loses at long ctx" was a STREAMED-measurement artifact — SSE bundles accepted
    # tokens; see BENCHMARKING.md.) Short-ctx latency: set MAXLEN=32768.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-20,20,20,18}"
    PAR="--pipeline-parallel-size 4"; SPEC=2; DEFLEN=950000; UTIL_DEFAULT=0.97 ;;
  tp2pp2-mtp)  # TP2xPP2 + native MTP self-speculation, ~580K ctx — the SPEED/CONTEXT BALANCE pick.
    # Non-streamed: 1m [easy 40.4/code 36.1/complex 30.1] tok/s @ ceiling ~580K — ~2x pp4-mtp's
    # decode while still reaching >0.5M context. Best all-round serving config.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=2; DEFLEN=560000 ;;
  pp4-tpdraft) # PP4 target (MLA KV split by layer -> long ctx) + DSpark draft sharded
    # TP4 across the PP ranks (draft-TP-over-PP). Draft's 64 KV heads shard 16/rank,
    # freeing the KV that a co-located draft would eat -> ~2x the spec-decode ceiling.
    # Validated: KV 514,697 tokens @ 450K, mean accepted ~3.0, ~17-19 tok/s.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=450000; UTIL_DEFAULT=0.97; DRAFT_TP=4 ;;
  *) echo "unknown PARALLEL=$PARALLEL (use pp4-1m | pp4-mtp | tp2pp2 | tp2pp2-mtp | tp4-dspark | tp4-1m | tp4-1m-mtp | pp4-dspark | pp4-tpdraft)"; exit 1 ;;
esac
UTIL="${UTIL:-$UTIL_DEFAULT}"
# per-mode spec-token default: MTP=2 (matrix optimum, best all-round), DSpark=5; env NUM_SPEC overrides
if [ "$SPEC" = 2 ]; then NUM_SPEC="${NUM_SPEC_ENV:-${NSDEF:-2}}"; else NUM_SPEC="${NUM_SPEC_ENV:-${NSDEF:-5}}"; fi
# NOTE: 1M context and the DSpark drafter cannot co-fit on 4x96GB. The drafter
# needs ~92 GiB KV on its rank vs ~8 GiB available; capping the draft window
# (DRAFT_MAXLEN, below) does NOT free it — vLLM sizes the draft KV at target len.
# For 1M use pp4-1m (no spec); for spec decode use tp4-dspark / tp2pp2 (<=~200K).
MAXLEN="${MAXLEN:-$DEFLEN}"

# CUDA graphs (V2 hybrid kernel is graph-safe via torch.library custom ops).
# FULL_AND_PIECEWISE = one full-graph replay per decode forward (plain PIECEWISE is
# net-NEGATIVE — per-segment replay loop — do NOT use it).
# DEFAULT is per-mode: ON for TP configs, OFF for pure-PP. Graphs amortize TP's
# all-reduce/launch overhead spectacularly (measured @32K: tp4-dspark +25% ->74.6,
# tp2pp2 +54%, tp4-base +180%, tp2pp2-mtp +16-22%) but do ~nothing for pure-PP
# (bubble-bound: pp4 +0-2%). Override with CUDAGRAPH=0/1.
case "$PAR" in *tensor-parallel-size*) CG_DEFAULT=1 ;; *) CG_DEFAULT=0 ;; esac
CUDAGRAPH="${CUDAGRAPH:-$CG_DEFAULT}"
GRAPH_FLAGS=(--enforce-eager)
if [ "$CUDAGRAPH" = 1 ]; then
  CC="{\"mode\": 3, \"cudagraph_mode\": \"${CUDAGRAPH_MODE:-${CGMODE:-FULL_AND_PIECEWISE}}\""
  # SPEED HUNT (#14/#18): custom cudagraph capture sizes (e.g. CAP_SIZES="1,2,4,6,8,16"
  # to give ns=5 an exact graph-6 verify instead of padded graph-8).
  [ -n "${CAP_SIZES:-}" ] && CC="$CC, \"cudagraph_capture_sizes\": [${CAP_SIZES}]"
  CC="$CC}"
  GRAPH_FLAGS=(--compilation-config "$CC")
fi

ARGS=(vllm serve "$MODEL_DIR" $PAR
  --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8_ds_mla
  --max-model-len "$MAXLEN"
  --max-num-seqs "${MAX_NUM_SEQS:-2}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --no-enable-flashinfer-autotune
  "${GRAPH_FLAGS[@]}"
  --served-model-name "${SERVED_NAME:-glm-5.2}"
  --port "${PORT:-8001}")
# Function/tool calling (GLM-5.2 = GLM-4.7 lineage -> glm47_moe parser; chat template
# ships <tool_call> tags). Default on; set ENABLE_TOOLS=0 to disable, TOOL_PARSER to override.
if [ "${ENABLE_TOOLS:-1}" = 1 ]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-glm47}")
fi
[ -n "$DCP" ] && ARGS+=(--decode-context-parallel-size "$DCP" --dcp-comm-backend "${DCP_BACKEND:-a2a}")
if [ "$SPEC" = 1 ]; then
  SC="{\"model\": \"RedHatAI/GLM-5.2-speculator.dspark\", \"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC"
  [ -n "${DRAFT_MAXLEN:-}" ] && SC="$SC, \"max_model_len\": $DRAFT_MAXLEN"
  [ -n "$DRAFT_TP" ] && SC="$SC, \"draft_tensor_parallel_size\": $DRAFT_TP"
  SC="$SC}"
  ARGS+=(--speculative-config "$SC")
elif [ "$SPEC" = 2 ]; then
  # Native MTP self-speculation (checkpoint layer 78). method=deepseek_mtp -> mtp;
  # routed to V2 runner automatically (config/vllm.py force-V2 for mtp).
  SC="{\"method\": \"deepseek_mtp\", \"num_speculative_tokens\": $NUM_SPEC"
  # SPEED HUNT: draft-side knobs (all lossless by construction — rejection sampling).
  [ -n "${DRAFT_TP:-}" ] && SC="$SC, \"draft_tensor_parallel_size\": $DRAFT_TP"
  [ -n "${LOCAL_ARGMAX:-}" ] && SC="$SC, \"use_local_argmax_reduction\": true"
  SC="$SC}"
  ARGS+=(--speculative-config "$SC")
fi

echo ">> GLM-5.2 SM120  PARALLEL=$PARALLEL  MAXLEN=$MAXLEN  util=$UTIL  graph=$CUDAGRAPH  spec=$SPEC  draft_tp=${DRAFT_TP:-1}  dcp=${DCP:-1}  served=${SERVED_NAME:-glm-5.2}  model=$MODEL_DIR"
exec "${ARGS[@]}" "$@"
