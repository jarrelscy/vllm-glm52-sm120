#!/usr/bin/env bash
# Entrypoint for the GLM-5.2 hybrid SM120 image. Selects a serving topology via
# $PARALLEL and launches vllm serve on :8000.
#
#   PARALLEL=pp4-1m      PP4, NO speculator, 1M window   (default; the full-context config)
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
export VLLM_DISABLE_FP8_W8A16=1   # v2-only NVFP4 MoE kernels (bit-exact)

MODEL_DIR="${MODEL_DIR:-/models/1m}"
PARALLEL="${PARALLEL:-pp4-1m}"
NUM_SPEC="${NUM_SPEC:-5}"

UTIL_DEFAULT=0.95   # per-mode default; env UTIL overrides
DRAFT_TP=""         # non-empty -> draft_tensor_parallel_size in the spec config

case "$PARALLEL" in
  pp4-1m)      # full 1M window, NO speculator (verified: KV 1.27M tokens, ~22 tok/s)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=0; DEFLEN=1048576 ;;
  tp2pp2)      # TP2xPP2 + DSpark, ~223K, ~35 tok/s (best spec-decode single-stream)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=1; DEFLEN=200000 ;;
  tp4-dspark)  # TP4 + DSpark, ~200K, ~59.6 tok/s (fastest decode, no PP bubble)
    PAR="--tensor-parallel-size 4"; SPEC=1; DEFLEN=200000 ;;
  pp4-dspark)  # PP4 + DSpark, ~130K (drafter co-locates on last rank; dominated by tp2pp2)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=131072 ;;
  pp4-mtp)     # PP4 + native MTP self-speculation (checkpoint layer 78, method=mtp).
    # Coherent + lossless. Routed through the V2 model runner (see vllm/config/vllm.py:
    # force-V2 for method in {dspark,mtp}). SHORT-CONTEXT win only: ns=2 ~1.17x over base
    # at 32K; net-NEGATIVE >=~100K because GLM's DSA sparse attention already makes base
    # decode flat ~18-19 tok/s at any length (spec has nothing to amortize) while the MTP
    # draft/verify indexer scans scale with ctx. For long ctx use pp4-1m (no spec).
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=2; DEFLEN=32768; NUM_SPEC="${NUM_SPEC:-2}" ;;
  pp4-tpdraft) # PP4 target (MLA KV split by layer -> long ctx) + DSpark draft sharded
    # TP4 across the PP ranks (draft-TP-over-PP). Draft's 64 KV heads shard 16/rank,
    # freeing the KV that a co-located draft would eat -> ~2x the spec-decode ceiling.
    # Validated: KV 514,697 tokens @ 450K, mean accepted ~3.0, ~17-19 tok/s.
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=450000; UTIL_DEFAULT=0.97; DRAFT_TP=4 ;;
  *) echo "unknown PARALLEL=$PARALLEL (use pp4-1m | tp2pp2 | tp4-dspark | pp4-dspark | pp4-mtp | pp4-tpdraft)"; exit 1 ;;
esac
UTIL="${UTIL:-$UTIL_DEFAULT}"
# NOTE: 1M context and the DSpark drafter cannot co-fit on 4x96GB. The drafter
# needs ~92 GiB KV on its rank vs ~8 GiB available; capping the draft window
# (DRAFT_MAXLEN, below) does NOT free it — vLLM sizes the draft KV at target len.
# For 1M use pp4-1m (no spec); for spec decode use tp4-dspark / tp2pp2 (<=~200K).
MAXLEN="${MAXLEN:-$DEFLEN}"

ARGS=(vllm serve "$MODEL_DIR" $PAR
  --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8_ds_mla
  --max-model-len "$MAXLEN"
  --max-num-seqs 2
  --max-num-batched-tokens 2048
  --no-enable-flashinfer-autotune
  --enforce-eager
  --served-model-name "${SERVED_NAME:-glm-5.2}"
  --port "${PORT:-8001}")
if [ "$SPEC" = 1 ]; then
  SC="{\"model\": \"RedHatAI/GLM-5.2-speculator.dspark\", \"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC"
  [ -n "${DRAFT_MAXLEN:-}" ] && SC="$SC, \"max_model_len\": $DRAFT_MAXLEN"
  [ -n "$DRAFT_TP" ] && SC="$SC, \"draft_tensor_parallel_size\": $DRAFT_TP"
  SC="$SC}"
  ARGS+=(--speculative-config "$SC")
elif [ "$SPEC" = 2 ]; then
  # Native MTP self-speculation (checkpoint layer 78). method=deepseek_mtp -> mtp;
  # routed to V2 runner automatically (config/vllm.py force-V2 for mtp).
  ARGS+=(--speculative-config "{\"method\": \"deepseek_mtp\", \"num_speculative_tokens\": $NUM_SPEC}")
fi

echo ">> GLM-5.2 SM120  PARALLEL=$PARALLEL  MAXLEN=$MAXLEN  util=$UTIL  spec=$SPEC  draft_tp=${DRAFT_TP:-1}  served=${SERVED_NAME:-glm-5.2}  model=$MODEL_DIR"
exec "${ARGS[@]}" "$@"
