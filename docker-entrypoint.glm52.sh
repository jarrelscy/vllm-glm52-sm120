#!/usr/bin/env bash
# Entrypoint for the GLM-5.2 hybrid SM120 image. Selects a serving topology via
# $PARALLEL and launches vllm serve on :8000.
#
#   PARALLEL=pp4-1m     PP4, NO speculator, 1M window   (default; the full-context config)
#   PARALLEL=pp4-dspark PP4 + DSpark, ~256K             (spec decode, reduced context)
#   PARALLEL=tp2pp2     TP2xPP2 + DSpark, ~200K         (spec decode, fastest single-stream decode)
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
NV=$(python -c "import os,nvidia;print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
for c in cu13 cu12; do [ -n "${NV:-}" ] && [ -d "$NV/$c" ] && export CUDA_HOME="$NV/$c" && break; done

export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DISABLE_FP8_W8A16=1   # v2-only NVFP4 MoE kernels (bit-exact)

MODEL_DIR="${MODEL_DIR:-/models/1m}"
PARALLEL="${PARALLEL:-pp4-1m}"
NUM_SPEC="${NUM_SPEC:-5}"

case "$PARALLEL" in
  pp4-1m)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=0; DEFLEN=1048576 ;;
  pp4-dspark)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
    PAR="--pipeline-parallel-size 4"; SPEC=1; DEFLEN=262144 ;;
  tp2pp2)
    export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-39,39}"
    PAR="--tensor-parallel-size 2 --pipeline-parallel-size 2"; SPEC=1; DEFLEN=200000 ;;
  *) echo "unknown PARALLEL=$PARALLEL (use pp4-1m | pp4-dspark | tp2pp2)"; exit 1 ;;
esac
MAXLEN="${MAXLEN:-$DEFLEN}"

ARGS=(vllm serve "$MODEL_DIR" $PAR
  --gpu-memory-utilization 0.95
  --kv-cache-dtype fp8_ds_mla
  --max-model-len "$MAXLEN"
  --max-num-seqs 2
  --max-num-batched-tokens 2048
  --no-enable-flashinfer-autotune
  --enforce-eager
  --port 8000)
[ "$SPEC" = 1 ] && ARGS+=(--speculative-config "{\"model\": \"RedHatAI/GLM-5.2-speculator.dspark\", \"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC}")

echo ">> GLM-5.2 SM120  PARALLEL=$PARALLEL  MAXLEN=$MAXLEN  spec=$SPEC  model=$MODEL_DIR"
exec "${ARGS[@]}"
