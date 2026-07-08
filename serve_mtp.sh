#!/usr/bin/env bash
# GLM-5.2 PP4 + MTP (deepseek_mtp) self-speculative decode — coherence bring-up.
# MTP = checkpoint layer 78 (num_nextn_predict_layers=1), MLA attention.
# Small max-model-len for fast boot/iteration; raise once coherent.
set -euo pipefail
WORK="$HOME/glm52"
MODEL_DIR="${MODEL_DIR:-/data/huggingface/glm52-models/1m}"
INC=/home/jarrelscy/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12

source "$WORK/vllm/.venv/bin/activate"
export CUDA_HOME="$WORK/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CPATH=$INC C_INCLUDE_PATH=$INC FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-21,19,19,19}"
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DISABLE_FP8_W8A16=${VLLM_DISABLE_FP8_W8A16:-1}

ASYNC_FLAG=""
[ "${SYNC:-0}" = "1" ] && ASYNC_FLAG="--no-async-scheduling"

# CUDA-graph toggle. Default = safe eager fallback (enforce-eager).
# CUDAGRAPH=1 -> drop enforce-eager and enable compiled cudagraphs for the V2
# hybrid kernel path. Default mode FULL_AND_PIECEWISE (FULL decode graph = one
# replay/forward; the sparse-MLA-SM120 backend supports it). Do NOT use plain
# PIECEWISE: on this PP4/batch-1 latency-bound decode its per-segment replay
# loop is net-NEGATIVE (measured -15..-20% for MTP). See CUDAGRAPH_PROGRESS.md.
GRAPH_FLAGS=(--enforce-eager)
if [ "${CUDAGRAPH:-0}" = "1" ]; then
  CGMODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
  GRAPH_FLAGS=(--compilation-config '{"mode": 3, "cudagraph_mode": "'"$CGMODE"'"}')
fi

SPEC_FLAG=(--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": '"${NUM_SPEC:-1}"'}')
[ "${NO_SPEC:-0}" = "1" ] && SPEC_FLAG=()
[ "${DSPARK:-0}" = "1" ] && SPEC_FLAG=(--speculative-config '{"model": "RedHatAI/GLM-5.2-speculator.dspark", "method": "dspark", "num_speculative_tokens": '"${NUM_SPEC:-5}"'}')

# PARALLEL_ARGS overrides the topology (default PP4). e.g. "--tensor-parallel-size 2 --pipeline-parallel-size 2"
read -r -a PAR_ARR <<< "${PARALLEL_ARGS:---pipeline-parallel-size 4}"

exec vllm serve "$MODEL_DIR" \
  "${PAR_ARR[@]}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len "${MAXLEN:-32768}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --no-enable-flashinfer-autotune \
  "${SPEC_FLAG[@]}" \
  "${GRAPH_FLAGS[@]}" \
  $ASYNC_FLAG \
  --served-model-name glm-5.2 \
  --port "${PORT:-8001}"
