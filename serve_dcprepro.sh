#!/usr/bin/env bash
# TP4 + DCP4 repro/fix serve for the >=100K decode crash — boots the MAIN tree
# (/home/jarrelscy/glm52/vllm @ cudagraphs-v2, fully built, HAS the DCP fix).
set -euo pipefail
WORK="$HOME/glm52"
MODEL_DIR="${MODEL_DIR:-/data/huggingface/glm52-models/1m}"
INC=/home/jarrelscy/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12

source "$WORK/vllm/.venv/bin/activate"
export CUDA_HOME="$WORK/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CPATH=$INC C_INCLUDE_PATH=$INC FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DISABLE_FP8_W8A16=${VLLM_DISABLE_FP8_W8A16:-1}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

TP="${TP:-4}"; DCP="${DCP:-4}"; DCP_BACKEND="${DCP_BACKEND:-ag_rs}"
GRAPH_FLAGS=(--enforce-eager)
if [ "${CUDAGRAPH:-0}" = "1" ]; then
  GRAPH_FLAGS=(--compilation-config '{"mode": 3, "cudagraph_mode": "'"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"'"}')
fi
SPEC_FLAG=()
[ "${NO_SPEC:-1}" = "1" ] || SPEC_FLAG=(--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": '"${NUM_SPEC:-1}"'}')

exec vllm serve "$MODEL_DIR" \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size "$DCP" \
  --dcp-comm-backend "$DCP_BACKEND" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len "${MAXLEN:-150000}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --no-enable-flashinfer-autotune \
  "${SPEC_FLAG[@]}" \
  "${GRAPH_FLAGS[@]}" \
  --served-model-name glm-5.2 \
  --port "${PORT:-8001}"
