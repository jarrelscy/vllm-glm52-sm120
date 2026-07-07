#!/usr/bin/env bash
# SHIP CONFIG — GLM-5.2 PP4 target + DSpark draft Tensor-Parallel over the PP
# ranks (draft-TP-over-PP), long-context speculative decode.
#
#   target: GlmMoeDsaForCausalLM, pipeline_parallel_size=4, tensor_parallel_size=1
#   draft:  RedHatAI/GLM-5.2-speculator.dspark, method=dspark,
#           draft_tensor_parallel_size=4  (TP4 across the 4 PP ranks)
#           -> draft KV shards 64->16 heads/rank (the memory win)
#
# Validated on 4x RTX PRO 6000 (96 GiB): max-model-len 450000 fits with a
# 514,697-token KV cache (14% margin); mean accepted length ~3.0, pos0 ~0.75-0.86,
# ~17-19 tok/s, coherent. ~2x beyond the TP2xPP2 200-223K ceiling.
#
# 1M is NOT feasible on this HW: even sharded, the dense-MHA draft's full-attention
# KV is ~21 GiB/rank at 1M; + target KV (~14) + weights (~78) > 96 GiB. Windowing
# the draft (SWA) would remove that term but that path is block-pool-dead.
#
# Env: MODEL_DIR, MAXLEN (default 450000), GPU_UTIL (default 0.97), NUM_SPEC (5).
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
export VLLM_DISABLE_FP8_W8A16=1

exec vllm serve "$MODEL_DIR" \
  --pipeline-parallel-size 4 \
  --gpu-memory-utilization "${GPU_UTIL:-0.97}" \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len "${MAXLEN:-450000}" \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"model": "RedHatAI/GLM-5.2-speculator.dspark", "method": "dspark", "num_speculative_tokens": '"${NUM_SPEC:-5}"', "draft_tensor_parallel_size": 4}' \
  --enforce-eager \
  --port "${PORT:-8001}"
