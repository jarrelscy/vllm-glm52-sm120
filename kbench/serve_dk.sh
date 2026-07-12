#!/usr/bin/env bash
# DECODE-K dev-loop server boot: production tp4-1m-mtp config with this
# worktree's changed files bind-mounted over the shipped glm52-sm120 image.
# Usage: serve_dk.sh [extra -e flags...]   e.g.
#   serve_dk.sh -e GLM_MOE_DEDUP=1 -e GLM_MOE_LANE_ROWS=1 -e GLM_NVFP4_LUT256=1
# ONLY run while holding the GPU lease (gpulease.sh acquire decode-k N).
set -euo pipefail
WT=/home/jarrelscy/glm52/vllm-decodeK
docker rm -f glm-dk 2>/dev/null || true
docker run -d --name glm-dk --gpus all --ipc host --shm-size 16g \
  --env-file /home/jarrelscy/homeassistant/.env \
  -e PARALLEL="${PARALLEL:-tp4-1m-mtp}" \
  ${MAXLEN:+-e MAXLEN=$MAXLEN} \
  -p 8001:8001 \
  -v /data/huggingface/glm52-models/v2:/models/1m:ro \
  -v "$WT/csrc/quantization/aqlm_moe/aqlm_moe_v2.cu":/opt/vllm/csrc/quantization/aqlm_moe/aqlm_moe_v2.cu:ro \
  -v "$WT/vllm/model_executor/layers/quantization/nvfp4_aqlm_hybrid.py":/opt/vllm/vllm/model_executor/layers/quantization/nvfp4_aqlm_hybrid.py:ro \
  "$@" \
  glm52-sm120:latest
echo "glm-dk started; waiting for :8001 ..."
for i in $(seq 1 240); do
  if curl -s -m 2 http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "READY after ~$((i*15))s"; exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' glm-dk 2>/dev/null)" != "true" ]; then
    echo "CONTAINER DIED"; docker logs --tail 50 glm-dk; exit 1
  fi
  sleep 15
done
echo "TIMEOUT waiting for server"; exit 1
