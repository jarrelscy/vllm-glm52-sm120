#!/usr/bin/env bash
# Inkling-512k-NVFP4-AQLM-hybrid on SM120 (4x RTX PRO 6000, TP4).
# Final speed-campaign configs (task #134, gated 2026-07-19):
#
#   MODE=512k-mtp (default)  524,288 ctx + MTP ns=2 (lossless spec decode)
#                            decode 33-42 prose / 50-53 code / 57-60 counting
#                            tok/s (server-mode), 27.5-39.6 @512K depth
#                            prefill ~1128 tok/s warm at 512K
#   MODE=640k                655,360 ctx, no MTP (longest context)
#                            decode ~40.8 short / 34.5 @640K depth
#                            prefill ~1075 tok/s warm at 640K
#
# Both: FULL_DECODE_ONLY CUDA graphs (capture sizes [1,2,4]), bf16 KV
# (lossless -- fp8 KV excluded by directive), util 0.97, mnbt 2048.
# Runtime knobs: MODEL_DIR, MODE, MAXLEN, NUM_SPEC (512k-mtp only; ns=2 is
# the swept optimum -- larger ns is bimodal on draft-hostile prose).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/inkling}"
MODE="${MODE:-512k-mtp}"

source /opt/vllm/.venv/bin/activate
NV="/opt/vllm/.venv/lib/python3.12/site-packages/nvidia"
CUX="$NV/cu13"; [ -d "$CUX" ] || CUX="$NV/cu12"
export CUDA_HOME="$CUX" PATH="$CUX/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL SHM-bounces on this box (all-NODE PCIe topology, AMD-Vi); driver P2P is
# ~55 GB/s/pair. Routing NCCL over P2P halves prefill RS/AG time (the >64-token
# sconv fallback path): 4K prefill 1802 -> 1912 tok/s, 16K 1822 -> 1927 (+6%),
# 120K fill A/B + coherence + unit suite gated 2026-07-19 (lane A, task #136).
# Transport-only change: ring reduction order is unchanged (numerics-preserving).
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"

case "$MODE" in
  512k-mtp)
    MAXLEN="${MAXLEN:-524288}"
    SPEC_FLAGS=(--speculative-config
      '{"method": "mtp", "num_speculative_tokens": '"${NUM_SPEC:-2}"'}')
    ;;
  640k)
    MAXLEN="${MAXLEN:-655360}"
    SPEC_FLAGS=()
    ;;
  *)
    echo "unknown MODE=$MODE (expected 512k-mtp or 640k)" >&2
    exit 1
    ;;
esac

exec vllm serve "$MODEL_DIR" \
  --served-model-name inkling \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 4 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype auto \
  --no-async-scheduling \
  --trust-remote-code \
  --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]}' \
  "${SPEC_FLAGS[@]}"
